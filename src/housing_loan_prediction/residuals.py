"""Analyse des erreurs du modèle final (GES) à partir des prédictions hors pli.

Lancer avec : uv run python -u -m housing_loan_prediction.residuals
(nécessite les prédictions hors pli de `oof/`, voir `stacking oof`)

Pourquoi des groupes : un dossier seul ne dit pas si le modèle se trompe (prédire 0,90 et tomber sur un
refus arrive une fois sur dix). On compare donc, sur des groupes de dossiers, la probabilité moyenne
prédite au taux d'acceptation réel. Un écart qui se répète sur tout un groupe est une erreur systématique.

Garde-fous :
- probabilités honnêtes : pour chaque dossier, poids de la GES appris sans ce dossier (CV au niveau méta,
  mêmes plis que partout) ; les modèles de base sont eux-mêmes hors pli ;
- liste de groupes fixée avant de regarder (define_groups) ;
- un groupe n'est signalé que s'il a au moins MIN_N dossiers et un écart significatif après correction de
  Bonferroni (autant de tests que de modalités examinées).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from housing_loan_prediction.data import load_clean
from housing_loan_prediction.preprocessing import ID_COL
from housing_loan_prediction.stacking import fit_ges, load_predictions, meta_oof

MIN_N = 200
ALPHA = 0.05
MIN_CLASS = 20  # AUC interne calculée seulement si le groupe a au moins 20 acceptés et 20 refusés
EXEMPT = "1111"
DTI_MIDDLE = {str(v): "36%-49%" for v in range(36, 50)}


def define_groups(df):
    """Groupes fixés à l'avance : nom lisible -> modalité de chaque dossier."""
    exempt_cols = [c for c in df.columns if df[c].dtype == object and (df[c] == EXEMPT).any()]
    yes_no = {0: "non", 1: "oui"}
    return {
        "taille du prêt (quintile)": pd.qcut(df["loan_amount"], 5, labels=["Q1 petits", "Q2", "Q3", "Q4", "Q5 gros"]),
        "type de prêt": df["loan_type"].map({"1": "conventionnel", "2": "FHA", "3": "VA", "4": "USDA"}),
        "rang de l'hypothèque": df["lien_status"].map({"1": "premier rang", "2": "second rang"}),
        "occupation": df["occupancy_type"].map({"1": "résidence principale", "2": "secondaire", "3": "investissement"}),
        "type de logement": df["derived_dwelling_category"],
        "DTI (endettement)": df["debt_to_income_ratio"].replace(DTI_MIDDLE),
        "évaluation automatique (aus-1)": df["aus-1"],
        "type de score de crédit": df["applicant_credit_score_type"],
        "organisme payeur initial": df["initially_payable_to_institution"],
        "dossier Exempt (un code 1111)": (df[exempt_cols] == EXEMPT).any(axis=1).astype(int).map(yes_no),
        "LTV manquant": df["ltv_missing"].map(yes_no),
        "revenu manquant ou ≤ 0": df["income_missing"].map(yes_no),
        "valeur du bien manquante": df["property_value_missing"].map(yes_no),
        "quartier inconnu": df["tract_unknown"].map(yes_no),
        "co-emprunteur": df["has_coapplicant"].map(yes_no),
        "État": df["state_code"],
    }


def internal_auc(y, p):
    """AUC à l'intérieur d'un groupe (NaN si trop peu d'acceptés ou de refusés)."""
    return roc_auc_score(y, p) if min(y.sum(), len(y) - y.sum()) >= MIN_CLASS else np.nan


def group_table(groups, y, p):
    """Par modalité : effectif, taux réel, proba moyenne, écart (réel - prédit) et son z, AUC interne."""
    resid = y - p
    rows = []
    for gname, labels in groups.items():
        for label, idx in pd.Series(np.arange(len(y))).groupby(labels.to_numpy(), observed=True):
            idx = idx.to_numpy()
            n = len(idx)
            gap = resid[idx].mean()
            se = resid[idx].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            rows.append({"groupe": gname, "modalité": label, "n": n, "taux réel": y[idx].mean(),
                         "proba moyenne": p[idx].mean(), "écart": gap, "z": gap / se if se else np.nan,
                         "AUC interne": internal_auc(y[idx], p[idx])})
    return pd.DataFrame(rows)


def calibration_table(y, p, bins=10):
    """Dossiers rangés en 10 tranches de même effectif selon la proba prédite : proba moyenne vs taux réel."""
    deciles = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "d": deciles}).groupby("d").agg(
        n=("y", "size"), proba_moyenne=("p", "mean"), taux_reel=("y", "mean"))
    t["écart"] = t["taux_reel"] - t["proba_moyenne"]
    return t


def base_models_on(rows, groups, P, y):
    """Pour chaque groupe : AUC interne de chaque modèle de base, pour voir si l'un d'eux classe mieux."""
    for _, row in rows.iterrows():
        idx = np.flatnonzero((groups[row["groupe"]] == row["modalité"]).to_numpy())
        aucs = "  ".join(f"{m} {internal_auc(y[idx], P[m].to_numpy()[idx]):.3f}" for m in P.columns)
        print(f"    {row['groupe']} = {row['modalité']} (n = {len(idx)}) : GES {row['AUC interne']:.3f} | {aucs}")


if __name__ == "__main__":
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    P = load_predictions(0, train_clean[ID_COL])
    p = meta_oof(P, y, fit_ges)
    print(f"AUC hors pli de la GES (poids appris sur 4 plis, notés sur le 5e) : {roc_auc_score(y, p):.4f}")

    print("\n=== 1) Calibration : dossiers rangés en 10 tranches de proba prédite ===")
    print(calibration_table(y, p).round(3).to_string())

    groups = define_groups(train_clean)
    table = group_table(groups, y, p)
    z_crit = norm.ppf(1 - ALPHA / (2 * len(table)))
    cols = ["groupe", "modalité", "n", "taux réel", "proba moyenne", "écart", "z", "AUC interne"]

    print(f"\n=== 2) Erreur systématique par groupe ({len(table)} modalités, |z| > {z_crit:.2f}, n >= {MIN_N}) ===")
    flagged = table[(table["n"] >= MIN_N) & (table["z"].abs() > z_crit)]
    if flagged.empty:
        print("  Aucun groupe ne s'écarte significativement.")
    else:
        print(flagged.sort_values("z", key=abs, ascending=False)[cols].round(3).to_string(index=False))

    print("\n=== 3) Groupes les moins bien classés (n >= 1000), et AUC interne de chaque modèle de base ===")
    low = table[table["n"] >= 1000].nsmallest(8, "AUC interne")
    print(low[cols].round(3).to_string(index=False))
    print()
    base_models_on(low, groups, P, y)

    print("\n=== 4) Refus prédits avec confiance ===")
    n_ref = (y == 0).sum()
    n_conf = ((y == 0) & (p > 0.95)).sum()
    print(f"  refusés avec une proba d'accord > 0,95 : {n_conf} sur {n_ref} refus ({n_conf / n_ref:.1%})")
