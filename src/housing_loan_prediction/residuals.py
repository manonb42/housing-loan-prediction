"""Analyse des résidus du modèle final (stacking logistique) à partir des prédictions hors pli.

Lancer avec : uv run python -u -m housing_loan_prediction.residuals

Pourquoi des groupes : un dossier seul ne dit pas si le modèle se trompe (prédire 0,90 et tomber sur un
refus arrive une fois sur dix). On compare donc, sur des groupes de dossiers, la probabilité moyenne
prédite au taux d'acceptation réel. Un écart qui se répète sur tout un groupe est une erreur systématique.

Garde-fous :
- probabilités honnêtes : pour chaque dossier, stacking appris sans ce dossier (CV au niveau méta, mêmes
  plis que partout) ; les modèles de base sont eux-mêmes hors pli ;
- liste de groupes fixée avant de regarder (define_groups) ;
- un groupe n'est signalé que s'il a au moins MIN_N dossiers et un écart significatif après correction de
  Bonferroni (autant de tests que de modalités examinées) ;
- toute correction suggérée devra être revalidée en CV avant d'être retenue.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from housing_loan_prediction.boosting_libs import OOF_DIR
from housing_loan_prediction.data import load_clean
from housing_loan_prediction.preprocessing import ID_COL
from housing_loan_prediction.stacking import fit_ges, fit_logistic, load_predictions, meta_oof

MIN_N = 200
ALPHA = 0.05
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


def log_loss_terms(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def group_table(groups, y, p):
    """Par modalité : effectif, taux réel, proba moyenne, écart (réel - prédit) et son z, AUC interne."""
    resid, ll = y - p, log_loss_terms(y, p)
    rows = []
    for gname, labels in groups.items():
        for label, idx in pd.Series(np.arange(len(y))).groupby(labels.to_numpy(), observed=True):
            idx = idx.to_numpy()
            n = len(idx)
            gap = resid[idx].mean()
            se = resid[idx].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            yy = y[idx]
            auc = roc_auc_score(yy, p[idx]) if min(yy.sum(), n - yy.sum()) >= 20 else np.nan
            rows.append({"groupe": gname, "modalité": label, "n": n, "taux réel": yy.mean(),
                         "proba moyenne": p[idx].mean(), "écart": gap, "z": gap / se if se else np.nan,
                         "AUC interne": auc, "part log-perte": ll[idx].sum() / ll.sum()})
    return pd.DataFrame(rows)


def calibration_table(y, p, bins=10):
    deciles = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "d": deciles}).groupby("d").agg(
        n=("y", "size"), proba_moyenne=("p", "mean"), taux_reel=("y", "mean"))
    t["écart"] = t["taux_reel"] - t["proba_moyenne"]
    return t


def base_models_on(flagged, groups, P, y):
    """Pour chaque groupe signalé : proba moyenne, écart et AUC interne de chaque modèle de base."""
    for _, row in flagged.iterrows():
        idx = np.flatnonzero((groups[row["groupe"]] == row["modalité"]).to_numpy())
        yy = y[idx]
        print(f"\n  {row['groupe']} = {row['modalité']} (n = {len(idx)}, taux réel {yy.mean():.3f}) :")
        for model in P.columns:
            pm = P[model].to_numpy()[idx]
            auc = roc_auc_score(yy, pm) if min(yy.sum(), len(yy) - yy.sum()) >= 20 else np.nan
            print(f"    {model:<22} proba moyenne {pm.mean():.3f}  écart {yy.mean() - pm.mean():+.3f}  "
                  f"AUC interne {auc:.3f}")


if __name__ == "__main__":
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    P = load_predictions(0, train_clean[ID_COL])

    p_stack = expit(meta_oof(P, y, fit_logistic))
    p_ges = meta_oof(P, y, fit_ges)
    print(f"AUC honnête (méta hors pli) : stacking logistique {roc_auc_score(y, p_stack):.4f}, "
          f"GES {roc_auc_score(y, p_ges):.4f}")
    pd.DataFrame({ID_COL: train_clean[ID_COL], "accepted": p_stack}).to_csv(
        OOF_DIR / "stacking_logistique_meta.csv", index=False)

    print("\n=== 1) Calibration globale : dossiers rangés en 10 tranches de proba prédite ===")
    print(calibration_table(y, p_stack).round(3).to_string())

    groups = define_groups(train_clean)
    table = group_table(groups, y, p_stack)
    n_tests = len(table)
    z_crit = norm.ppf(1 - ALPHA / (2 * n_tests))
    table.to_csv(OOF_DIR / "residus_par_groupe.csv", index=False)

    print(f"\n=== 2) Écart par groupe ({n_tests} modalités testées, seuil |z| > {z_crit:.2f}, n >= {MIN_N}) ===")
    flagged = table[(table["n"] >= MIN_N) & (table["z"].abs() > z_crit)].sort_values("z", key=abs, ascending=False)
    cols = ["groupe", "modalité", "n", "taux réel", "proba moyenne", "écart", "z", "AUC interne", "part log-perte"]
    if flagged.empty:
        print("  Aucun groupe ne s'écarte significativement.")
    else:
        print(flagged[cols].round(3).to_string(index=False))

    print("\n=== 3) AUC interne la plus basse (n >= 1000) : groupes où le modèle classe mal ===")
    low = table[(table["n"] >= 1000)].nsmallest(8, "AUC interne")
    print(low[cols].round(3).to_string(index=False))

    print("\n=== 4) Où se concentrent les grosses erreurs (part de la log-perte > 2x la part des dossiers) ===")
    heavy = table[(table["n"] >= MIN_N) & (table["part log-perte"] > 2 * table["n"] / len(y))]
    print(heavy.sort_values("part log-perte", ascending=False)[cols].head(10).round(3).to_string(index=False))

    print("\n=== 5) Erreurs très confiantes ===")
    print(f"  refusés prédits > 0,95 : {((y == 0) & (p_stack > 0.95)).sum()} dossiers "
          f"(sur {(y == 0).sum()} refus)")
    print(f"  acceptés prédits < 0,05 : {((y == 1) & (p_stack < 0.05)).sum()} dossiers "
          f"(sur {(y == 1).sum()} acceptations)")

    if not flagged.empty:
        print("\n=== 6) Les 5 modèles de base sur les groupes signalés ===")
        base_models_on(flagged.head(8), groups, P, y)
