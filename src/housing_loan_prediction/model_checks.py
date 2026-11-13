"""Contrôles pour justifier le choix du modèle et vérifier l'absence de surapprentissage.

Lancer avec : uv run python -u -m housing_loan_prediction.model_checks [refit | seed | bootstrap | metrics]

- `refit` (long, ~1 h) : réapprend les 5 modèles sur un NOUVEAU découpage en 5 plis (graine NEW_SEED), sans
  aucun réglage (hyperparamètres déjà retenus ; CatBoost à 1000 arbres fixés d'avance). Pour chaque pli,
  AUC sur le pli d'entraînement et sur le pli de validation (écart = degré d'apprentissage par cœur).
- `seed` : compare le classement des modèles et des combinaisons entre l'ancien et le nouveau découpage.
- `bootstrap` : intervalles de confiance à 95 % des AUC et des écarts d'AUC entre modèles (bootstrap apparié
  des dossiers sur les prédictions hors pli). Mesure le bruit dû à l'échantillon d'évaluation, à modèles fixés.
- `metrics` : log-perte, score de Brier, calibration, matrices de confusion aux seuils 0,3 / 0,5 / 0,8.
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from housing_loan_prediction.boosting_libs import OOF_DIR, build_model as build_lib_model, make_frames
from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import RANDOM_STATE, make_cv
from housing_loan_prediction.preprocessing import ID_COL, TARGET
from housing_loan_prediction.stacking import (CANDIDATES, base_models, fit_ges, fit_logistic, load_predictions,
                                              meta_oof)

NEW_SEED = 2026
NEW_DIR = OOF_DIR / f"seed{NEW_SEED}"
FILE_NAMES = {"catboost": "catboost", "hist_gb": "hist_gb", "forêt": "random_forest", "lasso": "lasso",
              "svm_lin": "svm_lineaire"}
N_BOOT = 1000
THRESHOLDS = [0.3, 0.5, 0.8]
# Paires comparées : chaque étape de complexité contre la précédente
PAIRS = [("forêt", "lasso"), ("hist_gb", "forêt"), ("catboost", "hist_gb"), ("GES", "catboost"),
         ("stacking", "catboost"), ("stacking", "GES"), ("stacking", "lasso")]
MAIN = ["lasso", "forêt", "hist_gb", "catboost", "GES", "stacking"]


def all_predictions(P, y, seed=RANDOM_STATE):
    """Prédictions hors pli des 5 modèles + GES et stacking logistique (méta hors pli), en probabilités."""
    out = P.copy()
    out["GES"] = meta_oof(P, y, fit_ges, seed)
    out["stacking"] = expit(meta_oof(P, y, fit_logistic, seed))
    return out


# --- C + E : nouveau découpage, AUC train et validation ---

def refit_new_seed(train_clean, y):
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    sk_models = base_models(train_clean)
    X_cb, _, cat_cols = make_frames(train_clean, train_clean, "catboost")
    folds = list(make_cv(random_state=NEW_SEED).split(train_clean, y))
    rows = []
    for name in ["lasso", "svm_lineaire", "random_forest", "hist_gb", "catboost"]:
        oof, start = np.empty(len(y)), time.perf_counter()
        for k, (tr, va) in enumerate(folds):
            if name == "catboost":
                model = build_lib_model("catboost", cat_cols, n_trees=1000).fit(X_cb.iloc[tr], y[tr])
                p_tr = model.predict_proba(X_cb.iloc[tr])[:, 1]
                p_va = model.predict_proba(X_cb.iloc[va])[:, 1]
            else:
                model = clone(sk_models[name]).fit(train_clean.iloc[tr], y[tr])
                p_tr = model.predict_proba(train_clean.iloc[tr])[:, 1]
                p_va = model.predict_proba(train_clean.iloc[va])[:, 1]
            oof[va] = p_va
            rows.append({"modèle": name, "pli": k + 1, "AUC train": roc_auc_score(y[tr], p_tr),
                         "AUC validation": roc_auc_score(y[va], p_va)})
            print(f"  {name} pli {k + 1} : train {rows[-1]['AUC train']:.4f}  validation "
                  f"{rows[-1]['AUC validation']:.4f}", flush=True)
        pd.DataFrame({ID_COL: train_clean[ID_COL], TARGET: oof}).to_csv(NEW_DIR / f"{name}.csv", index=False)
        print(f"  -> {name} : AUC hors pli {roc_auc_score(y, oof):.4f} ({time.perf_counter() - start:.0f} s)",
              flush=True)
        pd.DataFrame(rows).to_csv(NEW_DIR / "train_vs_validation.csv", index=False)


def report_gap():
    t = pd.read_csv(NEW_DIR / "train_vs_validation.csv")
    g = t.groupby("modèle", sort=False)[["AUC train", "AUC validation"]].mean()
    g["écart"] = g["AUC train"] - g["AUC validation"]
    print("\n=== C) AUC sur le pli d'entraînement vs le pli de validation (moyenne des 5 plis) ===")
    print(g.round(4).to_string())


def compare_seeds(train_clean, y):
    ids = train_clean[ID_COL]
    old = all_predictions(load_predictions(0, ids), y)
    new_P = pd.DataFrame({name: pd.read_csv(NEW_DIR / f"{file}.csv").set_index(ID_COL)[TARGET].reindex(ids).to_numpy()
                          for name, file in FILE_NAMES.items()})
    new = all_predictions(new_P[list(CANDIDATES)], y, seed=NEW_SEED)
    print(f"\n=== E) AUC hors pli : découpage d'origine (graine {RANDOM_STATE}) vs nouveau (graine {NEW_SEED}) ===")
    t = pd.DataFrame({"graine 42": [roc_auc_score(y, old[c]) for c in old.columns],
                      f"graine {NEW_SEED}": [roc_auc_score(y, new[c]) for c in new.columns]}, index=old.columns)
    t["différence"] = t.iloc[:, 1] - t.iloc[:, 0]
    t["rang 42"] = t["graine 42"].rank(ascending=False).astype(int)
    t[f"rang {NEW_SEED}"] = t[f"graine {NEW_SEED}"].rank(ascending=False).astype(int)
    print(t.round(4).to_string())
    print(f"\n  Poids GES (nouveau découpage) : {fit_ges(new_P, y)[1]}")
    print(f"  Coefficients stacking (nouveau découpage) : {fit_logistic(new_P, y)[1]}")
    report_gap()


# --- A : bootstrap ---

def bootstrap(preds, y, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    cols = MAIN
    X = preds[cols].to_numpy()
    aucs = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        aucs[b] = [roc_auc_score(y[idx], X[idx, j]) for j in range(len(cols))]
        if (b + 1) % 200 == 0:
            print(f"    {b + 1}/{n_boot} tirages", flush=True)
    aucs = pd.DataFrame(aucs, columns=cols)
    point = {c: roc_auc_score(y, preds[c]) for c in cols}

    print(f"\n=== A) AUC et intervalle de confiance à 95 % ({n_boot} tirages bootstrap des dossiers) ===")
    for c in cols:
        lo, hi = aucs[c].quantile([0.025, 0.975])
        print(f"    {c:<10} {point[c]:.4f}  [{lo:.4f} ; {hi:.4f}]   paires mal classées {1 - point[c]:.2%}")

    print("\n  Écarts entre modèles (apparié : mêmes dossiers tirés pour les deux) :")
    for a, b in PAIRS:
        d = aucs[a] - aucs[b]
        lo, hi = d.quantile([0.025, 0.975])
        verdict = "significatif" if lo > 0 or hi < 0 else "égalité (0 dans l'intervalle)"
        err_red = 1 - (1 - point[a]) / (1 - point[b])
        print(f"    {a:<9} - {b:<9} {point[a] - point[b]:+.4f}  [{lo:+.4f} ; {hi:+.4f}]  "
              f"{a} meilleur dans {(d > 0).mean():.0%} des tirages, erreurs de classement {-err_red:+.0%}  -> {verdict}")


# --- B : autres mesures et matrices de confusion ---

def calibration_error(y, p, bins=10):
    """Écart moyen |taux réel - proba moyenne| sur 10 tranches de même effectif, pondéré par l'effectif."""
    d = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "d": d}).groupby("d").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return (t["n"] * (t["y"] - t["p"]).abs()).sum() / t["n"].sum()


def metrics(preds, y):
    print("\n=== B) Autres mesures (prédictions hors pli) ===")
    rows = []
    for c in MAIN:
        p = np.clip(preds[c].to_numpy(), 1e-6, 1 - 1e-6)
        rows.append({"modèle": c, "AUC": roc_auc_score(y, p), "log-perte": log_loss(y, p),
                     "Brier": brier_score_loss(y, p), "erreur de calibration": calibration_error(y, p)})
    print(pd.DataFrame(rows).set_index("modèle").round(4).to_string())

    n_ref, n_acc = (y == 0).sum(), (y == 1).sum()
    print(f"\n  Matrices de confusion ({n_acc} acceptés, {n_ref} refusés). Prédit « accepté » si proba >= seuil.")
    for s in THRESHOLDS:
        print(f"\n  Seuil {s} :")
        print(f"    {'modèle':<10} {'refus vus':>10} {'refus ratés':>12} {'acceptés vus':>13} {'acceptés ratés':>15}"
              f" {'% refus détectés':>17} {'% acceptés gardés':>18} {'erreurs':>8}")
        for c in MAIN:
            pred = preds[c].to_numpy() >= s
            tn, fp = ((y == 0) & ~pred).sum(), ((y == 0) & pred).sum()
            tp, fn = ((y == 1) & pred).sum(), ((y == 1) & ~pred).sum()
            print(f"    {c:<10} {tn:>10} {fp:>12} {tp:>13} {fn:>15} {tn / n_ref:>17.1%} {tp / n_acc:>18.1%}"
                  f" {fp + fn:>8}")


if __name__ == "__main__":
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    step = sys.argv[1] if len(sys.argv) > 1 else "metrics"

    if step == "refit":
        refit_new_seed(train_clean, y)
        report_gap()
    elif step == "seed":
        compare_seeds(train_clean, y)
    else:
        preds = all_predictions(load_predictions(0, train_clean[ID_COL]), y)
        if step == "bootstrap":
            bootstrap(preds, y)
        if step == "metrics":
            metrics(preds, y)
