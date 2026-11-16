"""Contrôle de robustesse : on refait toute la procédure sur un autre découpage en 5 plis.

Lancer avec : uv run python -u -m housing_loan_prediction.model_checks [refit | seed]
(nécessite les prédictions hors pli de `oof/`, voir `stacking oof`)

- `refit` (long, ~1 h) : réapprend les 5 modèles sur un NOUVEAU découpage en 5 plis (graine NEW_SEED),
  sans aucun réglage (hyperparamètres déjà retenus ; CatBoost à 1000 arbres fixés d'avance), et enregistre
  leurs prédictions hors pli dans `oof/seed2026/`.
- `seed` : compare les AUC et le classement des modèles et des combinaisons (GES, stacking) entre l'ancien
  et le nouveau découpage, et affiche les poids appris sur le nouveau.
Si le classement ou les poids changeaient beaucoup, nos choix dépendraient du hasard du découpage.
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from housing_loan_prediction.boosting_libs import OOF_DIR, build_model as build_lib_model, make_frames
from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import RANDOM_STATE, make_cv
from housing_loan_prediction.preprocessing import ID_COL, TARGET
from housing_loan_prediction.stacking import (CANDIDATES, base_models, fit_ges, fit_logistic, load_predictions,
                                              meta_oof)

NEW_SEED = 2026
NEW_DIR = OOF_DIR / f"seed{NEW_SEED}"
# nom dans CANDIDATES -> nom du fichier de prédictions hors pli
FILE_NAMES = {"catboost": "catboost", "hist_gb": "hist_gb", "forêt": "random_forest", "lasso": "lasso",
              "svm_lin": "svm_lineaire"}


def all_predictions(P, y, seed=RANDOM_STATE):
    """Prédictions hors pli des 5 modèles + GES et stacking logistique (méta hors pli), en probabilités."""
    out = P.copy()
    out["GES"] = meta_oof(P, y, fit_ges, seed)
    out["stacking"] = expit(meta_oof(P, y, fit_logistic, seed))
    return out


def refit_new_seed(train_clean, y):
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    sk_models = base_models(train_clean)
    X_cb, _, cat_cols = make_frames(train_clean, train_clean, "catboost")
    folds = list(make_cv(random_state=NEW_SEED).split(train_clean, y))
    for name in FILE_NAMES.values():
        oof, start = np.empty(len(y)), time.perf_counter()
        for tr, va in folds:
            if name == "catboost":
                model = build_lib_model("catboost", cat_cols, n_trees=1000).fit(X_cb.iloc[tr], y[tr])
                oof[va] = model.predict_proba(X_cb.iloc[va])[:, 1]
            else:
                model = clone(sk_models[name]).fit(train_clean.iloc[tr], y[tr])
                oof[va] = model.predict_proba(train_clean.iloc[va])[:, 1]
        pd.DataFrame({ID_COL: train_clean[ID_COL], TARGET: oof}).to_csv(NEW_DIR / f"{name}.csv", index=False)
        print(f"  {name} : AUC hors pli {roc_auc_score(y, oof):.4f} ({time.perf_counter() - start:.0f} s)",
              flush=True)


def compare_seeds(train_clean, y):
    ids = train_clean[ID_COL]
    old = all_predictions(load_predictions(0, ids), y)
    new_P = pd.DataFrame({name: pd.read_csv(NEW_DIR / f"{file}.csv").set_index(ID_COL)[TARGET].reindex(ids).to_numpy()
                          for name, file in FILE_NAMES.items()})
    new = all_predictions(new_P[list(CANDIDATES)], y, seed=NEW_SEED)
    print(f"\n=== AUC hors pli : découpage d'origine (graine {RANDOM_STATE}) vs nouveau (graine {NEW_SEED}) ===")
    t = pd.DataFrame({f"graine {RANDOM_STATE}": [roc_auc_score(y, old[c]) for c in old.columns],
                      f"graine {NEW_SEED}": [roc_auc_score(y, new[c]) for c in new.columns]}, index=old.columns)
    t["différence"] = t.iloc[:, 1] - t.iloc[:, 0]
    t[f"rang {RANDOM_STATE}"] = t.iloc[:, 0].rank(ascending=False).astype(int)
    t[f"rang {NEW_SEED}"] = t.iloc[:, 1].rank(ascending=False).astype(int)
    print(t.round(4).to_string())
    print(f"\n  Poids GES (nouveau découpage) : {fit_ges(new_P, y)[1]}")
    print(f"  Coefficients stacking (nouveau découpage) : {fit_logistic(new_P, y)[1]}")


if __name__ == "__main__":
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    step = sys.argv[1] if len(sys.argv) > 1 else "seed"
    if step == "refit":
        refit_new_seed(train_clean, y)
    compare_seeds(train_clean, y)
