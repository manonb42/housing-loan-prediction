"""Recherche aléatoire des hyperparamètres du gradient boosting (HistGradientBoosting).

Lancer avec : uv run python -u -m housing_loan_prediction.random_search

Vérification des grilles de gradient_boosting.py (cours 6) : avec N essais, une grille ne teste
que N^(1/d) valeurs distinctes par hyperparamètre, un tirage aléatoire en teste N.
Pour chaque tirage, le nombre d'arbres est lu sur la courbe d'apprentissage (5 plis),
comme pour les grilles. Les résultats sont ajoutés au fur et à mesure dans RESULTS_PATH.
"""

import json
from pathlib import Path

import numpy as np

from housing_loan_prediction.data import load_clean
from housing_loan_prediction.gradient_boosting import learning_curve

N_DRAWS = 20
SEED = 2026
TREES_BUDGET = 15  # max_iter = TREES_BUDGET / learning_rate : ~2x le nombre d'arbres optimal observé
RESULTS_PATH = Path(__file__).resolve().parents[2] / "oof" / "random_search_gb.json"


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def draw(rng):
    """Un tirage des trois hyperparamètres des grilles ; le nombre d'arbres maximal suit le taux."""
    learning_rate = log_uniform(rng, 0.01, 0.2)
    return {
        "learning_rate": round(learning_rate, 4),
        "max_leaf_nodes": int(round(log_uniform(rng, 15, 255))),
        "min_samples_leaf": int(round(log_uniform(rng, 10, 200))),
        "max_iter": int(TREES_BUDGET / learning_rate),
    }


if __name__ == "__main__":
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    rng = np.random.default_rng(SEED)
    configs = [draw(rng) for _ in range(N_DRAWS)]
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = []
    for i, params in enumerate(configs, 1):
        print(f"\n[{i}/{N_DRAWS}] {params}", flush=True)
        _, val_curves = learning_curve(train_clean, y, with_train=False, **params)
        mean = val_curves.mean(axis=0)
        best = int(mean.argmax())
        results.append({**params, "auc": float(mean[best]), "auc_std": float(val_curves[:, best].std()),
                        "best_n_trees": best + 1, "fold_auc": val_curves[:, best].tolist()})
        print(f"  -> AUC = {mean[best]:.4f} (+/- {val_curves[:, best].std():.4f}) à {best + 1} arbres", flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=1))

    print("\nClassement :")
    for r in sorted(results, key=lambda r: -r["auc"]):
        print(f"  {r['auc']:.4f}  lr={r['learning_rate']:<7} feuilles={r['max_leaf_nodes']:<4} "
              f"min_leaf={r['min_samples_leaf']:<4} arbres={r['best_n_trees']}")
