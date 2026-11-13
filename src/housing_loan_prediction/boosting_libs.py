"""LightGBM et CatBoost : le même gradient boosting que HistGradientBoosting, autrement implémenté.

Lancer avec : uv run python -u -m housing_loan_prediction.boosting_libs [curves <lib> | plot | submit <lib>]
avec <lib> parmi lightgbm, catboost.

Pourquoi les tester : HistGradientBoosting ne traite en catégoriel que les variables à 255 modalités au
plus ; le comté (739) et la MSA (383) y sont des entiers sans sens. Les deux bibliothèques les gèrent
nativement, de deux façons différentes :
- LightGBM : à chaque nœud, modalités triées selon leur gradient moyen, puis meilleure coupure en deux
  groupes ; flexible, donc exposé au surapprentissage sur les modalités peu remplies ;
- CatBoost : chaque modalité remplacée par le taux d'acceptation des dossiers « précédents » (ordre
  aléatoire), tiré vers la moyenne globale quand elle a peu de dossiers, sans fuite de la cible.

Peu de réglage, volontairement : LightGBM avec des arbres de taille comparable à HistGB (lr 0.05),
CatBoost avec ses réglages par défaut. Seul le nombre d'arbres est choisi, sur la courbe d'apprentissage
(AUC de validation après chaque arbre, mêmes 5 plis que les autres modèles). Les prédictions hors pli
sont enregistrées dans `oof/` pour le stacking.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from housing_loan_prediction.data import DATA_DIR, load_clean
from housing_loan_prediction.evaluation import RANDOM_STATE, make_cv
from housing_loan_prediction.gradient_boosting import COMPARE_PATHS, FIGURES_DIR, TEXT_MUTED, check_submission
from housing_loan_prediction.preprocessing import DROP_SURE, ID_COL, TARGET, feature_columns

ROOT = DATA_DIR.parent
OOF_DIR = ROOT / "oof"
LEARNING_RATE = 0.05
MAX_TREES = 2000

# Tailles d'arbres comparables à celle retenue pour HistGradientBoosting (63 feuilles)
PARAMS = {
    "lightgbm": {"num_leaves": 63, "min_child_samples": 20},
    "catboost": {},  # réglages par défaut : 1000 arbres symétriques de profondeur 6, L2 = 3, pas choisi par CatBoost
}
CATBOOST_DEFAULT_TREES = 1000
COLORS = {"lightgbm": "#2a78d6", "catboost": "#1baf7a"}


def split_columns(train_clean):
    """Mêmes variables que la forêt et HistGB (DROP_SURE, DTI en nombre) ; catégorielles à part."""
    log_cols, num_cols, flag_cols, cat_cols = feature_columns(train_clean, [*DROP_SURE, "debt_to_income_ratio"])
    return log_cols + num_cols + flag_cols, cat_cols


def make_frames(train_clean, test_clean, lib):
    """Numériques telles quelles (NaN gérés par les arbres) ; catégorielles en `category` (LightGBM)
    ou en texte (CatBoost). Les modalités sont celles du train : une modalité inconnue du test devient
    NaN (LightGBM) ou "nan" (CatBoost)."""
    num_cols, cat_cols = split_columns(train_clean)
    frames = []
    for df in (train_clean, test_clean):
        X = df[num_cols + cat_cols].copy()
        for col in cat_cols:
            X[col] = pd.Categorical(df[col], categories=sorted(train_clean[col].unique()))
            if lib == "catboost":
                X[col] = X[col].astype(str)
        frames.append(X)
    return frames[0], frames[1], cat_cols


def max_trees(lib):
    return CATBOOST_DEFAULT_TREES if lib == "catboost" else MAX_TREES


def build_model(lib, cat_cols, n_trees=None):
    n_trees = n_trees or max_trees(lib)
    common = {"learning_rate": LEARNING_RATE, "random_state": RANDOM_STATE}
    if lib == "lightgbm":
        return LGBMClassifier(n_estimators=n_trees, verbose=-1, **common, **PARAMS[lib])
    # Pas de learning_rate : CatBoost le choisit selon la taille des données et le nombre d'arbres.
    # use_best_model=False : le pli de validation sert à tracer la courbe, pas à arrêter l'apprentissage
    return CatBoostClassifier(iterations=n_trees, cat_features=cat_cols, eval_metric="AUC", use_best_model=False,
                              verbose=0, thread_count=-1, random_seed=RANDOM_STATE, **PARAMS[lib])


def fit_with_curve(model, lib, X_tr, y_tr, X_va, y_va):
    """Apprend le modèle et renvoie l'AUC de validation après chaque arbre."""
    if lib == "lightgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
        return np.array(model.evals_result_["valid_0"]["auc"])
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return np.array(model.get_evals_result()["validation"]["AUC"])


def predict(model, lib, X, n_trees):
    """Probabilité d'acceptation en n'utilisant que les `n_trees` premiers arbres."""
    if lib == "lightgbm":
        return model.predict_proba(X, num_iteration=n_trees)[:, 1]
    return model.predict_proba(X, ntree_end=n_trees)[:, 1]


def run_curves(lib, train_clean, test_clean, y):
    """Courbe d'apprentissage sur 5 plis, nombre d'arbres retenu, prédictions hors pli enregistrées."""
    X, _, cat_cols = make_frames(train_clean, test_clean, lib)
    curves, models, folds = [], [], list(make_cv().split(X, y))
    for k, (tr, va) in enumerate(folds):
        start = time.perf_counter()
        model = build_model(lib, cat_cols)
        curves.append(fit_with_curve(model, lib, X.iloc[tr], y[tr], X.iloc[va], y[va]))
        models.append(model)
        best = curves[-1].argmax()
        lr = f", learning_rate choisi = {model.learning_rate_:.4f}" if lib == "catboost" else ""
        print(f"    pli {k + 1} : meilleure AUC = {curves[-1][best]:.4f} à {best + 1} arbres "
              f"({time.perf_counter() - start:.0f} s{lr})")

    curves = np.array(curves)
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    n_best = int(mean.argmax()) + 1
    print(f"  -> AUC CV maximale = {mean[n_best - 1]:.4f} (+/- {std[n_best - 1]:.4f}) à {n_best} arbres")
    for n in (100, 250, 500, 750, 1000, 1500, 2000):
        if n <= len(mean):
            print(f"    {n:>5} arbres : AUC = {mean[n - 1]:.4f} (+/- {std[n - 1]:.4f})")

    oof = np.empty(len(y))
    for model, (_, va) in zip(models, folds):
        oof[va] = predict(model, lib, X.iloc[va], n_best)
    print(f"  AUC hors pli (toutes les prédictions ensemble, {n_best} arbres) : {roc_auc_score(y, oof):.4f}")

    OOF_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: train_clean[ID_COL], TARGET: oof}).to_csv(OOF_DIR / f"{lib}.csv", index=False)
    np.save(OOF_DIR / f"{lib}_curves.npy", curves)
    print(f"  Prédictions hors pli : {OOF_DIR / f'{lib}.csv'}")


def plot_curves():
    """Courbes de validation moyennes de LightGBM et CatBoost, avec le meilleur HistGB pour repère."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    best_auc = 0
    for lib, color in COLORS.items():
        path = OOF_DIR / f"{lib}_curves.npy"
        if not path.exists():
            continue
        mean = np.load(path).mean(axis=0)
        best_auc = max(best_auc, mean.max())
        ax.plot(np.arange(1, len(mean) + 1), mean, lw=2, color=color,
                label=f"{lib} : {mean.max():.4f} à {mean.argmax() + 1} arbres")
    ax.axhline(0.9151, color=TEXT_MUTED, lw=1, ls="--")
    ax.annotate("HistGradientBoosting (lr 0.02) : 0.9151", (MAX_TREES, 0.9151), xytext=(-4, 4),
                textcoords="offset points", ha="right", color=TEXT_MUTED, fontsize=9)
    ax.set_ylim(best_auc - 0.008, best_auc + 0.002)
    ax.set_xlabel("nombre d'arbres")
    ax.set_ylabel("AUC ROC de validation (moyenne des 5 plis)")
    ax.set_title("Gradient boosting : LightGBM (learning_rate = 0.05), CatBoost (défaut)", loc="left", fontsize=11)
    ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "boosting_libs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Graphique : {path}")


def write_submission(lib, train_clean, test_clean, y):
    """Apprend sur tout le train avec un peu plus d'arbres que le maximum de la courbe (25 % de dossiers
    en plus qu'un pli, plateau large), comme pour HistGB."""
    curves = np.load(OOF_DIR / f"{lib}_curves.npy")
    n_trees = int(round(1.1 * (curves.mean(axis=0).argmax() + 1)))
    X, X_test, cat_cols = make_frames(train_clean, test_clean, lib)
    model = build_model(lib, cat_cols, n_trees).fit(X, y)
    submission = pd.DataFrame({ID_COL: test_clean[ID_COL], TARGET: predict(model, lib, X_test, n_trees)})
    path = ROOT / "submissions" / f"{lib}.csv"
    submission.to_csv(path, index=False)
    print(f"  {n_trees} arbres, soumission écrite : {path}")
    oof = pd.read_csv(OOF_DIR / f"{lib}.csv")[TARGET].to_numpy()
    check_submission(submission, oof, y, {**COMPARE_PATHS, "HistGB": ROOT / "submissions" / "gradient_boosting.csv"})


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()
    y = y.to_numpy()
    step = sys.argv[1] if len(sys.argv) > 1 else "curves"
    libs = sys.argv[2:] or list(PARAMS)

    if step == "curves":
        for lib in libs:
            print(f"\n=== {lib} : {PARAMS[lib] or 'réglages par défaut'}, courbe sur {max_trees(lib)} arbres ===")
            run_curves(lib, train_clean, test_clean, y)
        plot_curves()
    if step == "plot":
        plot_curves()
    if step == "submit":
        for lib in libs:
            print(f"\n=== Soumission {lib} ===")
            write_submission(lib, train_clean, test_clean, y)
