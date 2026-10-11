"""Gradient boosting d'arbres (HistGradientBoostingClassifier) : AUC en validation croisée.

Lancer avec : uv run python -u -m housing_loan_prediction.gradient_boosting [default | tree_size | learning_rate | submit]

Le modèle part du taux moyen d'acceptation, puis ajoute des petits arbres un par un : chacun
apprend les erreurs (résidus) des précédents et n'est ajouté qu'à hauteur de `learning_rate`.
Contrairement à la forêt, trop d'arbres finit par surapprendre : le nombre d'arbres se choisit
sur la courbe d'apprentissage (AUC de validation en fonction du nombre d'arbres).

Données : mêmes colonnes que pour la forêt, sans log ni standardisation, NaN laissés tels quels
(le modèle apprend de quel côté de chaque coupure les envoyer). Les catégorielles sont traitées
nativement, sauf celles qui dépassent 255 modalités (comté, MSA) : elles restent des entiers.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

from housing_loan_prediction.data import DATA_DIR, load_clean
from housing_loan_prediction.evaluation import RANDOM_STATE, make_cv
from housing_loan_prediction.preprocessing import DROP_SURE, ID_COL, TARGET, build_tree_preprocessor, feature_columns

MAX_CATEGORIES = 255  # limite du mode catégoriel natif
CURVE_MAX_ITER = 1000
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

MAX_LEAF_NODES_GRID = [15, 31, 63]
MIN_SAMPLES_LEAF_GRID = [20, 100]
GRID_MAX_ITER = 400

# Un pas k fois plus petit demande environ k fois plus d'arbres
LEARNING_RATE_MAX_ITER = {0.1: 300, 0.05: 600, 0.02: 1500}
LR_MAX_LEAF_NODES_GRID = [63, 127]

# Retenu par la grille (AUC CV max 0.9151 à 369 arbres) ; un peu plus d'arbres pour le modèle final,
# appris sur tout le train (25 % de dossiers en plus qu'un pli), le plateau étant large
BEST_GB = {"learning_rate": 0.02, "max_leaf_nodes": 63, "max_iter": 400}
SUBMISSION_PATH = DATA_DIR.parent / "submissions" / "gradient_boosting.csv"
COMPARE_PATHS = {"forêt": DATA_DIR.parent / "submissions" / "random_forest.csv",
                 "lasso": DATA_DIR.parent / "submissions" / "lasso_drop_sure.csv"}

SERIES_COLORS = {"validation": "#2a78d6", "train": "#eb6834"}
GRID_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_MUTED = "#6b6a63"


def categorical_mask(train_clean, drop=DROP_SURE):
    """Colonnes de sortie de `build_tree_preprocessor` à traiter en catégoriel natif."""
    log_cols, num_cols, flag_cols, cat_cols = feature_columns(train_clean, [*drop, "debt_to_income_ratio"])
    n_num = len(log_cols + num_cols + flag_cols)
    native = [train_clean[col].nunique() <= MAX_CATEGORIES for col in cat_cols]
    return np.array([False] * n_num + native)


def build_gb(train_clean, **params):
    params = {"early_stopping": False, "random_state": RANDOM_STATE, **params}
    return make_pipeline(
        build_tree_preprocessor(train_clean, drop=DROP_SURE),
        HistGradientBoostingClassifier(categorical_features=categorical_mask(train_clean), **params),
    )


def staged_auc(model, X, y):
    """AUC après 1, 2, ..., M arbres, sans réapprendre le modèle."""
    X_t = model[0].transform(X)
    return np.array([roc_auc_score(y, s) for s in model[-1].staged_decision_function(X_t)])


def learning_curve(train_clean, y, max_iter=CURVE_MAX_ITER, with_train=True, **params):
    """AUC train et validation en fonction du nombre d'arbres, pour chacun des 5 plis."""
    train_curves, val_curves = [], []
    for k, (tr, va) in enumerate(make_cv().split(train_clean, y)):
        model = build_gb(train_clean, max_iter=max_iter, **params).fit(train_clean.iloc[tr], y[tr])
        if with_train:
            train_curves.append(staged_auc(model, train_clean.iloc[tr], y[tr]))
        val_curves.append(staged_auc(model, train_clean.iloc[va], y[va]))
        best = val_curves[-1].argmax()
        print(f"    pli {k + 1} : meilleure AUC validation = {val_curves[-1][best]:.4f} à {best + 1} arbres")
    return np.array(train_curves), np.array(val_curves)


def report_curve(train_curves, val_curves, checkpoints=(50, 100, 200, 300, 500, 750, 1000)):
    val_mean, val_std = val_curves.mean(axis=0), val_curves.std(axis=0)
    best = val_mean.argmax()
    print(f"  -> AUC CV maximale = {val_mean[best]:.4f} (+/- {val_std[best]:.4f}) à {best + 1} arbres")
    for n in checkpoints:
        if n <= len(val_mean):
            print(f"    {n:>5} arbres : AUC train = {train_curves[:, n - 1].mean():.4f}  "
                  f"AUC validation = {val_mean[n - 1]:.4f} (+/- {val_std[n - 1]:.4f})")
    return best + 1


def plot_curve(train_curves, val_curves, title, path):
    n_trees = np.arange(1, train_curves.shape[1] + 1)
    val_mean, val_std = val_curves.mean(axis=0), val_curves.std(axis=0)
    best = val_mean.argmax()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n_trees, train_curves.mean(axis=0), color=SERIES_COLORS["train"], lw=2, label="train")
    ax.plot(n_trees, val_mean, color=SERIES_COLORS["validation"], lw=2, label="validation (moyenne des 5 plis)")
    ax.fill_between(n_trees, val_mean - val_std, val_mean + val_std, color=SERIES_COLORS["validation"],
                    alpha=0.15, lw=0)
    ax.axvline(best + 1, color=TEXT_MUTED, lw=1, ls="--")
    ax.annotate(f"max validation : {val_mean[best]:.4f}\nà {best + 1} arbres", (best + 1, val_mean[best]),
                xytext=(8, -36), textcoords="offset points", color=TEXT_MUTED, fontsize=9)
    ax.set_ylim(max(0.85, val_mean.min() - 0.005), min(1.0, train_curves.max() + 0.005))
    ax.set_xlabel("nombre d'arbres")
    ax.set_ylabel("AUC ROC")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Graphique : {path}")


def run_grid(train_clean, y, configs):
    """Courbe d'apprentissage (validation seulement) pour chaque réglage ; on compare les maximums."""
    results = {}
    for params in configs:
        label = ", ".join(f"{k} = {v}" for k, v in params.items() if k != "max_iter")
        print(f"\n  {label} ({params['max_iter']} arbres)")
        results[label] = learning_curve(train_clean, y, with_train=False, **params)[1]

    print("\n  Résumé (AUC validation maximale, moyenne des 5 plis) :")
    for label, val_curves in results.items():
        mean = val_curves.mean(axis=0)
        best = mean.argmax()
        print(f"    {label:<45} AUC = {mean[best]:.4f} (+/- {val_curves.std(axis=0)[best]:.4f}) à {best + 1} arbres")
    return results


def plot_grid(results, styles, title, path):
    """Courbes de validation moyennes de la grille ; `styles` donne (couleur, trait, légende) par réglage."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, val_curves), (color, ls, legend) in zip(results.items(), styles):
        mean = val_curves.mean(axis=0)
        ax.plot(np.arange(1, len(mean) + 1), mean, lw=2, ls=ls, color=color, label=legend)
    best_auc = max(curves.mean(axis=0).max() for curves in results.values())
    ax.set_ylim(best_auc - 0.008, best_auc + 0.002)
    ax.set_xlabel("nombre d'arbres")
    ax.set_ylabel("AUC ROC de validation (moyenne des 5 plis)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Graphique : {path}")


def tree_size_grid():
    configs, styles = [], []
    for i, max_leaf_nodes in enumerate(MAX_LEAF_NODES_GRID):
        for min_samples_leaf, ls in zip(MIN_SAMPLES_LEAF_GRID, ["-", "--"]):
            configs.append({"max_leaf_nodes": max_leaf_nodes, "min_samples_leaf": min_samples_leaf,
                            "max_iter": GRID_MAX_ITER})
            styles.append((GRID_COLORS[i], ls, f"{max_leaf_nodes} feuilles, min {min_samples_leaf} dossiers/feuille"))
    return configs, styles


def learning_rate_grid():
    configs, styles = [], []
    for i, (learning_rate, max_iter) in enumerate(LEARNING_RATE_MAX_ITER.items()):
        for max_leaf_nodes, ls in zip(LR_MAX_LEAF_NODES_GRID, ["-", "--"]):
            configs.append({"learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes, "max_iter": max_iter})
            styles.append((GRID_COLORS[i], ls, f"learning_rate = {learning_rate}, {max_leaf_nodes} feuilles"))
    return configs, styles


def check_submission(submission, oof_proba, y):
    """Contrôles avant envoi : format, valeurs, et cohérence des prédictions du test avec celles du train."""
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    proba = submission[TARGET]
    print("\n  Format :")
    print(f"    lignes : {len(submission)} (attendu {len(sample)}), colonnes : {list(submission.columns)}")
    print(f"    mêmes id que sample_submission : {set(submission[ID_COL]) == set(sample[ID_COL])}, "
          f"id en double : {submission[ID_COL].duplicated().sum()}")
    print(f"    NaN : {proba.isna().sum()}, valeurs dans [0, 1] : {proba.between(0, 1).all()}")

    print("\n  Distribution des probabilités (train hors pli vs test) :")
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print("    " + " ".join(f"{f'q{int(q * 100)}':>7}" for q in quantiles) + "  moyenne")
    for name, values in [("train", oof_proba), ("test", proba.to_numpy())]:
        print(f"    {' '.join(f'{v:7.3f}' for v in np.quantile(values, quantiles))}  {values.mean():.3f}  ({name})")
    print(f"    taux d'acceptation réel du train : {y.mean():.3f}")

    print("\n  Accord avec les soumissions précédentes (corrélation des rangs, Spearman) :")
    for name, path in COMPARE_PATHS.items():
        if path.exists():
            other = pd.read_csv(path).set_index(ID_COL)[TARGET].reindex(submission[ID_COL])
            print(f"    {name} : {proba.corr(pd.Series(other.to_numpy(), index=proba.index), method='spearman'):.3f}")


def write_submission(train_clean, test_clean, y, path=SUBMISSION_PATH):
    """Contrôle hors pli du réglage final, puis apprentissage sur tout le train et écriture des probabilités."""
    oof_proba = cross_val_predict(build_gb(train_clean, **BEST_GB), train_clean, y, cv=make_cv(),
                                  method="predict_proba")[:, 1]
    print(f"  AUC hors pli du réglage final ({BEST_GB['max_iter']} arbres) : {roc_auc_score(y, oof_proba):.4f}")

    model = build_gb(train_clean, **BEST_GB).fit(train_clean, y)
    submission = pd.DataFrame({ID_COL: test_clean[ID_COL], TARGET: model.predict_proba(test_clean)[:, 1]})
    path.parent.mkdir(exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"  Soumission écrite : {path}")
    check_submission(submission, oof_proba, y)


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()
    y = y.to_numpy()
    n_native = categorical_mask(train_clean).sum()
    print(f"Catégorielles en natif : {n_native} ; comté et MSA (> {MAX_CATEGORIES} modalités) en entiers")
    step = sys.argv[1] if len(sys.argv) > 1 else "default"

    if step == "default":
        print(f"\n=== 1) Réglages par défaut (learning_rate = 0.1, 31 feuilles), courbe sur {CURVE_MAX_ITER} arbres ===")
        train_curves, val_curves = learning_curve(train_clean, y)
        report_curve(train_curves, val_curves)
        plot_curve(train_curves, val_curves, "Gradient boosting, réglages par défaut : AUC selon le nombre d'arbres",
                   FIGURES_DIR / "gb_learning_curve_default.png")

    if step == "tree_size":
        print(f"\n=== 2) Taille des arbres (learning_rate = 0.1), courbes sur {GRID_MAX_ITER} arbres ===")
        configs, styles = tree_size_grid()
        plot_grid(run_grid(train_clean, y, configs), styles,
                  "Gradient boosting (learning_rate = 0.1) : taille des arbres", FIGURES_DIR / "gb_tree_size.png")

    if step == "learning_rate":
        print("\n=== 3) Taux d'apprentissage x taille des arbres ===")
        configs, styles = learning_rate_grid()
        plot_grid(run_grid(train_clean, y, configs), styles,
                  "Gradient boosting : taux d'apprentissage et taille des arbres",
                  FIGURES_DIR / "gb_learning_rate.png")

    if step == "submit":
        print(f"\n=== Soumission : {BEST_GB} ===")
        write_submission(train_clean, test_clean, y)
