"""Figures du rapport (partie II), enregistrées dans `figures/`.

Lancer avec : uv run python -u -m housing_loan_prediction.report_figures [manquants | selection | grilles | modeles]
(sans argument : toutes les figures)

- `manquants` -> fig1_manquants.png (figure 5) : taux d'acceptation selon que la variable est renseignée
  ou manquante.
- `selection` -> fig2_selection_variables.png (figure 6) : AUC du lasso selon les variables retirées.
- `grilles` -> fig_gb_grilles.png (figure 7) : courbes de validation des deux grilles du gradient boosting.
- `modeles` -> fig4_comparaison_modeles.png (figure 8) et fig5_combinaisons.png (figure 9) : AUC de chaque
  modèle et de chaque combinaison. Nécessite les prédictions hors pli de `oof/` (voir le README).

Les calculs longs (validations croisées, courbes) sont mis en cache dans `oof/` : une fois faits, relancer
le script ne fait que redessiner.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from housing_loan_prediction.boosting_libs import OOF_DIR
from housing_loan_prediction.data import load_clean, load_raw
from housing_loan_prediction.evaluation import cv_auc, make_cv
from housing_loan_prediction.gradient_boosting import (FIGURES_DIR, GRID_MAX_ITER, LEARNING_RATE_MAX_ITER,
                                                       LR_MAX_LEAF_NODES_GRID, MAX_LEAF_NODES_GRID,
                                                       MIN_SAMPLES_LEAF_GRID, learning_curve)
from housing_loan_prediction.logistic import build_model
from housing_loan_prediction.preprocessing import DROP_MAYBE, DROP_SURE, ID_COL, TARGET
from housing_loan_prediction.stacking import METHODS, fit_ges, load_predictions, meta_oof
from housing_loan_prediction.submission import BEST_C
from housing_loan_prediction.trees import build_tree

CV_CACHE = OOF_DIR / "report_figures_cv.json"
GRID_CACHE = OOF_DIR / "gb_grid_curves.npz"
BEST_ALPHA = 1e-4  # élagage retenu pour l'arbre (trees.py)
DEMOGRAPHIC = DROP_MAYBE[:9]  # race, ethnie et sexe (emprunteur, co-emprunteur, versions dérivées)

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, MUTED, GRID, FOLD_LINE = "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b3"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def fr(v, digits=3):
    """Nombre avec une virgule décimale."""
    return f"{v:.{digits}f}".replace(".", ",")


def comma_axis(axis, digits=3):
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: fr(v, digits)))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=200)
    plt.close(fig)
    print(f"  -> {FIGURES_DIR / name}")


def cached_cv(key, compute):
    """AUC par pli, calculées une fois puis relues dans CV_CACHE."""
    cache = json.loads(CV_CACHE.read_text()) if CV_CACHE.exists() else {}
    if key not in cache:
        print(f"  calcul : {key}", flush=True)
        cache[key] = [float(s) for s in compute()]
        CV_CACHE.parent.mkdir(exist_ok=True)
        CV_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def plot_folds(ax, x, M, label_mean):
    """Une ligne grise par pli, un point par moyenne, annoté."""
    for k in range(M.shape[1]):
        ax.plot(x, M[:, k], color=FOLD_LINE, lw=1, marker="o", ms=4, zorder=1, label="un pli" if k == 0 else None)
    ax.plot(x, M.mean(1), color=BLUE, lw=2 if label_mean else 0, marker="o", ms=8, zorder=2,
            label="moyenne des 5 plis")
    for i, m in enumerate(M.mean(1)):
        ax.annotate(fr(m, 4), (i, m), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)


# --- Figure 5 : valeurs manquantes ---

def fig_missing():
    train, _ = load_raw()
    y = train[TARGET]
    variables = {"loan_to_value_ratio": "LTV", "property_value": "valeur\ndu bien", "debt_to_income_ratio": "DTI",
                 "income": "revenu", "loan_term": "durée"}
    observed = [y[train[c].notna()].mean() * 100 for c in variables]
    missing = [y[train[c].isna()].mean() * 100 for c in variables]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x, w = np.arange(len(variables)), 0.38
    ax.bar(x - w / 2, observed, w - 0.03, color=BLUE, label="renseignée")
    ax.bar(x + w / 2, missing, w - 0.03, color=ORANGE, label="manquante")
    for i, v in zip(x, missing):
        ax.text(i + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.axhline(y.mean() * 100, color=MUTED, lw=1, ls="--", label=f"moyenne ({y.mean():.0%})")
    ax.set_xticks(x, list(variables.values()))
    ax.set_ylim(0, 112)
    ax.set_ylabel("taux d'acceptation (%)")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "fig1_manquants.png")


# --- Figure 6 : sélection des variables (lasso) ---

def fig_selection():
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    variable_sets = {
        "Toutes\nles variables": [],
        "Sans les 6\nredondantes": DROP_SURE,
        "Sans les 6 et\nrace/ethnie/sexe": DROP_SURE + DEMOGRAPHIC,
        "Sans les 6\net les 14": DROP_SURE + DROP_MAYBE,
    }

    def lasso_cv(drop):
        model = build_model(train_clean, drop, "lasso").set_params(logisticregression__C=BEST_C)
        return cv_auc(model, train_clean, y)

    M = np.array([cached_cv(f"lasso: {label}", lambda d=drop: lasso_cv(d)) for label, drop in variable_sets.items()])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(variable_sets))
    plot_folds(ax, x, M, label_mean=True)
    ax.set_xticks(x, list(variable_sets))
    ax.set_xlim(-0.3, len(variable_sets) - 0.4)
    ax.set_ylabel("AUC du lasso (validation croisée)")
    comma_axis(ax.yaxis)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig2_selection_variables.png")


# --- Figure 7 : grilles du gradient boosting ---

def fig_gb_grids():
    size_configs = {f"a_{l}_{m}": {"max_leaf_nodes": l, "min_samples_leaf": m, "max_iter": GRID_MAX_ITER}
                    for l in MAX_LEAF_NODES_GRID for m in MIN_SAMPLES_LEAF_GRID}
    lr_configs = {f"b_{lr}_{l}": {"learning_rate": lr, "max_leaf_nodes": l, "max_iter": it}
                  for lr, it in LEARNING_RATE_MAX_ITER.items() for l in LR_MAX_LEAF_NODES_GRID}
    curves = dict(np.load(GRID_CACHE)) if GRID_CACHE.exists() else {}
    missing = {k: p for k, p in {**size_configs, **lr_configs}.items() if k not in curves}
    if missing:
        train_clean, _, y = load_clean()
        y = y.to_numpy()
        for key, params in missing.items():
            print(f"  courbe : {key}", flush=True)
            curves[key] = learning_curve(train_clean, y, with_train=False, **params)[1].mean(axis=0)
            np.savez(GRID_CACHE, **curves)

    colors = [BLUE, ORANGE, AQUA]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for color, l in zip(colors, MAX_LEAF_NODES_GRID):
        for m, ls in zip(MIN_SAMPLES_LEAF_GRID, ["-", "--"]):
            c = curves[f"a_{l}_{m}"]
            a.plot(np.arange(1, len(c) + 1), c, color=color, ls=ls, lw=1.6, label=f"{l} feuilles, min. {m}")
    for color, lr in zip(colors, LEARNING_RATE_MAX_ITER):
        for l, ls in zip(LR_MAX_LEAF_NODES_GRID, ["-", "--"]):
            c = curves[f"b_{lr}_{l}"]
            b.plot(np.arange(1, len(c) + 1), c, color=color, ls=ls, lw=1.6,
                   label=f"η = {f'{lr:g}'.replace('.', ',')}, {l} feuilles")
    for ax, title in [(a, "(a) Taille des arbres (η = 0,1)"), (b, "(b) Taux d'apprentissage et taille des arbres")]:
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel("nombre d'arbres")
        comma_axis(ax.yaxis)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    a.set_ylabel("AUC de validation (moyenne des 5 plis)")
    a.set_ylim(0.906, 0.9185)
    a.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, columnspacing=1.0)
    b.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "fig_gb_grilles.png")


# --- Figures 8 et 9 : modèles et combinaisons ---

def fig_models():
    train_clean, _, y = load_clean()
    y = y.to_numpy()
    ids = train_clean[ID_COL]
    P = load_predictions(0, ids)
    folds = list(make_cv().split(P, y))
    per_model = {c: [roc_auc_score(y[va], P[c].to_numpy()[va]) for _, va in folds] for c in P.columns}
    lightgbm = pd.read_csv(OOF_DIR / "lightgbm.csv").set_index(ID_COL)[TARGET].reindex(ids).to_numpy()
    per_model["lightgbm"] = [roc_auc_score(y[va], lightgbm[va]) for _, va in folds]
    per_model["arbre élagué"] = cached_cv("arbre élagué", lambda: cv_auc(build_tree(train_clean, BEST_ALPHA),
                                                                         train_clean, y))
    per_method = {}
    for name, fit in METHODS.items():
        scores = []
        for tr, va in folds:
            combine, _ = fit(P.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[va], combine(P.iloc[va])))
        per_method[name] = scores

    # Figure 8 : (a) AUC de chaque modèle, moyenne +/- écart-type entre plis ; (b) courbes ROC
    labels = {"svm_lin": "SVM linéaire", "lasso": "Lasso", "arbre élagué": "Arbre élagué", "lightgbm": "LightGBM",
              "forêt": "Forêt aléatoire", "hist_gb": "HistGradientBoosting", "catboost": "CatBoost"}
    order = sorted(labels, key=lambda k: np.mean(per_model[k]))
    rows = [(labels[k], per_model[k], GREY if k == "lightgbm" else BLUE, "o") for k in order]
    rows.append(("Combinaison GES", per_method["GES (sélection gloutonne)"], ORANGE, "D"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, (_, scores, color, marker) in enumerate(rows):
        m, s = np.mean(scores), np.std(scores)
        a.errorbar(m, i, xerr=s, fmt=marker, color=color, ms=6, capsize=3, lw=1.5)
        a.text(m + s + 0.0012, i, fr(m), va="center", fontsize=8, color=INK)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    comma_axis(a.xaxis)
    a.set_xlim(0.875, 0.93)
    a.set_xlabel("AUC en validation croisée (moyenne ± écart-type des 5 plis)")
    a.xaxis.grid(True, color=GRID)
    a.set_axisbelow(True)
    a.set_title("(a) AUC de chaque modèle", loc="left", color=INK)

    p_ges = meta_oof(P, y, fit_ges)
    roc_series = [("Lasso", P["lasso"], MUTED, "-"), ("Forêt aléatoire", P["forêt"], AQUA, "-"),
                  ("CatBoost", P["catboost"], BLUE, "-"), ("GES", p_ges, ORANGE, "--")]
    for name, p, color, ls in roc_series:
        fpr, tpr, _ = roc_curve(y, p)
        b.plot(fpr, tpr, color=color, lw=1.6, ls=ls, label=f"{name} ({fr(roc_auc_score(y, p))})")
    b.plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    b.set_xlabel("taux de faux positifs (refus prédits accordés)")
    b.set_ylabel("taux de vrais positifs")
    b.set_title("(b) Courbes ROC (prédictions hors pli)", loc="left", color=INK)
    b.legend(frameon=False, fontsize=8, loc="lower right")
    b.set_aspect("equal")
    comma_axis(b.xaxis, 1)
    comma_axis(b.yaxis, 1)
    save(fig, "fig4_comparaison_modeles.png")

    # Figure 9 : combinaisons, AUC par pli (poids appris sur 4 plis, notés sur le 5e)
    names = {"meilleur modèle seul": "CatBoost\nseul", "moyenne des rangs": "Moyenne\ndes rangs",
             "moyenne des probabilités": "Moyenne des\nprobabilités", "GES (sélection gloutonne)": "GES",
             "stacking logistique": "Stacking\nlogistique"}
    M = np.array([per_method[k] for k in names])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    plot_folds(ax, x, M, label_mean=False)
    ges = list(names).index("GES (sélection gloutonne)")
    ax.scatter([ges], [M[ges].mean()], s=110, color=ORANGE, zorder=3, label="retenu (GES)")
    ax.set_xticks(x, list(names.values()))
    ax.set_xlim(-0.3, len(names) - 0.3)
    ax.set_ylabel("AUC (combinaison apprise sur 4 plis,\nnotée sur le 5e)")
    comma_axis(ax.yaxis)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig5_combinaisons.png")


FIGURES = {"manquants": fig_missing, "selection": fig_selection, "grilles": fig_gb_grids, "modeles": fig_models}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        print(f"\n=== {name} ===")
        FIGURES[name]()
