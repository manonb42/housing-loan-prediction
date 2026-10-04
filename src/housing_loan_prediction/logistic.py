"""Régressions logistiques sans pénalité, ridge (L2) et lasso (L1) : AUC en validation croisée.

Lancer avec : uv run python -m housing_loan_prediction.logistic

Dans scikit-learn, la force de pénalité est réglée par C = 1 / lambda : un petit C pénalise fort.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import best_fold_scores, cv_auc, report, report_grid, tune_auc
from housing_loan_prediction.preprocessing import DROP_MAYBE, DROP_SURE, build_linear_preprocessor

C_PARAM = "logisticregression__C"
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]

PENALTIES = {
    "sans pénalité": {"C": np.inf},
    "ridge": {"l1_ratio": 0.0},
    "lasso": {"l1_ratio": 1.0, "solver": "liblinear"},
}

DROP_SETS = {
    "toutes les variables": [],
    "DROP_SURE": DROP_SURE,
    "DROP_SURE + DROP_MAYBE": DROP_SURE + DROP_MAYBE,
}


def build_model(train_clean, drop, penalty):
    return make_pipeline(
        build_linear_preprocessor(train_clean, drop=drop),
        LogisticRegression(max_iter=2000, **PENALTIES[penalty]),
    )


def coefs_by_variable(model):
    """Nombre de coefficients et de coefficients non nuls par variable d'origine (une modalité = un coef)."""
    preprocessor, logreg = model[0], model[-1]
    variables = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "cat":
            for col, categories in zip(columns, transformer.categories_):
                variables += [col] * len(categories)
        else:
            variables += list(columns)
    nonzero = pd.Series(logreg.coef_.ravel() != 0, index=variables)
    return nonzero.groupby(level=0).agg(n_coefs="size", n_nonzero="sum").sort_values("n_nonzero")


def run_tuned(train_clean, y, drop_name, penalty):
    model = build_model(train_clean, DROP_SETS[drop_name], penalty)
    search = tune_auc(model, {C_PARAM: C_GRID}, train_clean, y)
    print(f"\n{penalty}, {drop_name} : meilleur C = {search.best_params_[C_PARAM]}")
    report_grid(search, C_PARAM)
    report(f"  -> {penalty}, {drop_name}", best_fold_scores(search))
    return search


def report_selection(name, model):
    coefs = coefs_by_variable(model)
    unused = coefs.index[coefs["n_nonzero"] == 0].tolist()
    print(f"\n{name} : {coefs['n_nonzero'].sum()} coefficients non nuls sur {coefs['n_coefs'].sum()}")
    print(f"  Variables entièrement mises à zéro ({len(unused)}) : {unused}")
    partial = coefs[(coefs["n_nonzero"] > 0) & (coefs["n_nonzero"] < coefs["n_coefs"])]
    if len(partial):
        print("  Variables catégorielles dont seule une partie des modalités est gardée :")
        for var, row in partial.iterrows():
            print(f"    {var} : {row['n_nonzero']} / {row['n_coefs']}")


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()

    print("=== 1) Mêmes données (DROP_SURE + DROP_MAYBE) : sans pénalité, ridge, lasso ===")
    drop_name = "DROP_SURE + DROP_MAYBE"
    report(f"\nsans pénalité, {drop_name}",
           cv_auc(build_model(train_clean, DROP_SETS[drop_name], "sans pénalité"), train_clean, y))
    run_tuned(train_clean, y, drop_name, "ridge")
    lassos = {drop_name: run_tuned(train_clean, y, drop_name, "lasso")}

    print("\n=== 2) Lasso selon les variables retenues ===")
    for name in ["toutes les variables", "DROP_SURE"]:
        lassos[name] = run_tuned(train_clean, y, name, "lasso")
    for name, search in lassos.items():
        report_selection(f"Lasso, {name} (C = {search.best_params_[C_PARAM]})", search.best_estimator_)
