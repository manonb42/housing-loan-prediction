"""Évaluation des modèles par validation croisée, avec l'AUC ROC comme métrique commune."""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

N_SPLITS = 5
RANDOM_STATE = 42


def make_cv(n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Plis stratifiés à graine fixe : tous les modèles sont évalués sur les mêmes découpages."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cv_auc(model, X, y):
    """AUC ROC de chaque pli d'une validation croisée stratifiée.

    Le preprocessing doit être dans `model` (Pipeline) pour être réappris dans chaque pli.
    """
    return cross_val_score(model, X, y, cv=make_cv(), scoring="roc_auc", n_jobs=-1)


def tune_auc(model, param_grid, X, y):
    """Choisit les hyperparamètres qui maximisent l'AUC moyenne sur les mêmes plis que `cv_auc`.

    Le meilleur modèle est réentraîné sur tout `X` (`search.best_estimator_`).
    """
    search = GridSearchCV(model, param_grid, cv=make_cv(), scoring="roc_auc", n_jobs=-1)
    return search.fit(X, y)


def best_fold_scores(search):
    """AUC de chaque pli pour les meilleurs hyperparamètres d'une recherche."""
    i = search.best_index_
    return np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(search.n_splits_)])


def report(name, scores):
    """Affiche l'AUC moyenne, son écart-type entre plis et le détail par pli."""
    folds = " ".join(f"{s:.4f}" for s in scores)
    print(f"{name} : AUC CV = {scores.mean():.4f} (+/- {scores.std():.4f})  [plis : {folds}]")


def report_grid(search, param):
    """Affiche l'AUC moyenne pour chaque valeur testée de `param`."""
    results = search.cv_results_
    for value, mean, std in zip(results[f"param_{param}"], results["mean_test_score"], results["std_test_score"]):
        print(f"    {param.split('__')[-1]} = {value:<8} AUC = {mean:.4f} (+/- {std:.4f})")
