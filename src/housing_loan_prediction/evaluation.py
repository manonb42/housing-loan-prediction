"""Évaluation des modèles par validation croisée, avec l'AUC ROC comme métrique commune."""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

N_SPLITS = 5
RANDOM_STATE = 42


def make_cv(n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Build the stratified folds shared by all the models.

    Args:
        n_splits: Number of folds.
        random_state: Seed of the shuffling.

    Returns:
        StratifiedKFold splitter.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cv_auc(model, X, y):
    """Compute the AUC of a model on each cross-validation fold.

    Args:
        model: Pipeline including its preprocessing.
        X: Training data.
        y: Target.

    Returns:
        Array of the AUC of each fold.
    """
    return cross_val_score(model, X, y, cv=make_cv(), scoring="roc_auc", n_jobs=-1)


def tune_auc(model, param_grid, X, y):
    """Choose the hyperparameters maximizing the cross-validated AUC.

    Args:
        model: Pipeline including its preprocessing.
        param_grid: Grid of hyperparameters.
        X: Training data.
        y: Target.

    Returns:
        Fitted GridSearchCV, with the best model refitted on all of `X`.
    """
    search = GridSearchCV(model, param_grid, cv=make_cv(), scoring="roc_auc", n_jobs=-1)
    return search.fit(X, y)


def best_fold_scores(search):
    """Get the AUC of each fold for the best hyperparameters of a search.

    Args:
        search: Fitted GridSearchCV.

    Returns:
        Array of the AUC of each fold.
    """
    i = search.best_index_
    return np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(search.n_splits_)])


def report(name, scores):
    """Print the mean AUC, its standard deviation across folds and the AUC of each fold.

    Args:
        name: Label of the model.
        scores: AUC of each fold.
    """
    folds = " ".join(f"{s:.4f}" for s in scores)
    print(f"{name} : AUC CV = {scores.mean():.4f} (+/- {scores.std():.4f})  [plis : {folds}]")


def report_grid(search, param):
    """Print the mean AUC for each tested value of a hyperparameter.

    Args:
        search: Fitted GridSearchCV.
        param: Name of the hyperparameter in the pipeline.
    """
    results = search.cv_results_
    for value, mean, std in zip(results[f"param_{param}"], results["mean_test_score"], results["std_test_score"]):
        print(f"    {param.split('__')[-1]} = {value:<8} AUC = {mean:.4f} (+/- {std:.4f})")
