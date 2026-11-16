"""Arbre de décision élagué et forêt aléatoire : AUC en validation croisée.

Lancer avec : uv run python -m housing_loan_prediction.trees
Fichier de soumission : uv run python -m housing_loan_prediction.trees submit

Arbre seul (CART) : on fait pousser l'arbre complet, puis on l'élague par coût-complexité.
`ccp_alpha` est le prix payé par feuille : plus il est grand, plus l'arbre élagué est petit.
On choisit `ccp_alpha` par validation croisée.

Forêt aléatoire : les arbres ne sont pas élagués. Chacun est appris sur un échantillon bootstrap
et ne regarde que `max_features` variables tirées au hasard à chaque coupure. C'est la moyenne
des arbres qui réduit la variance, pas l'élagage. Hyperparamètres :
- `max_features` : le plus important, il décorrèle les arbres (petit = arbres plus différents) ;
- `min_samples_leaf` : taille minimale des feuilles, lisse les probabilités prédites ;
- `n_estimators` : pas un vrai hyperparamètre, plus il y a d'arbres mieux c'est, jusqu'à un plateau.
Chaque arbre laisse de côté environ un tiers des dossiers (out-of-bag) : leur AUC sert
d'estimation de validation sans réapprendre la forêt 5 fois, ce qui rend la grille abordable.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from housing_loan_prediction.data import DATA_DIR, load_clean
from housing_loan_prediction.evaluation import (
    RANDOM_STATE, best_fold_scores, cv_auc, report, report_grid, tune_auc,
)
from housing_loan_prediction.preprocessing import DROP_SURE, ID_COL, TARGET, build_tree_preprocessor

ALPHA_PARAM = "decisiontreeclassifier__ccp_alpha"
ALPHA_GRID = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

N_TREES = 200
MAX_FEATURES_GRID = ["sqrt", 0.2, 0.33]
MIN_SAMPLES_LEAF_GRID = [1, 5, 20]
N_TREES_GRID = [50, 100, 200, 400]

BEST_FOREST = {"max_features": 0.2, "min_samples_leaf": 20, "n_estimators": 500}
SUBMISSION_PATH = DATA_DIR.parent / "submissions" / "random_forest.csv"


def build_tree(train_clean, ccp_alpha=0.0):
    """Build a decision tree with its preprocessing.

    Args:
        train_clean: Cleaned training set.
        ccp_alpha: Cost-complexity pruning parameter.

    Returns:
        Pipeline of the tree preprocessing and the decision tree.
    """
    return make_pipeline(
        build_tree_preprocessor(train_clean, drop=DROP_SURE),
        DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=RANDOM_STATE),
    )


def build_forest(train_clean, max_features="sqrt", min_samples_leaf=1, n_estimators=N_TREES, **kwargs):
    """Build a random forest with its preprocessing.

    Args:
        train_clean: Cleaned training set.
        max_features: Fraction or rule for the number of variables drawn at each split.
        min_samples_leaf: Minimum number of applications in a leaf.
        n_estimators: Number of trees.
        **kwargs: Other parameters of RandomForestClassifier.

    Returns:
        Pipeline of the tree preprocessing and the random forest.
    """
    return make_pipeline(
        build_tree_preprocessor(train_clean, drop=DROP_SURE),
        RandomForestClassifier(
            n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples_leaf,
            oob_score=True, n_jobs=-1, random_state=RANDOM_STATE, **kwargs,
        ),
    )


def oob_auc(forest, y):
    """Compute the AUC of the out-of-bag predictions of a forest.

    Args:
        forest: Fitted RandomForestClassifier with oob_score=True.
        y: Target.

    Returns:
        Out-of-bag AUC.
    """
    proba = forest.oob_decision_function_[:, 1]
    seen = ~np.isnan(proba)
    return roc_auc_score(y[seen], proba[seen])


def run_tree(train_clean, y):
    """Evaluate the full tree, then tune its pruning by cross-validation.

    Args:
        train_clean: Cleaned training set.
        y: Target.

    Returns:
        Fitted GridSearchCV on ccp_alpha.
    """
    full = build_tree(train_clean).fit(train_clean, y)[-1]
    print(f"Arbre complet : {full.get_n_leaves()} feuilles, profondeur {full.get_depth()}")
    report("\nArbre complet", cv_auc(build_tree(train_clean), train_clean, y))

    search = tune_auc(build_tree(train_clean), {ALPHA_PARAM: ALPHA_GRID}, train_clean, y)
    pruned = search.best_estimator_[-1]
    print(f"\nArbre élagué : meilleur ccp_alpha = {search.best_params_[ALPHA_PARAM]} "
          f"({pruned.get_n_leaves()} feuilles, profondeur {pruned.get_depth()})")
    report_grid(search, ALPHA_PARAM)
    report("  -> arbre élagué", best_fold_scores(search))
    return search


def run_forest_grid(train_clean, y):
    """Compare the forests of the grid on their out-of-bag AUC.

    Args:
        train_clean: Cleaned training set.
        y: Target.

    Returns:
        Best pair (max_features, min_samples_leaf).
    """
    results = {}
    for max_features in MAX_FEATURES_GRID:
        for min_samples_leaf in MIN_SAMPLES_LEAF_GRID:
            forest = build_forest(train_clean, max_features, min_samples_leaf).fit(train_clean, y)[-1]
            results[(max_features, min_samples_leaf)] = oob_auc(forest, y)
            print(f"    max_features = {max_features:<5} min_samples_leaf = {min_samples_leaf:<3} "
                  f"AUC OOB = {results[(max_features, min_samples_leaf)]:.4f}")
    return max(results, key=results.get)


def run_n_trees(train_clean, y, max_features, min_samples_leaf):
    """Print the out-of-bag AUC of a forest as trees are added.

    Args:
        train_clean: Cleaned training set.
        y: Target.
        max_features: Fraction or rule for the number of variables drawn at each split.
        min_samples_leaf: Minimum number of applications in a leaf.
    """
    model = build_forest(train_clean, max_features, min_samples_leaf, n_estimators=N_TREES_GRID[0], warm_start=True)
    for n_trees in N_TREES_GRID:
        model.set_params(randomforestclassifier__n_estimators=n_trees).fit(train_clean, y)
        print(f"    n_estimators = {n_trees:<4} AUC OOB = {oob_auc(model[-1], y):.4f}")


def write_submission(train_clean, test_clean, y, path=SUBMISSION_PATH):
    """Fit the retained forest on the whole training set and write its test predictions.

    Args:
        train_clean: Cleaned training set.
        test_clean: Cleaned test set.
        y: Target.
        path: Path of the submission file.
    """
    model = build_forest(train_clean, **BEST_FOREST).fit(train_clean, y)
    print(f"Forêt finale : AUC OOB = {oob_auc(model[-1], y):.4f}")
    submission = pd.DataFrame({ID_COL: test_clean[ID_COL], TARGET: model.predict_proba(test_clean)[:, 1]})
    path.parent.mkdir(exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"Soumission écrite : {path} ({len(submission)} lignes)")


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()
    if sys.argv[1:] == ["submit"]:
        write_submission(train_clean, test_clean, y)
        sys.exit()
    y = y.to_numpy()

    print("=== 1) Arbre de décision : arbre complet puis élagage par coût-complexité ===")
    run_tree(train_clean, y)

    print(f"\n=== 2) Forêt aléatoire ({N_TREES} arbres) : grille sur l'AUC out-of-bag ===")
    max_features, min_samples_leaf = run_forest_grid(train_clean, y)
    print(f"  -> meilleur : max_features = {max_features}, min_samples_leaf = {min_samples_leaf}")

    print("\n=== 3) Nombre d'arbres, avec les meilleurs hyperparamètres ===")
    run_n_trees(train_clean, y, max_features, min_samples_leaf)

    print("\n=== 4) Meilleure forêt sur les mêmes plis que les autres modèles ===")
    report(f"Forêt (max_features = {max_features}, min_samples_leaf = {min_samples_leaf})",
           cv_auc(build_forest(train_clean, max_features, min_samples_leaf), train_clean, y))
