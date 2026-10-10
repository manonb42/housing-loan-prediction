"""SVM (machines à vecteurs de support) : AUC en validation croisée.

Lancer avec : uv run python -u -m housing_loan_prediction.svm

Le SVM ne prédit pas une probabilité mais un score, la distance signée à la frontière
(`decision_function`) : positif = côté « accepté », d'autant plus sûr qu'on est loin de la marge.
L'AUC ne dépend que de l'ordre des dossiers, ce score suffit donc pour la calculer.

Hyperparamètre du SVM linéaire : C, le prix d'une erreur sur le train (C = 1 / lambda,
comme pour la logistique) : un petit C donne une marge large et un modèle très régularisé.
"""

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import best_fold_scores, report, report_grid, tune_auc
from housing_loan_prediction.preprocessing import DROP_SURE, build_linear_preprocessor

LINEAR_C_PARAM = "linearsvc__C"
LINEAR_C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]


def build_linear_svm(train_clean, C=1.0):
    return make_pipeline(
        build_linear_preprocessor(train_clean, drop=DROP_SURE),
        LinearSVC(C=C, max_iter=5000),
    )


def run_linear(train_clean, y):
    search = tune_auc(build_linear_svm(train_clean), {LINEAR_C_PARAM: LINEAR_C_GRID}, train_clean, y)
    print(f"SVM linéaire : meilleur C = {search.best_params_[LINEAR_C_PARAM]}")
    report_grid(search, LINEAR_C_PARAM)
    report("  -> SVM linéaire", best_fold_scores(search))
    return search


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()

    print("=== 1) SVM linéaire sur tout le train (DROP_SURE), grille sur C ===")
    run_linear(train_clean, y)
