"""Combinaison de modèles à partir de leurs prédictions hors pli (out-of-fold).

Lancer avec : uv run python -u -m housing_loan_prediction.stacking [oof [modèles] | compare | submit <méthodes>]
(méthodes de submit : mots-clés, ex. `submit ges moyenne logistique`)

1) `oof` : prédictions hors pli du lasso, de la forêt, du SVM linéaire et de HistGB, sur les mêmes 5 plis
   (réglages déjà retenus, pas de nouvelle recherche). Celles de CatBoost sont écrites par
   `boosting_libs curves catboost`.
2) `compare` : diversité des modèles (corrélations), puis cinq façons de combiner, évaluées par une
   validation croisée au niveau du méta-modèle : la combinaison est apprise sur 4 plis de prédictions
   hors pli et notée sur le 5e. Noter sur les données qui ont servi à choisir les poids serait optimiste.
3) `submit <méthodes>` : combinaison apprise sur toutes les prédictions hors pli, appliquée aux
   prédictions test des modèles appris sur tout le train (fichiers de `submissions/`), écrite dans
   `submissions/ensemble_<méthode>.csv`.
"""

import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from housing_loan_prediction.boosting_libs import OOF_DIR, ROOT
from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import make_cv
from housing_loan_prediction.gradient_boosting import BEST_GB, build_gb, check_submission
from housing_loan_prediction.logistic import build_model
from housing_loan_prediction.preprocessing import DROP_SURE, ID_COL, TARGET
from housing_loan_prediction.submission import BEST_C
from housing_loan_prediction.svm import build_linear_svm
from housing_loan_prediction.trees import BEST_FOREST, build_forest

SUBMISSIONS = ROOT / "submissions"
# name -> (out-of-fold predictions, test predictions of the model fitted on the whole training set)
CANDIDATES = {
    "catboost": (OOF_DIR / "catboost.csv", SUBMISSIONS / "catboost.csv"),
    "hist_gb": (OOF_DIR / "hist_gb.csv", SUBMISSIONS / "gradient_boosting.csv"),
    "forêt": (OOF_DIR / "random_forest.csv", SUBMISSIONS / "random_forest.csv"),
    "lasso": (OOF_DIR / "lasso.csv", SUBMISSIONS / "lasso_drop_sure.csv"),
    "svm_lin": (OOF_DIR / "svm_lineaire.csv", SUBMISSIONS / "svm_lineaire.csv"),
}
SVM_C = 0.003
GES_STEPS = 50
EPS = 1e-6


def build_svm(train_clean):
    """Build the linear SVM with Platt calibration.

    Args:
        train_clean: Cleaned training set.

    Returns:
        CalibratedClassifierCV around the linear SVM.
    """
    return CalibratedClassifierCV(build_linear_svm(train_clean, C=SVM_C), method="sigmoid", cv=3)


def base_models(train_clean):
    """Build the base models whose out-of-fold predictions are computed here.

    Args:
        train_clean: Cleaned training set.

    Returns:
        Dictionary from the model name to its unfitted pipeline.
    """
    return {
        "lasso": build_model(train_clean, DROP_SURE, "lasso").set_params(logisticregression__C=BEST_C),
        "random_forest": build_forest(train_clean, **BEST_FOREST),
        "svm_lineaire": build_svm(train_clean),
        "hist_gb": build_gb(train_clean, **BEST_GB),
    }


def write_oof(train_clean, test_clean, y, names):
    """Write the out-of-fold predictions of the base models.

    Args:
        train_clean: Cleaned training set.
        test_clean: Cleaned test set.
        y: Target.
        names: Names of the models to run.
    """
    for name, model in base_models(train_clean).items():
        if name not in names:
            continue
        oof = cross_val_predict(model, train_clean, y, cv=make_cv(), method="predict_proba")[:, 1]
        pd.DataFrame({ID_COL: train_clean[ID_COL], TARGET: oof}).to_csv(OOF_DIR / f"{name}.csv", index=False)
        print(f"  {name} : AUC hors pli = {roc_auc_score(y, oof):.4f} -> {OOF_DIR / f'{name}.csv'}")
        if name == "svm_lineaire":
            model.fit(train_clean, y)
            submission = pd.DataFrame({ID_COL: test_clean[ID_COL], TARGET: model.predict_proba(test_clean)[:, 1]})
            submission.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
            print(f"  Soumission écrite : {SUBMISSIONS / f'{name}.csv'}")
            check_submission(submission, oof, y, {n: p[1] for n, p in CANDIDATES.items() if n != "svm_lin"})


def load_predictions(which, ids):
    """Load the predictions of the candidate models.

    Args:
        which: 0 for the out-of-fold predictions, 1 for the test predictions.
        ids: Ids giving the order of the rows.

    Returns:
        DataFrame with one column per model.
    """
    cols = {}
    for name, paths in CANDIDATES.items():
        cols[name] = pd.read_csv(paths[which]).set_index(ID_COL)[TARGET].reindex(ids).to_numpy()
    return pd.DataFrame(cols)


def fit_best_single(P, y):
    """Select the model with the best AUC.

    Args:
        P: Predictions, one column per model.
        y: Target.

    Returns:
        Tuple (function from predictions to scores, weights).
    """
    best = max(P.columns, key=lambda c: roc_auc_score(y, P[c]))
    return lambda Q: Q[best].to_numpy(), {best: 1.0}


def fit_mean(P, y):
    """Average the probabilities of all the models.

    Args:
        P: Predictions, one column per model.
        y: Target, unused.

    Returns:
        Tuple (function from predictions to scores, weights).
    """
    return lambda Q: Q.mean(axis=1).to_numpy(), {c: 1 / P.shape[1] for c in P.columns}


def to_ranks(Q):
    """Replace each column by its normalized ranks.

    Args:
        Q: Predictions, one column per model.

    Returns:
        DataFrame of ranks divided by the number of rows.
    """
    return Q.apply(lambda col: rankdata(col) / len(col))


def fit_rank_mean(P, y):
    """Average the ranks of all the models.

    Args:
        P: Predictions, one column per model.
        y: Target, unused.

    Returns:
        Tuple (function from predictions to scores, weights).
    """
    return lambda Q: to_ranks(Q).mean(axis=1).to_numpy(), {c: 1 / P.shape[1] for c in P.columns}


def fit_ges(P, y, steps=GES_STEPS):
    """Fit a greedy ensemble selection (Caruana et al., 2004).

    Args:
        P: Predictions, one column per model.
        y: Target.
        steps: Number of greedy steps, with replacement.

    Returns:
        Tuple (function from predictions to scores, weights).
    """
    X = P.to_numpy()
    counts, total = np.zeros(X.shape[1]), np.zeros(len(X))
    for t in range(steps):
        aucs = [roc_auc_score(y, (total + X[:, j]) / (t + 1)) for j in range(X.shape[1])]
        j = int(np.argmax(aucs))
        counts[j] += 1
        total += X[:, j]
    w = counts / counts.sum()
    return lambda Q: Q.to_numpy() @ w, dict(zip(P.columns, w.round(3)))


def fit_logistic(P, y):
    """Fit a logistic regression on the logits of the predictions.

    Args:
        P: Predictions, one column per model.
        y: Target.

    Returns:
        Tuple (function from predictions to scores, coefficients).
    """
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(P.clip(EPS, 1 - EPS)), y)
    return (lambda Q: meta.decision_function(logit(Q.clip(EPS, 1 - EPS))),
            dict(zip(P.columns, meta.coef_.ravel().round(3))))


METHODS = {
    "meilleur modèle seul": fit_best_single,
    "moyenne des probabilités": fit_mean,
    "moyenne des rangs": fit_rank_mean,
    "GES (sélection gloutonne)": fit_ges,
    "stacking logistique": fit_logistic,
}


def describe(P, y):
    """Print the AUC of each model and the correlations of their predictions and residuals.

    Args:
        P: Out-of-fold predictions, one column per model.
        y: Target.
    """
    print("\n  AUC hors pli de chaque modèle :")
    for c in P.columns:
        print(f"    {c:<10} {roc_auc_score(y, P[c]):.4f}")
    print("\n  Corrélation des rangs des prédictions (Spearman) :")
    print(pd.DataFrame(spearmanr(P)[0], index=P.columns, columns=P.columns).round(3).to_string())
    print("\n  Corrélation des résidus y - p (des modèles qui se trompent pareil n'apportent rien à l'ensemble) :")
    print((P.rsub(y, axis=0)).corr().round(3).to_string())


def compare(P, y):
    """Print the meta-level cross-validated AUC of each combination method.

    Args:
        P: Out-of-fold predictions, one column per model.
        y: Target.
    """
    print("\n  Validation croisée au niveau du méta-modèle (combinaison apprise sur 4 plis, notée sur le 5e) :")
    folds = list(make_cv().split(P, y))
    for name, fit in METHODS.items():
        scores = []
        for tr, va in folds:
            combine, _ = fit(P.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[va], combine(P.iloc[va])))
        _, weights = fit(P, y)
        scores = np.array(scores)
        print(f"    {name:<27} AUC = {scores.mean():.4f} (+/- {scores.std():.4f})  "
              f"[{' '.join(f'{s:.4f}' for s in scores)}]")
        print(f"      poids appris sur tout : {weights}")


def to_proba(method, score):
    """Map a combined score to [0, 1] without changing its order.

    Args:
        method: Name of the combination method.
        score: Combined score.

    Returns:
        Probabilities, or normalized ranks for the rank mean.
    """
    if method == "stacking logistique":
        return expit(score)
    if method == "moyenne des rangs":
        return rankdata(score) / len(score)
    return score


SUBMISSION_NAMES = {
    "moyenne des probabilités": "ensemble_moyenne.csv",
    "moyenne des rangs": "ensemble_moyenne_rangs.csv",
    "GES (sélection gloutonne)": "ensemble_ges.csv",
    "stacking logistique": "ensemble_stacking_logistique.csv",
}


def write_submission(method, P, y, test_clean):
    """Fit a combination on the out-of-fold predictions and write its test predictions.

    Args:
        method: Name of the combination method, key of `METHODS`.
        P: Out-of-fold predictions, one column per model.
        y: Target.
        test_clean: Cleaned test set.
    """
    combine, weights = METHODS[method](P, y)
    print(f"\n=== {method} : poids {weights} ===")
    proba = to_proba(method, combine(load_predictions(1, test_clean[ID_COL])))
    submission = pd.DataFrame({ID_COL: test_clean[ID_COL], TARGET: proba})
    path = SUBMISSIONS / SUBMISSION_NAMES[method]
    submission.to_csv(path, index=False)
    print(f"  Soumission écrite : {path}")
    check_submission(submission, to_proba(method, combine(P)), y,
                     {name: paths[1] for name, paths in CANDIDATES.items()})


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()
    y = y.to_numpy()
    step = sys.argv[1] if len(sys.argv) > 1 else "compare"

    if step == "oof":
        write_oof(train_clean, test_clean, y, sys.argv[2:] or list(base_models(train_clean)))
    else:
        P = load_predictions(0, train_clean[ID_COL])
        if step == "compare":
            describe(P, y)
            compare(P, y)
        if step == "submit":
            for key in sys.argv[2:]:
                method = next(m for m in SUBMISSION_NAMES if key.lower() in m.lower())
                write_submission(method, P, y, test_clean)
