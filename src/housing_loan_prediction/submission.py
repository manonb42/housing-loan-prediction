"""Soumission du meilleur modèle linéaire : lasso (C = 0.3) sans les variables de DROP_SURE.

Lancer avec : uv run python -u -m housing_loan_prediction.submission
"""

import pandas as pd

from housing_loan_prediction.data import DATA_DIR, load_clean
from housing_loan_prediction.logistic import build_model
from housing_loan_prediction.preprocessing import DROP_SURE, ID_COL, TARGET

BEST_C = 0.3


def make_submission(model, train_clean, y, test_clean, path):
    """Fit a model on the whole training set and write its test predictions.

    Args:
        model: Pipeline including its preprocessing.
        train_clean: Cleaned training set.
        y: Target.
        test_clean: Cleaned test set.
        path: Path of the submission file.

    Returns:
        DataFrame of the submission (id, probability of acceptance).
    """
    model.fit(train_clean, y)
    submission = pd.DataFrame({ID_COL: test_clean[ID_COL], TARGET: model.predict_proba(test_clean)[:, 1]})
    submission.to_csv(path, index=False)
    return submission


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()
    model = build_model(train_clean, DROP_SURE, "lasso").set_params(logisticregression__C=BEST_C)
    submission = make_submission(model, train_clean, y, test_clean, DATA_DIR.parent / "submissions" / "lasso_drop_sure.csv")
    print(submission.shape)
    print(submission[TARGET].describe())
