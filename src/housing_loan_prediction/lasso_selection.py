"""Régression logistique sans pénalité, avant et après retrait des variables mises à zéro par le lasso.

Lancer avec : uv run python -u -m housing_loan_prediction.lasso_selection
"""

from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import cv_auc, report
from housing_loan_prediction.logistic import build_model
from housing_loan_prediction.preprocessing import DROP_SURE

# Variables entièrement mises à zéro par le lasso (C = 0.3), lu dans la sortie de logistic.py
LASSO_ZERO_ALL = ["applicant_not_person", "co-applicant_race_observed", "has_coapplicant", "negative_amortization"]
LASSO_ZERO_DROP_SURE = ["applicant_not_person", "has_coapplicant", "negative_amortization"]

CONFIGS = {
    "toutes les variables": [],
    "toutes les variables - zéros du lasso": LASSO_ZERO_ALL,
    "DROP_SURE": DROP_SURE,
    "DROP_SURE - zéros du lasso": DROP_SURE + LASSO_ZERO_DROP_SURE,
}


if __name__ == "__main__":
    train_clean, test_clean, y = load_clean()
    for name, drop in CONFIGS.items():
        report(f"sans pénalité, {name}", cv_auc(build_model(train_clean, drop, "sans pénalité"), train_clean, y))
