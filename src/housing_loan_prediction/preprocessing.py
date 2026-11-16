"""Preprocessing des données HMDA, issu de l'exploration de notebooks/01_eda_preprocessing.ipynb.

Deux temps :
1. `clean` : corrections et recodages ligne à ligne, identiques pour le train et le test.
   Aucune statistique n'y est apprise à partir de la cible.
2. `build_linear_preprocessor` : transformations apprises sur les données d'entraînement
   (médianes, bornes, moyennes, écarts-types, modalités). À placer dans un `Pipeline`
   scikit-learn avec le modèle, pour qu'elles soient réapprises dans chaque pli de
   validation croisée.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "accepted"
ID_COL = "id"

RAW_NUM_COLS = [
    "loan_amount", "loan_to_value_ratio", "property_value", "income", "loan_term", "total_units",
    "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]
TRACT_COLS = RAW_NUM_COLS[6:]

NO_COAPP_CODES = {
    "co-applicant_sex": 5, "co-applicant_race-1": 8, "co-applicant_ethnicity-1": 5,
    "co-applicant_sex_observed": 4, "co-applicant_race_observed": 4, "co-applicant_ethnicity_observed": 4,
}

CREDIT_SCORE_MERGE = {5: 6, 13: 14}  # VantageScore 2.0 -> 3.0, FICO Score 10 -> 10T
APPLICANT_RACE_MERGE = {24: 2, 41: 4, 42: 4, 43: 4}
COAPPLICANT_RACE_MERGE = {24: 2, 25: 2, 41: 4, 42: 4, 43: 4, 44: 4}
OUTSIDE_STATES = {"GU": "Territoire hors États", "VI": "Territoire hors États"}
MSA_OUTSIDE = 99999
RARE_THRESHOLD = 30

# Values 36 to 49 are already numbers
DTI_TRANCHES = {"<20%": 10, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}

NEW_NUM_COLS = ["loan_to_property", "loan_to_income", "dti_numeric"]
FLAG_COLS = [
    "has_coapplicant", "applicant_not_person",
    "ltv_missing", "property_value_missing", "income_missing", "loan_term_missing", "tract_unknown",
]

NUM_COLS = RAW_NUM_COLS + NEW_NUM_COLS
LOG_COLS = ["loan_amount", "property_value", "income", "loan_to_property", "loan_to_income"]

DROP_SURE = [
    "derived_loan_product_type", "construction_method",
    "applicant_race_observed", "applicant_sex_observed",
    "co-applicant_race_observed", "co-applicant_sex_observed",
]
DROP_MAYBE = [
    "derived_race", "applicant_race-1", "co-applicant_race-1",
    "derived_ethnicity", "applicant_ethnicity-1", "co-applicant_ethnicity-1",
    "derived_sex", "applicant_sex", "co-applicant_sex",
    "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
    "county_code",
    "business_or_commercial_purpose",
    "tract_owner_occupied_units",
]


def find_rare_msa(train, threshold=RARE_THRESHOLD):
    """Find the MSA codes with few applications in the training set.

    Args:
        train: Raw training set.
        threshold: Minimum number of applications for an MSA to be kept.

    Returns:
        Set of MSA codes with fewer than `threshold` applications.
    """
    counts = train["derived_msa-md"].value_counts()
    return set(counts[counts < threshold].index)


def find_frequent_counties(train, threshold=RARE_THRESHOLD):
    """Find the counties with enough applications in the training set.

    Args:
        train: Raw training set.
        threshold: Minimum number of applications for a county to be kept.

    Returns:
        Set of county codes with at least `threshold` applications.
    """
    counts = train["county_code"].value_counts()
    return set(counts[counts >= threshold].index)


def categorical_columns(df):
    """List the categorical explanatory columns.

    Args:
        df: Cleaned data set.

    Returns:
        Columns that are neither numerical, indicators, the id nor the target.
    """
    excluded = set(NUM_COLS) | set(FLAG_COLS) | {ID_COL, TARGET}
    return [col for col in df.columns if col not in excluded]


def clean(df, rare_msa, frequent_counties, is_train=False):
    """Apply the corrections and recodings decided during the exploration.

    Args:
        df: Raw data set (train or test).
        rare_msa: MSA codes to merge, from `find_rare_msa(train)`.
        frequent_counties: Counties to keep, from `find_frequent_counties(train)`.
        is_train: Whether to drop the 5 aberrant applications of the training set.

    Returns:
        Cleaned copy of `df`, with the derived features.
    """
    out = df.copy()

    if is_train:
        absurd = (
            (out["loan_to_value_ratio"] > 1000) | (out["income"] == 71000) | (out["property_value"] > 1e9)
        )
        out = out[~absurd].copy()

    out.loc[out["tract_median_age_of_housing_units"] == -1, "tract_median_age_of_housing_units"] = np.nan

    no_coapp_votes = sum((out[col] == code).astype(int) for col, code in NO_COAPP_CODES.items())
    out["has_coapplicant"] = (no_coapp_votes < 4).astype(int)
    out["applicant_not_person"] = (out["applicant_sex"] == 4).astype(int)

    ratio = out["loan_amount"] / out["property_value"] * 100
    ltv = out["loan_to_value_ratio"]
    is_fraction = (ltv < 1.5) & ((ltv * 100 - ratio).abs() < 10)
    out.loc[is_fraction, "loan_to_value_ratio"] = ltv[is_fraction] * 100

    out[TRACT_COLS] = out[TRACT_COLS].astype(float)
    tract_unknown = out["tract_population"] == 0
    out.loc[tract_unknown, TRACT_COLS] = np.nan

    out["total_units"] = out["total_units"].fillna(5)
    out["conforming_loan_limit"] = out["conforming_loan_limit"].fillna("NA")
    coapp_age_nan = out["co-applicant_age"].isna()
    no_coapp = out["has_coapplicant"] == 0
    out.loc[coapp_age_nan & no_coapp, "co-applicant_age"] = "Pas de co-emprunteur"
    out.loc[coapp_age_nan & ~no_coapp & (out["co-applicant_sex"] == 4), "co-applicant_age"] = "Sans objet"
    out.loc[out["applicant_age"].isna() & (out["applicant_not_person"] == 1), "applicant_age"] = "Sans objet"

    for col in ["applicant_credit_score_type", "co-applicant_credit_score_type"]:
        out[col] = out[col].replace(CREDIT_SCORE_MERGE)
    out["derived_ethnicity"] = out["derived_ethnicity"].replace({"Free Form Text Only": "Ethnicity Not Available"})
    out["derived_race"] = out["derived_race"].replace({"Free Form Text Only": "Race Not Available"})
    for col in ["applicant_ethnicity-1", "co-applicant_ethnicity-1"]:
        out[col] = out[col].fillna(3)  # 3 = Information not provided
    out["applicant_race-1"] = out["applicant_race-1"].fillna(6).replace(APPLICANT_RACE_MERGE)
    out["co-applicant_race-1"] = out["co-applicant_race-1"].fillna(6).replace(COAPPLICANT_RACE_MERGE)
    out["state_code"] = out["state_code"].replace(OUTSIDE_STATES)
    out.loc[out["derived_msa-md"].isin(rare_msa), "derived_msa-md"] = MSA_OUTSIDE
    county = out["county_code"]
    rare_county = county.notna() & ~county.isin(frequent_counties)
    out["county_code"] = county.astype("Int64").astype("string").astype(object)
    out.loc[rare_county, "county_code"] = "Autre comté " + out.loc[rare_county, "state_code"].fillna("?").astype(str)

    out["loan_to_property"] = out["loan_amount"] / out["property_value"] * 100
    out["loan_to_income"] = out["loan_amount"] / (out["income"].where(out["income"] > 0) * 1000)

    dti = out["debt_to_income_ratio"]
    out["dti_numeric"] = dti.map(DTI_TRANCHES).fillna(pd.to_numeric(dti, errors="coerce"))

    out["ltv_missing"] = out["loan_to_value_ratio"].isna().astype(int)
    out["property_value_missing"] = out["property_value"].isna().astype(int)
    out["income_missing"] = (out["income"].isna() | (out["income"] <= 0)).astype(int)
    out["loan_term_missing"] = out["loan_term"].isna().astype(int)
    out["tract_unknown"] = tract_unknown.astype(int)

    for col in categorical_columns(out):
        values = out[col]
        if pd.api.types.is_numeric_dtype(values):
            values = values.astype("Int64").astype("string")
        out[col] = values.astype("string").fillna("Manquant").astype(object)

    return out


def log1p_positive(X):
    """Apply log(1 + x) to non-negative values.

    Args:
        X: Array of numerical values.

    Returns:
        log(1 + x), with negative values set to 0 first.
    """
    return np.log1p(np.clip(X, 0, None))


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column between two quantiles learned on the training data."""

    def __init__(self, lower=0.001, upper=0.999):
        """Initialize the clipper.

        Args:
            lower: Lower quantile.
            upper: Upper quantile.
        """
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        """Learn the quantiles of each column.

        Args:
            X: Training data.
            y: Ignored.

        Returns:
            The fitted clipper.
        """
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Clip each column between its learned quantiles.

        Args:
            X: Data to transform.

        Returns:
            Clipped array.
        """
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        """Return the names of the output columns.

        Args:
            input_features: Names of the input columns.

        Returns:
            The input names, unchanged.
        """
        return np.asarray(input_features, dtype=object)


def feature_columns(df, drop=()):
    """Split the explanatory columns by type.

    Args:
        df: Cleaned data set.
        drop: Columns to leave out.

    Returns:
        Lists of log-transformed numerical, other numerical, indicator and categorical columns.
    """
    drop = set(drop)
    log_cols = [c for c in LOG_COLS if c not in drop]
    num_cols = [c for c in NUM_COLS if c not in drop and c not in LOG_COLS]
    flag_cols = [c for c in FLAG_COLS if c not in drop]
    cat_cols = [c for c in categorical_columns(df) if c not in drop]
    return log_cols, num_cols, flag_cols, cat_cols


def build_linear_preprocessor(df, drop=DROP_SURE):
    """Build the preprocessing of the linear models.

    Args:
        df: Cleaned data set, used only to list the columns.
        drop: Columns to leave out.

    Returns:
        ColumnTransformer with median imputation, log, quantile clipping and scaling of the numerical
        columns, and one-hot encoding of the categorical columns.
    """
    log_cols, num_cols, flag_cols, cat_cols = feature_columns(df, [*drop, "dti_numeric"])
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(log1p_positive, feature_names_out="one-to-one"),
        QuantileClipper(),
        StandardScaler(),
    )
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), QuantileClipper(), StandardScaler())
    return ColumnTransformer([
        ("log", log_pipeline, log_cols),
        ("num", num_pipeline, num_cols),
        ("flag", "passthrough", flag_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def build_tree_preprocessor(df, drop=DROP_SURE):
    """Build the preprocessing of the tree models.

    Args:
        df: Cleaned data set, used only to list the columns.
        drop: Columns to leave out.

    Returns:
        ColumnTransformer passing the numerical columns through and encoding each category as an integer.
    """
    log_cols, num_cols, flag_cols, cat_cols = feature_columns(df, [*drop, "debt_to_income_ratio"])
    return ColumnTransformer([
        ("num", "passthrough", log_cols + num_cols + flag_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])
