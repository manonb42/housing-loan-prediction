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

# Variables numériques des données brutes
RAW_NUM_COLS = [
    "loan_amount", "loan_to_value_ratio", "property_value", "income", "loan_term", "total_units",
    "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]
TRACT_COLS = RAW_NUM_COLS[6:]

# Codes « No co-applicant » des six colonnes de profil du co-emprunteur (étape 3.5)
NO_COAPP_CODES = {
    "co-applicant_sex": 5, "co-applicant_race-1": 8, "co-applicant_ethnicity-1": 5,
    "co-applicant_sex_observed": 4, "co-applicant_race_observed": 4, "co-applicant_ethnicity_observed": 4,
}

# Regroupements de modalités rares (étape 5.5)
CREDIT_SCORE_MERGE = {5: 6, 13: 14}  # VantageScore 2.0 -> 3.0, FICO Score 10 -> 10T
APPLICANT_RACE_MERGE = {24: 2, 41: 4, 42: 4, 43: 4}
COAPPLICANT_RACE_MERGE = {24: 2, 25: 2, 41: 4, 42: 4, 43: 4, 44: 4}
OUTSIDE_STATES = {"GU": "Territoire hors États", "VI": "Territoire hors États"}
MSA_OUTSIDE = 99999
RARE_THRESHOLD = 30

# Tranches de debt_to_income_ratio -> centre de la tranche (les valeurs 36 à 49 sont déjà des nombres)
DTI_TRANCHES = {"<20%": 10, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}

# Nouvelles variables
NEW_NUM_COLS = ["loan_to_property", "loan_to_income", "dti_numeric"]
FLAG_COLS = [
    "has_coapplicant", "applicant_not_person",
    "ltv_missing", "property_value_missing", "income_missing", "loan_term_missing", "tract_unknown",
]

NUM_COLS = RAW_NUM_COLS + NEW_NUM_COLS
# Montants très asymétriques, passés au log pour les modèles linéaires (étape 4.6)
LOG_COLS = ["loan_amount", "property_value", "income", "loan_to_property", "loan_to_income"]

# Variables à retirer (étape 6.7)
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
    """Zones MSA de moins de `threshold` dossiers dans le train (comptage seul, sans la cible)."""
    counts = train["derived_msa-md"].value_counts()
    return set(counts[counts < threshold].index)


def find_frequent_counties(train, threshold=RARE_THRESHOLD):
    """Comtés d'au moins `threshold` dossiers dans le train (comptage seul, sans la cible).

    On garde la liste des comtés fréquents plutôt que celle des rares : un comté du test
    absent du train est ainsi regroupé lui aussi.
    """
    counts = train["county_code"].value_counts()
    return set(counts[counts >= threshold].index)


def categorical_columns(df):
    """Toutes les colonnes explicatives qui ne sont ni numériques ni des indicateurs."""
    excluded = set(NUM_COLS) | set(FLAG_COLS) | {ID_COL, TARGET}
    return [col for col in df.columns if col not in excluded]


def clean(df, rare_msa, frequent_counties, is_train=False):
    """Applique toutes les corrections et recodages décidés pendant l'exploration.

    `rare_msa` et `frequent_counties` viennent de `find_rare_msa(train)` et
    `find_frequent_counties(train)`, pour que le train et le test soient recodés pareil.
    Avec `is_train=True`, les 5 dossiers aberrants du train sont retirés ; le test n'est jamais filtré.
    """
    out = df.copy()

    # Étape 4 : dossiers aberrants (LTV jusqu'à 600 millions, revenu de 71 M$, bien estimé à 1,58 Md$
    # pour un prêt de 62 M$ alors que le LTV déclaré est de 51 %)
    if is_train:
        absurd = (
            (out["loan_to_value_ratio"] > 1000) | (out["income"] == 71000) | (out["property_value"] > 1e9)
        )
        out = out[~absurd].copy()

    # Âge médian du bâti à -1 (5 dossiers du train, 1 du test) : valeur impossible, passée à NaN
    out.loc[out["tract_median_age_of_housing_units"] == -1, "tract_median_age_of_housing_units"] = np.nan

    # Étape 3 : indicateurs construits avant tout recodage des colonnes de profil
    no_coapp_votes = sum((out[col] == code).astype(int) for col, code in NO_COAPP_CODES.items())
    out["has_coapplicant"] = (no_coapp_votes < 4).astype(int)
    out["applicant_not_person"] = (out["applicant_sex"] == 4).astype(int)

    # Étape 4 : LTV saisis en fraction (0,82 au lieu de 82), quand le ratio recalculé le confirme
    ratio = out["loan_amount"] / out["property_value"] * 100
    ltv = out["loan_to_value_ratio"]
    is_fraction = (ltv < 1.5) & ((ltv * 100 - ratio).abs() < 10)
    out.loc[is_fraction, "loan_to_value_ratio"] = ltv[is_fraction] * 100

    # Étape 3 : quartier inconnu (population à 0), toutes les variables de quartier passées à NaN
    out[TRACT_COLS] = out[TRACT_COLS].astype(float)
    tract_unknown = out["tract_population"] == 0
    out.loc[tract_unknown, TRACT_COLS] = np.nan

    # Étape 3 : manquants structurels
    out["total_units"] = out["total_units"].fillna(5)
    out["conforming_loan_limit"] = out["conforming_loan_limit"].fillna("NA")
    coapp_age_nan = out["co-applicant_age"].isna()
    no_coapp = out["has_coapplicant"] == 0
    out.loc[coapp_age_nan & no_coapp, "co-applicant_age"] = "Pas de co-emprunteur"
    out.loc[coapp_age_nan & ~no_coapp & (out["co-applicant_sex"] == 4), "co-applicant_age"] = "Sans objet"
    out.loc[out["applicant_age"].isna() & (out["applicant_not_person"] == 1), "applicant_age"] = "Sans objet"

    # Étape 5 : regroupement des modalités rares
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

    # Étape 4 : ratios
    out["loan_to_property"] = out["loan_amount"] / out["property_value"] * 100
    out["loan_to_income"] = out["loan_amount"] / (out["income"].where(out["income"] > 0) * 1000)

    # DTI en nombre ordonné (centre de chaque tranche), pour les arbres ; NaN si non renseigné
    dti = out["debt_to_income_ratio"]
    out["dti_numeric"] = dti.map(DTI_TRANCHES).fillna(pd.to_numeric(dti, errors="coerce"))

    # Indicateurs de manquants pour les modèles linéaires (validés après l'étape 4)
    out["ltv_missing"] = out["loan_to_value_ratio"].isna().astype(int)
    out["property_value_missing"] = out["property_value"].isna().astype(int)
    out["income_missing"] = (out["income"].isna() | (out["income"] <= 0)).astype(int)
    out["loan_term_missing"] = out["loan_term"].isna().astype(int)
    out["tract_unknown"] = tract_unknown.astype(int)

    # Catégorielles en texte (codes entiers sans décimale) ; un NaN restant devient la modalité « Manquant »
    for col in categorical_columns(out):
        values = out[col]
        if pd.api.types.is_numeric_dtype(values):
            values = values.astype("Int64").astype("string")
        out[col] = values.astype("string").fillna("Manquant").astype(object)

    return out


def log1p_positive(X):
    """log(1 + x) sur les valeurs positives ; les valeurs négatives sont ramenées à 0."""
    return np.log1p(np.clip(X, 0, None))


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Borne chaque colonne entre deux quantiles appris sur les données d'entraînement."""

    def __init__(self, lower=0.001, upper=0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def feature_columns(df, drop=()):
    """Colonnes numériques (avec et sans log), indicateurs et catégorielles, hors variables retirées."""
    drop = set(drop)
    log_cols = [c for c in LOG_COLS if c not in drop]
    num_cols = [c for c in NUM_COLS if c not in drop and c not in LOG_COLS]
    flag_cols = [c for c in FLAG_COLS if c not in drop]
    cat_cols = [c for c in categorical_columns(df) if c not in drop]
    return log_cols, num_cols, flag_cols, cat_cols


def build_linear_preprocessor(df, drop=DROP_SURE):
    """Transformations apprises pour les modèles linéaires (régression logistique, lasso).

    Numériques : médiane du train pour les NaN, log des montants asymétriques, bornage aux
    quantiles 0,1 % et 99,9 %, centrage-réduction. Indicateurs 0/1 : inchangés.
    Catégorielles : one-hot ; une modalité du test absente du train donne une ligne de zéros.
    `df` sert seulement à lister les colonnes (sortie de `clean`).
    """
    # Le DTI reste en one-hot : son lien avec l'acceptation est en U inversé, qu'une seule pente ne capte pas
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
    """Transformations pour les arbres (arbre de décision, forêt aléatoire).

    Un arbre ne coupe que sur l'ordre des valeurs : ni log, ni bornage, ni centrage-réduction.
    Numériques et indicateurs : inchangés, les NaN sont gérés directement par les arbres.
    Catégorielles : un entier par modalité ; une modalité du test absente du train vaut -1.
    """
    # Le DTI passe en nombre ordonné : l'arbre retrouve lui-même les seuils de la courbe en U
    log_cols, num_cols, flag_cols, cat_cols = feature_columns(df, [*drop, "debt_to_income_ratio"])
    return ColumnTransformer([
        ("num", "passthrough", log_cols + num_cols + flag_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])
