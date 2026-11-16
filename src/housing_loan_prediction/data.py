"""Chargement des données brutes et application du nettoyage de `preprocessing.clean`."""

from pathlib import Path

import pandas as pd

from housing_loan_prediction.preprocessing import TARGET, clean, find_frequent_counties, find_rare_msa

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw(data_dir=DATA_DIR):
    """Read the raw training and test sets.

    Args:
        data_dir: Folder containing train.csv and test.csv.

    Returns:
        Tuple (train, test) of DataFrames.
    """
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    return train, test


def load_clean(data_dir=DATA_DIR):
    """Read and clean the training and test sets.

    Args:
        data_dir: Folder containing train.csv and test.csv.

    Returns:
        Tuple (train_clean, test_clean, y), cleaned with the same rare MSA and county lists.
    """
    train, test = load_raw(data_dir)
    rare_msa, frequent_counties = find_rare_msa(train), find_frequent_counties(train)
    train_clean = clean(train, rare_msa, frequent_counties, is_train=True)
    test_clean = clean(test, rare_msa, frequent_counties)
    return train_clean, test_clean, train_clean[TARGET]
