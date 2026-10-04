"""Chargement des données brutes et application du nettoyage de `preprocessing.clean`."""

from pathlib import Path

import pandas as pd

from housing_loan_prediction.preprocessing import TARGET, clean, find_rare_msa

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw(data_dir=DATA_DIR):
    """Lit train.csv et test.csv tels quels."""
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    return train, test


def load_clean(data_dir=DATA_DIR):
    """Renvoie (train_clean, test_clean, y), nettoyés avec les mêmes zones MSA rares."""
    train, test = load_raw(data_dir)
    rare_msa = find_rare_msa(train)
    train_clean = clean(train, rare_msa, is_train=True)
    test_clean = clean(test, rare_msa)
    return train_clean, test_clean, train_clean[TARGET]
