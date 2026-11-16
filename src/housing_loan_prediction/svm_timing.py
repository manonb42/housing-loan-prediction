"""Temps d'apprentissage du SVM à noyau gaussien (RBF) selon le nombre de dossiers.

Lancer avec : uv run python -u -m housing_loan_prediction.svm_timing
(machine au repos : un autre calcul en parallèle fausse les temps)

On estime l'exposant a de t = k * n^a par régression de log(t) sur log(n),
puis on extrapole à la taille d'un pli d'apprentissage (4/5 du train).
"""

import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from housing_loan_prediction.data import load_clean
from housing_loan_prediction.evaluation import RANDOM_STATE
from housing_loan_prediction.preprocessing import DROP_SURE, build_linear_preprocessor

SIZES = [5_000, 10_000, 20_000, 40_000]

if __name__ == "__main__":
    train_clean, _, y = load_clean()
    rng = np.random.default_rng(RANDOM_STATE)
    times = []
    for n in SIZES:
        idx = rng.choice(len(train_clean), size=n, replace=False)
        model = make_pipeline(build_linear_preprocessor(train_clean, drop=DROP_SURE), SVC(kernel="rbf"))
        start = time.perf_counter()
        model.fit(train_clean.iloc[idx], y.iloc[idx])
        times.append(time.perf_counter() - start)
        print(f"n = {n:>6} : {times[-1]:7.1f} s", flush=True)

    a, log_k = np.polyfit(np.log(SIZES), np.log(times), 1)
    n_fold = int(len(train_clean) * 4 / 5)
    print(f"\nExposant estimé : t ~ n^{a:.2f}")
    print(f"Extrapolation à n = {n_fold} (un pli) : {np.exp(log_k) * n_fold ** a / 60:.0f} min")
