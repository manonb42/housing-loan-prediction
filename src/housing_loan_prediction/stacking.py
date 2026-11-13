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
from housing_loan_prediction.evaluation import RANDOM_STATE, make_cv
from housing_loan_prediction.gradient_boosting import BEST_GB, build_gb, check_submission
from housing_loan_prediction.logistic import build_model
from housing_loan_prediction.preprocessing import DROP_SURE, ID_COL, TARGET
from housing_loan_prediction.submission import BEST_C
from housing_loan_prediction.svm import build_linear_svm
from housing_loan_prediction.trees import BEST_FOREST, build_forest

SUBMISSIONS = ROOT / "submissions"
# nom du modèle -> (prédictions hors pli, prédictions test du modèle appris sur tout le train)
CANDIDATES = {
    "catboost": (OOF_DIR / "catboost.csv", SUBMISSIONS / "catboost.csv"),
    "hist_gb": (OOF_DIR / "hist_gb.csv", SUBMISSIONS / "gradient_boosting.csv"),
    "forêt": (OOF_DIR / "random_forest.csv", SUBMISSIONS / "random_forest.csv"),
    "lasso": (OOF_DIR / "lasso.csv", SUBMISSIONS / "lasso_drop_sure.csv"),
    "svm_lin": (OOF_DIR / "svm_lineaire.csv", SUBMISSIONS / "svm_lineaire.csv"),
}
SVM_C = 0.003  # meilleur C de la grille (svm.py) : AUC CV 0.8934
GES_STEPS = 50
EPS = 1e-6


def build_svm(train_clean):
    """SVM linéaire (C retenu par la grille de svm.py). Il ne donne qu'une distance à la frontière : la
    calibration de Platt (sigmoïde apprise sur 3 sous-plis internes) la convertit en probabilité, pour
    pouvoir la moyenner avec celles des autres modèles. L'ordre des dossiers, donc l'AUC, change à peine."""
    return CalibratedClassifierCV(build_linear_svm(train_clean, C=SVM_C), method="sigmoid", cv=3)


def base_models(train_clean):
    return {
        "lasso": build_model(train_clean, DROP_SURE, "lasso").set_params(logisticregression__C=BEST_C),
        "random_forest": build_forest(train_clean, **BEST_FOREST),
        "svm_lineaire": build_svm(train_clean),
        "hist_gb": build_gb(train_clean, **BEST_GB),
    }


def write_oof(train_clean, test_clean, y, names):
    """Prédictions hors pli (mêmes 5 plis que les autres) ; pour le SVM, aussi la soumission sur le test
    (lasso et forêt en ont déjà une)."""
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
    """Matrice (dossiers x modèles), lignes dans l'ordre de `ids` ; which = 0 (hors pli) ou 1 (test)."""
    cols = {}
    for name, paths in CANDIDATES.items():
        cols[name] = pd.read_csv(paths[which]).set_index(ID_COL)[TARGET].reindex(ids).to_numpy()
    return pd.DataFrame(cols)


# --- Méthodes de combinaison : fit(P, y) renvoie une fonction P -> score ---

def fit_best_single(P, y):
    best = max(P.columns, key=lambda c: roc_auc_score(y, P[c]))
    return lambda Q: Q[best].to_numpy(), {best: 1.0}


def fit_mean(P, y):
    return lambda Q: Q.mean(axis=1).to_numpy(), {c: 1 / P.shape[1] for c in P.columns}


def to_ranks(Q):
    return Q.apply(lambda col: rankdata(col) / len(col))


def fit_rank_mean(P, y):
    return lambda Q: to_ranks(Q).mean(axis=1).to_numpy(), {c: 1 / P.shape[1] for c in P.columns}


def fit_ges(P, y, steps=GES_STEPS):
    """Greedy Ensemble Selection (Caruana et al., 2004) : à chaque étape on ajoute, avec remise, le modèle
    qui maximise l'AUC de la moyenne courante. Poids = fréquence de sélection (positifs, somme 1)."""
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
    """Stacking : régression logistique (légère pénalité L2) sur les logits des probabilités."""
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
    print("\n  AUC hors pli de chaque modèle :")
    for c in P.columns:
        print(f"    {c:<10} {roc_auc_score(y, P[c]):.4f}")
    print("\n  Corrélation des rangs des prédictions (Spearman) :")
    print(pd.DataFrame(spearmanr(P)[0], index=P.columns, columns=P.columns).round(3).to_string())
    print("\n  Corrélation des résidus y - p (des modèles qui se trompent pareil n'apportent rien à l'ensemble) :")
    print((P.rsub(y, axis=0)).corr().round(3).to_string())


def meta_oof(P, y, fit, seed=RANDOM_STATE):
    """Score combiné hors pli au niveau méta : pour chaque pli, combinaison apprise sur les 4 autres."""
    score = np.empty(len(y))
    for tr, va in make_cv(random_state=seed).split(P, y):
        combine, _ = fit(P.iloc[tr], y[tr])
        score[va] = combine(P.iloc[va])
    return score


def compare(P, y):
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
    """Ramène le score combiné dans [0, 1] : sigmoïde pour le stacking logistique, rangs pour la moyenne
    des rangs ; les autres méthodes sont déjà des moyennes de probabilités. L'ordre, donc l'AUC, ne change pas."""
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
            # ex. : submit ges moyenne logistique (mot-clé contenu dans le nom de la méthode)
            for key in sys.argv[2:]:
                method = next(m for m in SUBMISSION_NAMES if key.lower() in m.lower())
                write_submission(method, P, y, test_clean)
