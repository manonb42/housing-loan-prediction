# Housing Loan Prediction

Predict the probability that a home purchase loan application is accepted (`accepted`), from the HMDA data in
`data/`. The metric is the AUC ROC.

## Requirements

- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Python: uv installs the right version automatically, based on `.python-version`

To install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

```bash
git clone https://github.com/manonb42/housing-loan-prediction.git
cd housing-loan-prediction
uv sync
```

`uv sync` creates a `.venv` virtual environment and installs the exact dependency versions listed in `uv.lock`.

The data is not in the repository. Before running the project, put the challenge's `train.csv`, `test.csv` and
`sample_submission.csv` in a `data/` folder at the root of the project, or point to the folder where they are by
changing this line of `src/housing_loan_prediction/data.py`:

```python
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
```

For example `DATA_DIR = Path("/path/to/the/data")`.

## Usage

The final model is a greedy ensemble selection (GES) of three models: CatBoost (weight 0.56),
HistGradientBoosting (0.38) and a random forest (0.06). The weights are learned on the out-of-fold predictions of
five candidate models, on the same 5 folds. Its cross-validated AUC is 0.9171. The submission to hand in,
`submissions/ensemble_ges.csv`, is written by these commands, run in this order:

```bash
# 1. Single models trained on the whole train set (test predictions in submissions/)
uv run python -m housing_loan_prediction.submission                 # lasso
uv run python -m housing_loan_prediction.trees submit               # random forest
uv run python -m housing_loan_prediction.gradient_boosting submit   # HistGradientBoosting

# 2. CatBoost: learning curve on the 5 folds (out-of-fold predictions), then submissions/catboost.csv
uv run python -m housing_loan_prediction.boosting_libs curves catboost
uv run python -m housing_loan_prediction.boosting_libs submit catboost

# 3. Out-of-fold predictions of the lasso, random forest, linear SVM and HistGradientBoosting
uv run python -m housing_loan_prediction.stacking oof

# 4. GES weights learned on the out-of-fold predictions, applied to the test predictions
uv run python -m housing_loan_prediction.stacking submit ges
```

The second retained submission, CatBoost alone with its default settings (cross-validated AUC 0.9161), is
`submissions/catboost.csv`, written at step 2. The whole run takes about one hour, most of it for CatBoost.

## Code

All the code is in `src/housing_loan_prediction/`. Out-of-fold predictions go to `oof/` (not in the
repository), submissions to `submissions/`.

| File | Role |
|---|---|
| `preprocessing.py`, `data.py` | Cleaning, derived features, preprocessing pipelines for linear and tree models, reading the data (exploration in `notebooks/01_eda_preprocessing.ipynb`) |
| `evaluation.py` | Stratified 5-fold cross-validation (seed 42) and AUC |
| `logistic.py`, `submission.py` | Logistic regression without penalty, ridge and lasso; lasso submission |
| `svm.py` | Linear and RBF SVM |
| `trees.py` | Pruned decision tree and random forest, random forest submission |
| `gradient_boosting.py` | HistGradientBoosting: learning curves, grids on tree size and learning rate, submission |
| `boosting_libs.py` | CatBoost and LightGBM: learning curves, CatBoost submission |
| `stacking.py` | Out-of-fold predictions and combinations of the models (mean, rank mean, GES, logistic stacking), submissions |
