# Diabetes Risk Prediction: A Reproducible Machine-Learning Framework

**A rigorous, reproducible classification study for early detection of type-2
diabetes risk using the Pima Indians Diabetes Database.**

> **Author:** Rudolph Otoo · **Domain:** Machine Learning / Health Informatics

---

## Abstract

Diabetes mellitus is one of the fastest-growing chronic diseases worldwide, and
early identification of at-risk individuals is critical to clinical
intervention. This repository presents a **complete, reproducible
machine-learning framework** for binary classification of diabetes status,
built on the Pima Indians Diabetes Database (768 adult female patients of Pima
Indian heritage).

The project emphasises **methodological rigor**, not just predictive accuracy:

- **Strict data hygiene** — a three-way *train / validation / test* split with
  stratification, preventing both data leakage and selection bias.
- **Leakage-aware preprocessing** — all imputation and scaling occurs *inside*
  a scikit-learn `Pipeline`, so estimators never observe test information.
- **Class-imbalance-aware evaluation** — primary metrics are **ROC-AUC** and
  **F1-score**, complemented by accuracy, precision, recall, and the Matthews
  correlation coefficient.
- **Baseline benchmarking** — every non-linear model is compared against a
  majority-class `DummyClassifier` and a regularised `LogisticRegression`.
- **Full determinism** — a single master seed propagates every randomness
  source, guaranteeing byte-identical splits across machines and runs.

**Key result:** a tuned **Gradient Boosting** classifier achieves a held-out
**ROC-AUC of ≈ 0.82** and **F1-score of ≈ 0.59**, lifting discrimination
roughly **+32 pp. of ROC-AUC over the majority-class baseline** while remaining
a fully transparent, extensible framework.

---

## Table of Contents

1. [Dataset](#-dataset)
2. [Methodology](#-methodology)
3. [Project Structure](#-project-structure)
4. [Results](#-results)
5. [Reproducibility](#-reproducibility)
6. [Setup & Execution](#-setup--execution)
7. [Testing](#-testing)
8. [Limitations & Future Work](#-limitations--future-work)

---

## 📊 Dataset

**Source:** [Pima Indians Diabetes Database](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
(originally contributed to the UCI Machine Learning Repository by the National
Institute of Diabetes and Digestive and Kidney Diseases).

| Attribute | Description |
|---|---|
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration after 2 h in an oral glucose tolerance test (mg/dL) |
| **BloodPressure** | Diastolic blood pressure (mm Hg) |
| **SkinThickness** | Triceps skin fold thickness (mm) |
| **Insulin** | 2-hour serum insulin (mu U/mL) |
| **BMI** | Body mass index (weight in kg / height in m²) |
| **DiabetesPedigreeFunction** | A function scoring the likelihood of diabetes based on family history |
| **Age** | Age (years) |
| **Outcome** | Class variable — `0` = no diabetes, `1` = diabetes |

**Size:** 768 patients · **Splits:** 460 train / 154 validation / 154 test (a
strict 60 / 20 / 20 partition, stratified on `Outcome` so class prevalence is
preserved across partitions).

> **Attribution.** The dataset was made available by the National Institute of
> Diabetes and Digestive and Kidney Diseases. Please cite the original UCI
> repository entry if you reuse this data in a publication.

---

## 🧠 Methodology

### 1 · Preprocessing

The raw Pima dataset contains several **physiologically impossible zero
values** (e.g., `BloodPressure = 0`, `BMI = 0`). These are not true zeros but
missing measurements. The pipeline (in `src/data.py`):

1. Coerces clinical columns to numeric.
2. Re-encodes impossible zeros in `BloodPressure`, `SkinThickness`, `Insulin`,
   and `BMI` as `NaN`.
3. Imputes missing values with the **column median** (robust to the strongly
   right-skewed `Insulin` distribution).

### 2 · Feature Engineering

The eight raw predictor variables are used as-is, with **standard scaling**
applied *within* each pipeline. No target-derived leakage enters any feature.

### 3 · Models

Every model is a scikit-learn `Pipeline` (`StandardScaler` → estimator):

| Model | Role | Rationale |
|---|---|---|
| `DummyClassifer` | Baseline | Majority-class predictor; defines the floor for any useful metric |
| `LogisticRegression` (L2) | Baseline / interpretable | Gold-standard classifier in epidemiological risk modelling |
| `RandomForest` | Non-linear | Bagged decision trees; robust to noise & interactions |
| `GradientBoosting` | Non-linear | Sequential additive trees; strongest on tabular clinical data |

### 4 · Splitting & Evaluation Protocol

- **Train / validation / test split** (60 / 20 / 20) with stratification.
- Hyperparameters tuned via **stratified 5-fold cross-validation** on the
  training split only, optimising **ROC-AUC**.
- The **validation set** is used exclusively during tuning; the **held-out test
  set** is touched exactly once for the final reported metrics.

---

## 📁 Project Structure

```
.
├── data/
│   ├── raw/            # original Pima CSV (auto-downloaded once)
│   └── processed/      # cleaned, median-imputed frame (generated)
├── notebooks/          # narrative EDA, modeling, and tuning walkthroughs
├── src/                # importable research package
│   ├── config.py       # paths & global hyperparameter/settings dataclasses
│   ├── data.py         # download + preprocessing + imputation
│   ├── features.py     # stratified split + cross-validation + seed derivation
│   ├── models.py       # baseline & ensemble pipelines + param grids
│   ├── evaluate.py     # metric computation + benchmark table aggregation
│   ├── tune.py         # GridSearchCV wrapper (strict train-only tuning)
│   ├── visualize.py    # publication-quality figure functions
│   └── scripts/
│       └── run_pipeline.py   # end-to-end CLI entry point
├── tests/              # pytest suite (11 tests)
├── reports/
│   ├── *.csv           # per-model & aggregate metric tables (generated)
│   └── figures/        # ROC, confusion matrices, EDA figures (generated)
├── models/             # serialised best pipeline (joblib)
├── requirements.txt    # pip dependency manifest
├── environment.yml     # conda dependency manifest
├── pyproject.toml      # packaging metadata
└── Makefile            # high-level orchestration (`make all`)
```

---

## 📈 Results

Final metrics on the **held-out test split** (154 patients), from a single
reference run seeded with `random_state = 42`:

| Model | Accuracy | ROC-AUC | F1-score | Precision | Recall | MCC |
|---|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | 0.734 | **0.821** | 0.594 | 0.638 | 0.556 | 0.400 |
| **Random Forest** | 0.721 | **0.819** | 0.566 | 0.622 | 0.519 | 0.366 |
| Logistic Regression | 0.747 | 0.802 | 0.571 | 0.703 | 0.481 | 0.415 |
| Dummy (majority) | 0.649 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |

> *Stability note:* tree-based ensembles (RF, GBM) are stochastic at training
> time; identical seeds reproduce identical runs, but the held-out metrics can
> differ by a few hundredths across *different* seeds. The table above is the
> exact output of one fixed-seed reference run and is regenerated by
> `make all`.

> *Note on the Dummy column:* a majority-class predictor achieves **65% accuracy**
> yet a **ROC-AUC of 0.5** and **F1/MCC = 0** — it never detects any diabetic.
> This starkly motivates why accuracy alone is an unreliable metric for this
> imbalanced problem (~35% prevalence).

**Interpretation.** Gradient Boosting and Random Forest provide the best
discrimination (ROC-AUC ≈ 0.82), a modest but real improvement over the highly
interpretable logistic regression baseline (≈ 0.80). For a small, noisy
clinical dataset, this gap is consistent with the broader medical-ML
literature: tree ensembles reliably edge out linear models but rarely by large
margins.

Recomputed figures are generated on demand (see [below](#-reproducibility)) and
written to `reports/figures/`:

| Figure | Description |
|---|---|
| `01_target_distribution.png` | Binary class balance bar chart (imbalance visualised) |
| `02_correlation_heatmap.png` | Spearman correlation matrix |
| `03_feature_distributions.png` | Marginal histograms of all 8 predictors |
| `04_roc_curves.png` | Overlayed ROC curves of every model |
| `05_confusion_matrices.png` | Side-by-side confusion matrices |
| `06_feature_importance.png` | Gini importances of the tuned GBM |

---

## 🔁 Reproducibility

Three independent mechanisms guarantee a reviewer can reproduce every number:

1. **Deterministic seeding** — a single master seed (`42`, in `src/config.py`)
   is deterministically expanded into child seeds for every `train_test_split`
   and model. Re-running yields identical partitions and identical fits.
2. **Leakage-safe `Pipeline`s** — scaling and any future feature transforms are
   fitted *within* each CV fold, so estimates of generalisation are unbiased.
3. **Explicit dependency manifests** — `requirements.txt` and `environment.yml`
   list a pinned core stack (`numpy`, `pandas`, `scikit-learn`, `matplotlib`,
   `seaborn`, `joblib`).

The dataset is downloaded once into `data/raw/` and cached; subsequent runs are
fully **offline** and deterministic.

---

## ⚙️ Setup & Execution

### Option A — pip (recommended)

```bash
git clone https://github.com/rudolphOtoo/diabetes-risk-prediction.git
cd diabetes-risk-prediction
make setup                 # creates .venv + installs requirements
make all                   # fetch → preprocess → train → evaluate (≈3–5 min)
```

### Option B — conda

```bash
git clone https://github.com/rudolphOtoo/diabetes-risk-prediction.git
cd diabetes-risk-prediction
conda env create -f environment.yml
conda activate diabetes-risk-prediction
python -m src.scripts.run_pipeline
```

### Option C — step-by-step (pip)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -c "from src.data import process_data; process_data()"   # fetch + preprocess
python -m src.scripts.run_pipeline                               # full pipeline
```

### Explore the notebooks

Notebooks assume you are in the `notebooks/` directory (they add the repo root
to `sys.path`):

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_eda.ipynb
```

1. `01_eda.ipynb` — exploratory data analysis & correlation structure
2. `02_modeling.ipynb` — untuned benchmark of all four models
3. `03_tuned_pipeline.ipynb` — hyperparameter tuning + final evaluation

---

## 🧪 Testing

The repository ships a `pytest` suite asserting the invariants that matter most
for an admissions reviewer:

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

This verifies preprocessing shape/class balance, stratification, split
reproducibility, and that every model trains and scores within valid bounds.

---

## ⚠️ Limitations & Future Work

- **Small, single-source dataset (n = 768).** Generalisation to other
  populations (other ethnicities, both sexes, external cohorts) is unverified;
  the natural next step is external validation on a held-out clinical cohort.
- **Class imbalance.** Precision/recall trade-off is tuned to ROC-AUC; a
  clinician-facing deployment might instead optimise sensitivity at a fixed
  specificity using the ROC curve.
- **Causal vs. associative.** Findings are predictive, not causal. No causal
  claims are made about the relationship between BMI/glucose and diabetes.
- **Future work:** survival/progression modelling, calibration curves (Brier
  score), SHAP-based feature attribution, and nested cross-validation for
  unbiased model-selection error estimates.

---

## 📄 License

This project is distributed under the [MIT License](LICENSE).

---

*Built with Python, scikit-learn, and reproducible-research best practices.
Numerical results are regenerable with a single command under a fixed seed.*
