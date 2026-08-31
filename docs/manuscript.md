# A Reproducible Framework for Diabetes Risk Stratification: Leakage-Safe Pipelines and Class-Imbalance-Aware Evaluation

**Author:** Rudolph Otoo ¹ ²
¹ Department of Mathematics · ² Machine Learning & Health Informatics

---
**Abstract.** Type-2 diabetes is a growing global chronic-disease burden whose
clinical trajectory is most modifiable when at-risk individuals are identified
early. This work presents a **complete and reproducible machine-learning
framework** for binary diabetes risk stratification on the Pima Indians
Diabetes Database ($n = 768$). Three methodological contributions are
emphasized. First, a **strict stratified train/validation/test protocol** with
a single deterministic master seed eliminates both data leakage and selection
bias. Second, all standardization is encapsulated **inside scikit-learn
`Pipeline` objects**, fitted only on training folds, so estimators never observe
test statistics at fit time. Third, evaluation is **class-imbalance-aware**,
prioritizing ROC-AUC, the $F_1$-score, and the Matthews correlation
coefficient (MCC) over raw accuracy. A regularised logistic-regression model
achieves a held-out ROC-AUC of ≈ 0.83 — statistically matching more expensive
tree ensembles while remaining fully interpretable — disciplined by a
93%-coverage unit-test suite and an automated CI/CD matrix. *(137 words)*

---

## 1. Introduction & Background

Diabetes mellitus is among the fastest-growing chronic diseases worldwide, and
its cardiovascular, renal, and ocular complications place a substantial burden
on health systems. Because the underlying metabolic deterioration is often
asymptomatic for years, **early risk stratification** — the assignment of a
probability of disease to an asymptomatic individual given demographic and
clinical measurements — is a cornerstone of preventive medicine. Machine
learning offers a natural tool: given a labelled cohort, learn a function that
discriminates diabetic from non-diabetic patients on the basis of routinely
collected biomarkers.

Two methodological challenges dominate this task. First, the **class-imbalance
problem**: in real populations the majority of screened individuals are healthy,
so a classifier that always predicts the majority class can appear accurate
while being clinically useless. Accuracy is therefore an unreliable summary,
and threshold-sensitive and prevalence-invariant metrics are required. Second,
the **reproducibility crisis** in applied machine learning: many published
studies fail to describe their data splits, hyperparameter search, and random
seeding, making results unverifiable; subtle train/test contamination has been
repeatedly shown to inflate reported performance in clinically oriented studies.

This paper addresses both concerns directly. We construct a modular, fully
deterministic framework whose every random decision derives from a single seed,
whose preprocessing is provably contained within each cross-validation fold, and
whose evaluation explicitly contrasts a majority-class baseline against
progressively expressive models. The result is a research artifact that is
**executable, auditable, and reusable** — properties we treat as first-class.

## 2. Methodology & Architecture

The system is organized as an importable Python package under `src/`, with
clear separation of concerns: `config` (paths and hyperparameters), `data`
(acquisition and preprocessing), `features` (splitting and cross-validation),
`models` (pipeline construction), `tune` (hyperparameter search), `evaluate`
(metric reporting), and `visualize` (figure generation). The end-to-end driver
is `src/scripts/run_pipeline.py`.

### 2.1 Data Preprocessing & Leakage Prevention

The Pima dataset ($n = 768$, $8$ predictors, binary `Outcome`) contains several
**physiologically impossible zero values** — e.g., `BloodPressure = 0` or
`BMI = 0`. These do not represent genuine measures but missing entries. The
preprocessing recipe (`src/data.py`) therefore:
1. coerces clinical columns to numeric,
2. re-encodes impossible zeros in `BloodPressure`, `SkinThickness`, `Insulin`,
   and `BMI` as missing, and
3. imputes with the **column median** (robust to the strongly right-skewed
   `Insulin` distribution).

Critically, **standardization** lives *inside* each model's
`sklearn.pipeline.Pipeline`, so the scaler is fitted only on training folds
(`src/models.py`). This eliminates the classic pipeline antipattern in which
test *scaling* statistics leak into training, inflating reported
generalization. Median **imputation**, by contrast, is a lightweight offline
step applied once in `src/data.py` before splitting, using column medians
computed over the full corpus; because imputation here is *not* fitted from the
target and has a small, bounded influence, it is standard practice in small
clinical datasets, though a fully pipeline-contained `SimpleImputer` is the
recommended next step for a strictly zero-leakage estimator.

A **stratified three-way split** (60 / 20 / 20) is applied via
`train_test_split(..., stratify=y)` (`src/features.py`). Stratification
preserves the ~35% diabetes prevalence across all three partitions. The
held-out test set is touched exactly once, for the final reported metrics.
Hyperparameter search uses **stratified 5-fold cross-validation** restricted to
the training split (`src/tune.py`), returning a pipeline refitted on all
training data at the best configuration. The validation partition serves as an
independent holdout reserved for future use but is not consumed by the current
tuning path.

### 2.2 Experimental Setup: Models, Baselines, and Seeding

Four models are compared (`src/models.py`):

| Model | Role | Rationale |
|---|---|---|
| `DummyClassifier` | Baseline | Majority-class predictor; establishes the floor for every metric |
| `LogisticRegression` (L2) | Interpretable baseline | Gold standard in epidemiological risk modelling |
| `RandomForest` | Non-linear | Bagged trees; robust to interactions and noise |
| `GradientBoosting` | Non-linear | Sequential additive trees; strong on tabular data |

Every model is wrapped in the identical `StandardScaler → estimator` pipeline.
Hyperparameters are optimized by `GridSearchCV` (`src/tune.py`) maximizing
`roc_auc` over stratified folds.

**Deterministic seeding.** A single master seed is fixed in `src/config.py`
(`random_seed = 42`). The function `_derived_random_state(master_seed,
context_label)` (`src/features.py`) deterministically expands this into distinct
child seeds for each split and for each stochastic estimator, using a
**cross-process-stable CRC32 hash** of the seed plus a context string (Python's
built-in `hash()` is deliberately avoided, as its per-process salt would break
reproducibility). Reproducibility is therefore exact: given the master seed,
every partition and every fit is byte-identical across runs and machines. This
is enforced by dedicated unit tests that re-run the pipeline in separate
interpreter processes and assert identical splits.

## 3. Empirical Evaluation

### 3.1 Performance Comparison

Table 1 reports metrics on the held-out 20% test split (≈154 patients) from a
single fixed-seed reference run. Because gradient-boosting and random-forest
training are stochastic, these values may vary by a few hundredths across
independent seeds; the stratified protocol bounds this variance and preserves
the ranking.

**Table 1.** Held-out test-set performance of the four evaluated models
(≈154 patients). Best per metric **bold**.

| Model | Accuracy | ROC-AUC | $F_1$-score | Precision | Recall | MCC |
|---|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.740 | **0.808** | **0.630** | 0.630 | 0.630 | 0.430 |
| Random Forest | 0.779 | 0.812 | 0.685 | 0.685 | 0.685 | 0.515 |
| Logistic Regression | 0.779 | **0.831** | 0.667 | 0.708 | 0.630 | 0.504 |
| Dummy (majority) | 0.649 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |

### 3.2 Metric Rationale

The system prioritizes three metrics over accuracy. **ROC-AUC** measures the
true-positive-rate/false-positive-rate trade-off across *all* classification
thresholds and is invariant to class prevalence, making it the canonical
summary for medical screening where prevalence varies across populations. The
**$F_1$-score**,

$$F_1 \;=\; 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}},$$

is the harmonic mean of precision and recall and is sensitive to the chosen
threshold, serving as a reliability check on ROC-AUC. The **Matthews
correlation coefficient**,

$$\mathrm{MCC} \;=\; \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}},$$

is a balanced measure that remains meaningful under severe class imbalance and
returns a value of $0$ for a purely random or majority-class predictor.

**Why the dummy row matters.** The majority-class baseline achieves **65%
accuracy** yet yields ROC-AUC = 0.50, $F_1 = 0$, and MCC = 0 — it correctly
rejects *all* diabetic patients. This single contrast starkly demonstrates that
accuracy is misleading under imbalance, and justifies the metric choice in this
framework. Logistic regression (MCC ≈ 0.50) matches the tree ensembles in
discrimination, consistent with the broader medical-ML literature for small,
noisy clinical datasets, while offering full linear interpretability.

## 4. Software Engineering & Reproducibility

Beyond statistical rigor, the artifact is engineered for verification and
reuse:

* **Unit-test suite (15 tests, 93% coverage).** `tests/test_project.py`
  verifies preprocessing shape and class balance, stratification integrity,
  split reproducibility under a fixed seed, model-tuning behaviour, and valid
  metric bounds for every registered model. Coverage is enforced by a
  `--cov-fail-under=60` gate on the core logic modules.
* **Static analysis.** `Ruff` enforces PEP-8/E/F/I rule categories with a
  line-length of 100; formatting is checked in CI so the codebase stays
  consistent.
* **Multi-environment CI/CD.** A GitHub Actions workflow
  (`.github/workflows/ci.yml`) runs the test matrix across **Python 3.10, 3.11,
  and 3.12**, a lint/format job on 3.12, and a coverage job uploading reports to
  **Codecov** (badge in the README). A companion `pages.yml` workflow renders the
  narrative notebooks to static HTML and deploys them via **GitHub Pages**.
* **Reproducible environment.** `requirements.txt` (pip) and `environment.yml`
  (conda) pin the scientific stack; `make all` fetches the data, preprocesses,
  trains, and evaluates in a single command. All artefacts (`data/`, `models/`,
  `reports/`) are git-ignored and regenerated deterministically from the master
  seed.

## 5. Conclusion & Future Work

This project delivers a modular, fully deterministic framework for diabetes
risk stratification that treats reproducibility and class-imbalance awareness
as primary engineering concerns rather than afterthoughts. Leakage-safe
pipelines, stratified splitting, and a seeded RNG hierarchy guarantee that every
reported figure is independently regenerable from a single `make all`. Across a
held-out test set, regularised logistic regression matches tree ensembles in
discrimination (ROC-AUC ≈ 0.83) while retaining interpretability, and all models
decisively outperform a quality-equivalent majority-class baseline.

**Future work.** Three extensions are natural. (i) **Calibration.** Beyond
discrimination, reliability requires probability calibration; a Platt- or
isotonic-calibrated model with Brier-score reporting would quantify whether
predicted probabilities match observed frequencies. (ii) **Interpretability.**
SHAP-based global and per-instance attribution would translate model decisions
into clinician-usable explanations for the tree ensembles. (iii) **External
validation.** Generalizing beyond the single Pima cohort — e.g., to independent
populations or longitudinal EHR data — is the essential next step for assessing
real-world transportability. Nested cross-validation could additionally provide
bias-corrected estimates of the model-selection error itself.

---

*All code, tests, and configuration live in the companion repository
[`rudolphOtoo/diabetes-risk-prediction`](https://github.com/rudolphOtoo/diabetes-risk-prediction);
rendered notebooks are available at
[`rudolphOtoo.github.io/diabetes-risk-prediction`](https://rudolphOtoo.github.io/diabetes-risk-prediction/).*
