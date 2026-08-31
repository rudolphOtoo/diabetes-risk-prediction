# Paradigm Justification: Model Selection Philosophy for Diabetes Risk Prediction

**Author:** Rudolph Otoo

---

## Abstract

A recurring methodological question in computational disease modelling is
whether diabetes risk prediction is best framed as a *statistical machine
learning* (ML) problem, a *mechanistic dynamical system* (ODE) problem, or a
*hybrid* of both. This document provides a formal paradigm assessment justifying
the dual-pillar architecture adopted in this repository. We demonstrate that the
choice is not arbitrary but is dictated by the **scale of inference**,
**epistemic status of available knowledge**, and the **modelling objective**:
individual-level risk stratification on cross-sectional observational data is a
statistical classification task, while population-level disease prevalence
dynamics under demographic flux is a deterministic compartmental modelling task.
The two pillars address fundamentally different scientific questions with
complementary mathematical machinery.

---

## 1. Problem Taxonomy & Scale Analysis

### 1.1 Individual Risk Stratification as a Statistical/Phenomenological Problem

The core clinical question — *given an asymptomatic patient's glucose, BMI, age,
and family history, what is the probability that they will develop type-2
diabetes within five years?* — is a **cross-sectional binary classification
task** on observational data. We argue this is fundamentally a statistical
(phenomenological) problem for three interlocking reasons:

**Absence of closed-form first principles.** The mapping from the eight Pima
features to diabetes onset is mediated by complex, partially understood
pathophysiology involving insulin resistance, beta-cell exhaustion, adipokine
signalling, and genetic penetrance. No system of ordinary differential
equations is currently known that:

1. takes the Pima feature vector $(x_1, \dots, x_8) \in \mathbb{R}^8$ as an
   initial condition,
2. advances it through a mechanistic state-space, and
3. outputs the probability of a binary clinical outcome at a fixed horizon.

The underlying biology is **high-dimensional, stochastic, and个体-level
heterogeneous**: identical BMI and glucose profiles can yield divergent outcomes
depending on unobserved genetic, epigenetic, and lifestyle covariates. In the
absence of known governing equations at the individual level, the appropriate
modelling stance is **phenomenological** — learn a discriminative function
$f: \mathbb{R}^8 \to \{0, 1\}$ directly from labelled examples.

**Data structure dictates method.** The Pima dataset is a *cross-sectional
snapshot*: 768 independent patients, each measured once, with no temporal
trajectory. There is no time axis to differentiate along, no dynamical state to
propagate, and no coupling between individuals. The data-generating process is
independently and identically distributed (i.i.d.) conditional on covariates —
precisely the assumption that underpins supervised statistical learning.
Classifiers such as logistic regression, random forests, and gradient boosting
are designed for exactly this structure: learn a decision boundary in a
fixed-dimensional feature space from exchangeable labelled observations.

**Interaction complexity without mechanistic closure.** The eight Pima predictors
interact through nonlinear, partially synergistic pathways (e.g., age modulates
the BMI–glucose relationship; pedigree function captures unmeasured genetic
load). These interactions are real but not expressible as a tractable ODE system
at the individual scale. A gradient-boosted tree ensemble approximates the
induced decision surface $f^*(x)$ through additive function composition, which
is the mathematically correct response when the true function is complex and the
governing equations are unknown. Statistical ML does not require *understanding*
the mechanism; it requires only that the training distribution is
representative of the deployment distribution — a *correlational*, not
*causal*, requirement.

In summary, individual risk stratification is a problem of **statistical
discrimination in a fixed-dimensional feature space from i.i.d. cross-sectional
observations**, for which data-driven classifiers are the natural and principled
tool.

### 1.2 Dynamic / Population Modelling: Where Differential Equations Are Appropriate

Ordinary differential equations become the mathematically appropriate framework
when the modelling objective shifts from *classifying an individual* to
*describing the temporal evolution of a population-level quantity*. Two distinct
scales justify ODE-based modelling in the diabetes context:

#### Physiological Scale: Metabolic ODE Systems

At the organ/organism level, the glucose–insulin regulatory system is
well-described by coupled nonlinear ODEs with a long theoretical pedigree. The
canonical example is **Bergman's Minimal Model** of glucose-insulin dynamics:

$$
\begin{aligned}
\frac{dG}{dt} &= -p_1\, G - p_2\, X + P(t), \\[2mm]
\frac{dX}{dt} &= -p_3\, X + p_4\, I(t), \\[2mm]
\frac{dI}{dt} &= -n\,(I - I_b) + \gamma\, G(t)\, [G(t) - h]\, \mathbb{1}_{G > h},
\end{aligned}
$$

where $G(t)$ is plasma glucose, $I(t)$ is insulin concentration, $X(t)$ is a
remote insulin-effect compartment, $P(t)$ is the exogenous glucose input
(meal or IVGTT), and $p_1, \dots, p_4, n, \gamma, h$ are identifiable
physiological rate parameters. This system is a genuine *mechanistic* model: its
parameters have direct biophysical interpretations (glucose effectiveness,
insulin sensitivity, beta-cell responsiveness), its equations follow from mass
balance and receptor kinetics, and it has been validated against clinical IVGTT
data for over four decades (Bergman et al., 1979; Cobelli et al., 2007).

Such metabolic ODEs answer questions like *how does a perturbation in insulin
sensitivity propagate through glucose homeostasis?* — questions about **intra-
individual dynamics**, not inter-individual classification. They are outside the
scope of this repository's dataset (which provides no temporal glucose traces)
but represent the canonical domain where mechanistic ODE modelling of diabetes
is both mathematically rigorous and clinically validated.

#### Epidemiological Scale: Compartmental Population Transitions

At the population level, the spread and progression of a disease through a
closed cohort is naturally modelled by **compartmental ODE systems** — the same
mathematical framework that underpins classical epidemic theory (Kermack &
McKendrick, 1927; Anderson & May, 1991). While type-2 diabetes is non-infectious,
the compartmental abstraction generalises to *chronic disease staging*: a
population can be partitioned into

$$
\mathcal{C} = \{\text{Susceptible},\ \text{Pre-diabetic},\ \text{Diabetic},\ \text{Complications},\ \text{Exited}\},
$$

with transitions governed by rates $(\beta_{ij})$ that depend on demographic
fluxes (birth, migration, mortality), lifestyle risk-factor prevalence, and
intervention coverage. The governing system takes the generic form

$$
\frac{d\mathbf{x}}{dt} = \mathbf{A}(\theta)\,\mathbf{x}(t),
$$

where $\mathbf{x}(t) \in \mathbb{R}_+^{|\mathcal{C}|}$ is the compartment
vector and $\mathbf{A}(\theta)$ is a rate matrix parameterised by epidemiological
rates $\theta$. This is the framework implemented in this repository's SIR and
SEIRD layers (`src/models/ode/`), which model deterministic disease-prevalence
trajectories in a closed population of size $N$.

The key mathematical properties that make ODEs the correct tool at this scale
are:

1. **Continuous-time dynamics.** Disease staging is a continuous process with
   constant or time-varying rates, not a discrete-time Markov chain.
2. **Mass-action coupling.** The rate of transition from one compartment depends
   on the *current size* of the source compartment — a fundamentally dynamical
   interaction.
3. **Conservation structure.** The closed-population invariant
   $\sum_c x_c(t) = N$ is a *structural* property of the ODE system, not an
   empirical regularity — it holds by construction for all parameter values and
   all time.
4. **Analytical tractability.** The basic reproduction number
   $\mathcal{R}_0 = \beta / \gamma$, the endemic equilibrium, and the epidemic
   threshold are all derivable from the ODE system's Jacobian — properties that
   have no analogue in a statistical classifier.

In summary, ODEs are appropriate when the modelling objective is to describe
**how a population-level quantity evolves deterministically through time** under
known (or estimable) rate constants, with conservation laws and equilibrium
structure that follow from the equations themselves.

---

## 2. Methodological Trade-Off Matrix

The following table provides a systematic comparison across the key dimensions
that govern model selection in computational biology.

| Dimension | Statistical ML | Mechanistic ODEs |
|---|---|---|
| **Data requirements** | Large $n$ of labelled cross-sectional observations; no temporal structure needed. Performance degrades gracefully with missing covariates (via imputation pipelines). | Requires time-series data (longitudinal compartment counts or metabolic traces). Parameter recovery demands sufficient temporal resolution and multi-channel observations. Low-$n$ regimes are viable when parameters are identifiable from first principles. |
| **Interpretability** | *Post-hoc* interpretability via feature importance, SHAP values, or logistic-regression coefficients. Model explains *which* features drive predictions, not *why* the biology works that way. | *Intrinsic* interpretability: every parameter is a rate constant with a direct biophysical meaning. The model explains *why* dynamics unfold as they do, encoding causal structure by construction. |
| **Generalizability outside training distribution** | Fundamentally limited to the support of the training distribution. Extrapolation beyond the observed covariate range is unreliable. Transportability requires re-calibration on the target population. | Can extrapolate to unobserved regimes (novel parameter values, different population sizes, intervention scenarios) provided the mechanistic assumptions (rate structure, conservation) remain valid. The model's validity is determined by its assumptions, not by training data coverage. |
| **Identifiability** | Parameters (weights, splits) are uniquely determined by the optimisation landscape; no identifiability concerns per se (though overfitting requires regularisation). | Structural and practical identifiability must be verified. Some parameters (e.g., the latent rate $\sigma$ in SEIRD) are only identified when multiple compartments are observed jointly. Identifiability analysis is a prerequisite, not an afterthought. |
| **Clinical decision support** | Directly answers the clinical question *"what is this patient's risk?"* with a probability score. Naturally integrates into screening workflows and electronic health records. | Answers policy and planning questions *"how will prevalence change under intervention X?"* or *"what is the expected peak caseload?"*. Informs resource allocation and public-health strategy, not individual patient management. |
| **Uncertainty quantification** | Achieved through ensemble variance, bootstrap, or Bayesian wrappers — but is *aleatoric* (data noise), not *epistemic* (model structure). | Achieved through sensitivity analysis, Bayesian inference on rate parameters, or ensemble Kalman filtering — captures *epistemic* uncertainty about the dynamics themselves. |
| **Computational cost** | Low per prediction (milliseconds); high for training (minutes–hours with hyperparameter search). | Low per simulation (milliseconds for ODE solve); high for Bayesian parameter inference (hours for MCMC). |

The critical insight is that **these are not competing methods for the same
task** — they are *complementary methods for different tasks* that happen to
share a disease domain. The ML pipeline answers *"who is at risk?"*; the ODE
pipeline answers *"how does risk evolve through a population over time?"*. A
framework that implements both is not indecisive — it is *multi-scale*.

---

## 3. Evaluation of Current Repository Architecture

This repository implements a **dual-pillar architecture** with clean boundary
definitions between the statistical and mechanistic components. A scientific
reviewer should understand the following structural commitments:

### 3.1 The ML Pipeline: Individual-Level Risk Classification

| Aspect | Design |
|---|---|
| **Scope** | Binary classification of diabetes status from 8 cross-sectional clinical features |
| **Data** | Pima Indians Diabetes Database ($n = 768$, i.i.d. observations) |
| **Models** | 4 leakage-safe `sklearn` Pipelines: DummyClassifier, LogisticRegression (L2), RandomForest, GradientBoosting |
| **Preprocessing** | Median imputation + standardisation, encapsulated inside each Pipeline (zero train/test leakage) |
| **Validation** | Stratified 60/20/20 train/val/test split; 5-fold stratified CV on training fold only; test set touched exactly once |
| **Evaluation** | ROC-AUC (threshold-invariant), $F_1$, MCC (imbalance-robust); accuracy reported for contrast only |
| **Determinism** | Single master seed ($s = 42$) propagated via CRC32-derived child seeds; byte-identical across runs |
| **Entry point** | `python -m src.scripts.run_pipeline` / `make all` |
| **Output** | Per-model metric CSVs, 6 publication figures, serialised best pipeline (joblib) |

The ML pipeline does **not** model disease dynamics, time evolution, or
population-level transitions. It solves a single, well-defined statistical
decision problem: *given a feature vector, assign a risk score.*

### 3.2 The ODE Pipeline: Population-Level Compartment Dynamics

| Aspect | Design |
|---|---|
| **Scope** | Deterministic simulation of disease progression through a closed population via systems of ODEs |
| **Data** | Simulated trajectories; no observational data required (parameters from literature) |
| **Models** | SIR ($S$, $I$, $R$) and SEIRD ($S$, $E$, $I$, $R$, $D$) compartmental systems with explicit RHS |
| **Parameters** | Cited parameter table with units, bounds, and peer-reviewed sources; validated at construction |
| **Invariants** | Conservation ($\sum_c x_c = N$) and non-negativity verified programmatically on every trajectory |
| **Solver** | `scipy.integrate.solve_ivp` (LSODA), config-driven tolerances ($\text{rtol} = 10^{-8}$, $\text{atol} = 10^{-9}$) |
| **Identifiability** | Synthetic-data parameter recovery via constrained least squares; documented identifiability analysis |
| **Entry point** | `python -m src.scripts.run_ode_model --model {sir,seird}` / `make run-model` |
| **Output** | Trajectory CSVs (`results/analysis/`), fitted parameter tables |

The ODE pipeline does **not** classify individual patients, consume the Pima
dataset, or produce risk scores. It solves a single, well-defined dynamical
systems problem: *given a set of rate constants and initial conditions, simulate
the temporal evolution of compartment populations.*

### 3.3 Boundary Definitions

The two pillars share **no data flow, no model coupling, and no shared
evaluation metrics**. They are connected only by:

1. **Conceptual complementarity** — both address aspects of diabetes at
   different scales.
2. **Shared infrastructure** — common configuration (`src/config.py`), seed
   management, and CI/CD pipeline.
3. **Shared documentation** — unified repository with cross-referenced
   mathematical formulations.

This separation is **by design, not by accident**. A reviewer should understand
that implementing both paradigms in one repository does not conflate them; it
demonstrates the ability to match mathematical tools to scientific questions at
the appropriate scale.

---

## 4. Model Selection Rationale

The following two-paragraph statement is drafted for insertion into
`README.md` and `docs/manuscript.md` to explicitly defend the dual approach to
a scientific reviewer or PhD admissions committee.

> **Model Selection Rationale.** The dual-pillar architecture of this repository
> is not an exercise in methodological hedging but a deliberate response to the
> fact that diabetes risk prediction encompasses two fundamentally different
> scientific questions operating at different scales. At the *individual* scale,
> the task is cross-sectional binary classification: assign a probability of
> disease onset to an asymptomatic patient given a fixed vector of clinical
> biomarkers. This is a statistical discrimination problem for which no
> first-principles differential equation is known, the data are i.i.d. and
> cross-sectional, and the appropriate mathematical tools are supervised
> classifiers — logistic regression for interpretability, tree ensembles for
> non-linear interaction capture. At the *population* scale, the task is to
> describe how disease prevalence evolves through a cohort over time under
> demographic and intervention fluxes. This is a dynamical systems problem
> governed by coupled ODEs with conservation structure, equilibrium thresholds,
> and identifiable rate parameters — the same mathematical scaffolding that
> underpins classical epidemiological theory (Anderson & May, 1991; Kermack &
> McKendrick, 1927). Conflating these two scales under a single modelling
> paradigm would sacrifice the strengths of each: a classifier cannot predict
> epidemic trajectories, and an ODE system cannot stratify individual risk.
>
> This deliberate multi-scale design reflects a core principle of applied
> mathematical modelling: **the choice of mathematical framework should be
> dictated by the structure of the question, not by the availability of a
> single convenient method**. A statistical learner is not a poor substitute for
> a mechanistic model, nor vice versa — each is the *correct* tool for its
> respective scale of inference. By implementing both pillars with rigorous
> validation (leakage-safe pipelines, conservation-law verification,
> identifiability analysis, and reproducible seeding), this framework
> demonstrates that model selection in computational biology requires
> understanding *what question is being answered* before selecting *which
> mathematics to apply*. This is the intellectual discipline that distinguishes
> principled computational modelling from ad hoc method application, and it is
> the modelling philosophy that this repository is designed to embody.

---

## References

1. Anderson, R. M. & May, R. M. (1991). *Infectious Diseases of Humans:
   Dynamics and Control*. Oxford University Press.
2. Bergman, R. N., Ider, Y. Z., Bowden, C. R., & Cobelli, C. (1979).
   Quantitative estimation of insulin sensitivity. *American Journal of
   Physiology*, 236(6), E667–E677.
3. Cobelli, C., Dalla Man, C., Sparacino, G., Magni, L., De Nicolao, G., &
   Kovatchev, B. P. (2007). Diabetes: models of the glucose-insulin system.
   *IEEE Engineering in Medicine and Biology Magazine*, 26(6), 25–33.
4. Kermack, W. O. & McKendrick, A. G. (1927). A contribution to the
   mathematical theory of epidemics. *Proc. R. Soc. A*, 115, 700–721.
5. Bergman, R. N., Phillips, L. S., & Cobelli, C. (1985). Physiologic
   evaluation of factors controlling glucose tolerance in man. *Journal of
   Clinical Investigation*, 76(6), 1643–1654.
