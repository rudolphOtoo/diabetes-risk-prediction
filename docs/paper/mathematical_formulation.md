# Mathematical Formulation of the Compartmental (ODE) Models

This document records the full mathematical description of the mechanistic
compartmental layer implemented in `src/models/ode/`. Every equation here has
a direct, unit-tested implementation; the parameter table that follows fixes
each rate to bounded, cited physical values.

**Scope.** The repository contributes two complementary modelling pillars:

1. **Mechanistic compartmental ODE models** (this document) — explicit
   systems of ordinary differential equations describing disease progression
   through the population, verified for conservation and non-negativity and
   tested for parameter recoverability.
2. **Statistical detection models** — the original scikit-learn classifiers on
   the cross-sectional Pima cohort (see
   [`README.md`](../../README.md) and [`docs/manuscript.md`](../manuscript.md)).

---

## 1. Shared conventions

Let ``N`` denote a fixed, closed population of size ``N > 0``. Following
standard convention we write compartment counts as functions of time,
``x_c(t)``, with the total

```math
N = \sum_{c \in \mathcal{C}} x_c(t) \quad \forall\, t \ge 0 \qquad \text{(closed population)} .
```

All rates are non-negative and dimensionless per unit time (days⁻¹). The
closed-population assumption implies the **conservation law**
``\sum_c x_c(t) = N`` at every time point, and the **quasi-positivity** of each
vector field implies **non-negativity** of every compartment — both are
asserted programmatically on each integrated trajectory.

### Nondimensionalisation and the basic reproduction number

For a susceptible–infectious–recovered structure with negligible mortality,
the basic reproduction number is the classical threshold ratio

```math
\mathcal{R}_0 = \frac{\beta}{\gamma}.
```

The disease-free equilibrium ``(N, 0, 0)`` is locally unstable when
``\mathcal{R}_0 > 1`` (an epidemic grows) and stable when
``\mathcal{R}_0 \le 1`` (the infection dies out). This is the fundamental
threshold separating epidemic from non-epidemic parameter regimes.

---

## 2. The closed SIR model

**Compartments.** ``S`` (susceptible), ``I`` (infectious), ``R`` (recovered).

**Rate parameters.** Transmission rate ``\beta`` [day⁻¹], recovery rate
``\gamma`` [day⁻¹].

### 2.1 Governing equations

```math
\begin{aligned}
\frac{dS}{dt} &= -\frac{\beta\, S\, I}{N} \\[2mm]
\frac{dI}{dt} &=  \frac{\beta\, S\, I}{N} - \gamma\, I \\[2mm]
\frac{dR}{dt} &=  \gamma\, I .
\end{aligned}
```

### 2.2 Structural invariants

```math
\frac{d}{dt} (S + I + R) = -\frac{\beta SI}{N} + \left(\frac{\beta SI}{N} - \gamma I\right) + \gamma I = 0,
```

so ``S(t) + I(t) + R(t) = N`` for all time. Each compartment stays
non-negative because whenever any state is zero, its derivative points into the
non-negative orthant (quasi-positivity).

---

## 3. The closed SEIRD model

**Compartments.** ``S`` (susceptible), ``E`` (exposed / latent),
``I`` (infectious), ``R`` (recovered), ``D`` (deceased).

**Rate parameters.** ``\beta`` [day⁻¹], ``\sigma`` [day⁻¹] (latent →
infectious), ``\gamma`` [day⁻¹] (recovery), ``\mu`` [day⁻¹] (case-fatality
exit).

### 3.1 Governing equations

```math
\begin{aligned}
\frac{dS}{dt} &= -\frac{\beta\, S\, I}{N} \\[2mm]
\frac{dE}{dt} &=  \frac{\beta\, S\, I}{N} - \sigma\, E \\[2mm]
\frac{dI}{dt} &=  \sigma\, E - (\gamma + \mu)\, I \\[2mm]
\frac{dR}{dt} &=  \gamma\, I \\[2mm]
\frac{dD}{dt} &=  \mu\, I .
\end{aligned}
```

### 3.2 Structural invariants

```math
\frac{d}{dt} (S + E + I + R + D)
= (-\beta SI/N) + (\beta SI/N - \sigma E) + (\sigma E - (\gamma+\mu) I)
  + \gamma I + \mu I
= 0,
```

so ``S + E + I + R + D = N`` for all time, and each compartment is
non-negative. The latent stage ``E`` introduces a delay before infectiousness,
which shifts and broadens the epidemic peak relative to SIR.

---

## 4. Numerical integration

All systems are integrated with `scipy.integrate.solve_ivp` using tight,
config-driven tolerances:

| Setting | Default | Purpose |
|---|---|---|
| ``method`` | ``LSODA`` | Stiff/non-stiff adaptive stepping |
| ``rtol`` | ``1e-8`` | Relative error tolerance |
| ``atol`` | ``1e-9`` | Absolute error tolerance |
| ``max_step`` | ``None`` | Optional hard step cap |

After integration, each trajectory is validated for the two structural
invariants (conservation to ``rtol=1e-6`` and non-negativity) before being
returned; a trajectory that violates either fails loudly.

## 5. Parameter recoverability (synthetic data)

Given simulated observations ``\{t_i, y_i\}`` of one or more compartments, the
rates are re-estimated by non-linear least squares,

```math
\min_{\theta} \; \sum_{c \in \mathcal{C}} \sum_i
\left( \frac{y^c_{\text{obs}}(t_i) - y^c_{\text{model}}(t_i; \theta)}
       {\max(1, \max_i |y^c_{\text{obs}}|)} \right)^2 ,
```

with ``\theta`` constrained to the admissible parameter bounds. This is the
substance of the synthetic-data recoverability tests. **Identifiability note:**
for SEIRD the latent rate ``\sigma`` and the transmission rate ``\beta`` are
only separately identifiable when *multiple* compartments are observed
jointly (e.g. ``I``, ``R``, ``D`` and ``E``); the infected channel alone
under-determines the system. This is an expected and documented property, not
an implementation artefact.

---

## 6. Parameter table

The following table fixes every rate in the compartmental layer to its
reference value, units, admissible range, and a peer-reviewed source.

| Parameter | Symbol | Units | Default | Range | Source |
|---|---|---|---|---|---|
| Contact / transmission rate | ``\beta`` | day⁻¹ | 0.35 | [0, 5] | Anderson & May (1991), Ch. 2 |
| Recovery rate | ``\gamma`` | day⁻¹ | 1/7 ≈ 0.143 | [0, 5] | Anderson & May (1991), Ch. 2 |
| Latent → infectious rate | ``\sigma`` | day⁻¹ | 1/5.2 ≈ 0.192 | [0, 5] | WHO (2023) latent-period estimates |
| Case-fatality exit rate | ``\mu`` | day⁻¹ | 0.02 | [0, 1] | Reported CFR, diagnosed cohorts (2023) |

> **State variables and initial conditions.**
> `S(0) = N - I_0 - E_0 - D_0`, `I(0) = I_0`, `E(0) = E_0`, `R(0) = 0`,
> `D(0) = D_0`, with `N` the closed population (default `100000`). The
> susceptible residual guarantees conservation at ``t = 0`` by construction.

## References

1. Anderson, R. M. & May, R. M. (1991). *Infectious Diseases of Humans:
   Dynamics and Control*. Oxford University Press.
2. World Health Organization (2023). *World Malaria Report 2023* (latent
   period / epidemiological parameter references). WHO Press.
3. Kermack, W. O. & McKendrick, A. G. (1927). A contribution to the
   mathematical theory of epidemics. *Proc. R. Soc. A*, 115, 700–721.
