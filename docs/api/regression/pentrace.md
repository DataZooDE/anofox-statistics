# Penalized maximum likelihood (pentrace)

Penalized estimation in the style of Frank Harrell's rms (Regression Modeling
Strategies, section 9.10): the coefficients are shrunk by a quadratic penalty
`½ βᵀPβ` on the *standardised* coefficients, with a different penalty for each
type of column (linear main effects, the nonlinear columns of a spline, linear
interactions, nonlinear interactions). `pentrace_agg` is `rms::pentrace`: it
fits the model at every penalty of a grid and scores each fit by AIC, BIC and
the corrected AIC computed with the *effective* degrees of freedom, so the
penalty can be chosen without cross-validation. `penalized_ols_fit_agg` is
`rms::ols(penalty = )`, the penalized least-squares fit at one penalty. The
computations are anofox-regression's `solvers::pentrace` (`pentrace`,
`pentrace_ols`, `penalized_ols`, `penalty_matrix`, `penalty_grid`,
`effective_df_by_term`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `pentrace_agg(y, x, options)` | Aggregate | `rms::pentrace`: one row per penalty with the effective d.f., LR chi-square, AIC, BIC, corrected AIC and the best row |
| `penalized_ols_fit_agg(y, x [, options])` | Aggregate | `rms::ols(penalty = )`: coefficients, covariance, effective d.f. (total, per coefficient, per term) |
| `pentrace_by(source, group_col, y_col, x_cols, options := ...)` | Table macro | `pentrace_agg` per group, one row per group and penalty |
| `penalized_ols_fit_by(source, group_col, y_col, x_cols, options := ...)` | Table macro | `penalized_ols_fit_agg` per group, one row per group |

Rows with a NULL `y`, a NULL `x` or a NULL list element are skipped (as the
[path aggregates](paths.md)); a group without usable rows gives NULL. A
constant column (aliased with the intercept) and a group with no more rows than
coefficients give NULL.

The examples use this table; `age1 .. age3` play the part of a restricted cubic
spline basis of age (`rcs(age, 4)` in rms: one linear and two nonlinear
columns), `site_b` / `site_c` are the dummies of a three-level factor.

```sql
CREATE OR REPLACE TABLE pen_demo AS
SELECT i AS id,
       'cohort' || (i // 120) AS cohort,
       (20 + (i * 37) % 50) / 10.0 AS age1,
       greatest((20 + (i * 37) % 50) / 10.0 - 3.5, 0) ^ 3 / 10 AS age2,
       greatest((20 + (i * 37) % 50) / 10.0 - 5.0, 0) ^ 3 / 10 AS age3,
       (100 + (i * 13) % 40)::DOUBLE AS bp,
       (i % 2 = 0)::INTEGER::DOUBLE AS treated,
       (i % 3 = 1)::INTEGER::DOUBLE AS site_b,
       (i % 3 = 2)::INTEGER::DOUBLE AS site_c,
       0.4 * (20 + (i * 37) % 50) / 10.0 + 0.03 * (100 + (i * 13) % 40) + 0.8 * (i % 2 = 0)::INTEGER
           + 0.3 * (i % 3 = 1)::INTEGER + 1.5 * sin(i * 1.3) AS outcome,
       ((i * 7919) % 100 < (20 + (i * 37) % 50) * 1.2 - 15)::INTEGER::DOUBLE AS event,
       (i * 7 % 4)::DOUBLE AS grade
FROM range(240) r(i);
```

## Terms and penalty types

The penalty of a column depends on its type, so both aggregates take the same
term specification as [`wald_anova`](wald_anova.md#terms):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `terms` | `VARCHAR[]` | every column its own term `x1 .. xk` | One label per column of `x`; columns with the same label form a term (e.g. the columns of a spline basis). Every column needs a label |
| `nonlinear` | `BOOLEAN[]` | all `false` | One flag per column: the nonlinear columns of a term (the `'` and `''` columns of an rcs basis). Ignored for interaction columns |
| `interactions` | `MAP(VARCHAR, VARCHAR[])` | none | Interaction label to its two main-effect factor labels, e.g. `MAP {'age * treated': ['age', 'treated']}`. Columns in rms order (full `a * b` product, first factor fastest, or the restricted `a %ia% b`) |
| `categorical` | `VARCHAR[]` | none | Labels of the terms whose columns are the dummies of a factor |

- **Penalty matrix.** As `rms:::Penalty.matrix`: the sample variance of each
  column on the diagonal (so the penalty acts on standardised coefficients),
  except that the `L` dummies of a categorical term get the block
  `I − 1/(L + 1)`, which penalises the differences between all `L + 1`
  levels, the reference level included. Each entry is then scaled by
  `√λ_i √λ_j`, with `λ` the penalty of the column's type.
- **Types.** A column is *simple* (a linear main effect or a dummy),
  *nonlinear* (a nonlinear column of a main effect), *interaction* (an
  interaction column whose components are both linear) or *nonlinear
  interaction*. Unset penalties fall back as in rms: nonlinear to simple,
  interaction to nonlinear, nonlinear interaction to interaction.
- The intercept is never penalised. `terms` and the options must be constants.

## pentrace_agg

**Signature:**

```text
pentrace_agg(y DOUBLE, x DOUBLE[], options MAP | STRUCT) -> STRUCT[]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `penalties` | `DOUBLE[]` or `STRUCT` | required | A list of penalties for every column type (`pentrace(fit, c(0, 1, 2))`), or the values by type: `{'simple': [...], 'nonlinear': [...], 'interaction': [...], 'nonlinear_interaction': [...]}` (`pentrace(fit, list(simple = ..., nonlinear = ...))`). All combinations are traced, in R's `expand.grid` order (`simple` fastest), keeping only those in which no type has a smaller penalty than a simpler one (rms `complex.more = TRUE`) |
| `model` | `VARCHAR` | `'ols'` | `'ols'` (least squares, rms `ols`), `'logistic'` (binary response, rms `lrm`) or `'ordinal'` (an ordered response with any number of levels, the cumulative-link model of rms `lrm` / `orm`) |
| `link` | `VARCHAR` | `'logistic'` | `'logistic'` and `'ordinal'` only: `'logistic'`, `'probit'`, `'loglog'`, `'cloglog'` or `'cauchit'` (rms `orm(family = )`) |
| `criterion` (alias `which`) | `VARCHAR` | `'aic_c'` | The criterion the best row maximises: `'aic_c'` (rms default `which = "aic.c"`), `'aic'` or `'bic'` |
| `terms`, `nonlinear`, `interactions`, `categorical` | | | See [Terms and penalty types](#terms-and-penalty-types) |

Each list element (one per penalty, in grid order):

| Field | Type | Description |
|-------|------|-------------|
| `simple`, `nonlinear`, `interaction`, `nonlinear_interaction` | `DOUBLE` | The penalty of each column type, after the fallbacks |
| `df` | `DOUBLE` | Effective d.f. of the slopes (rms `d.f.`) |
| `lr_chisq` | `DOUBLE` | Model likelihood-ratio chi-square at the penalized estimate (rms `Model L.R.`) |
| `aic` | `DOUBLE` | `lr_chisq − 2 df` (rms's AIC on the chi-square scale: larger is better) |
| `bic` | `DOUBLE` | `lr_chisq − df · log n` |
| `aic_c` | `DOUBLE` | `lr_chisq − 2 df (1 + (df + 1) / (n − df − 1))`, the corrected AIC |
| `is_best` | `BOOLEAN` | The row with the largest `criterion` (rms `pentrace(...)$penalty`) |
| `n_observations` | `BIGINT` | Rows used |

For least squares the effective d.f. is rms's `sum(diag(X'X V)) / (SSE / n) − 1`
with `V = σ̂² (X'X + P)⁻¹` and `σ̂² = (SSE + β̂'Pβ̂) / n`; since `σ̂²` grows with
the penalty, this d.f. is not always decreasing in the penalty (rms behaves
the same).

**Examples:**

```sql
-- Uniform penalty on a spline, a treatment effect and a three-level factor
SELECT r.simple AS penalty, round(r.df, 3) AS df, round(r.aic_c, 3) AS aic_c, r.is_best
FROM (SELECT unnest(pentrace_agg(outcome, [age1, age2, age3, bp, treated, site_b, site_c],
          {'terms': ['age', 'age', 'age', 'bp', 'treated', 'site', 'site'],
           'nonlinear': [false, true, true, false, false, false, false],
           'categorical': ['site'],
           'penalties': [0, 1, 2, 4, 8, 16, 32]})) AS r
      FROM pen_demo);

-- Penalise the nonlinear spline columns more than the linear ones
SELECT r.simple, r.nonlinear, round(r.df, 3) AS df, round(r.aic_c, 3) AS aic_c, r.is_best
FROM (SELECT unnest(pentrace_agg(outcome, [age1, age2, age3, bp, treated],
          {'terms': ['age', 'age', 'age', 'bp', 'treated'],
           'nonlinear': [false, true, true, false, false],
           'penalties': {'simple': [0, 2], 'nonlinear': [0, 4, 16, 64]}})) AS r
      FROM pen_demo);

-- A penalized binary logistic model with a spline-by-treatment interaction (its two nonlinear
-- columns take the nonlinear-interaction penalty), choosing by BIC
SELECT r.simple AS penalty, round(r.df, 3) AS df, round(r.bic, 3) AS bic, r.is_best
FROM (SELECT unnest(pentrace_agg(event, [age1, age2, age3, treated, age1 * treated, age2 * treated, age3 * treated],
          {'terms': ['age', 'age', 'age', 'treated', 'age * treated', 'age * treated', 'age * treated'],
           'nonlinear': [false, true, true, false, false, false, false],
           'interactions': MAP {'age * treated': ['age', 'treated']},
           'penalties': [0, 1, 2, 4, 8], 'model': 'logistic', 'criterion': 'bic'})) AS r
      FROM pen_demo);

-- An ordinal response (four grades) with a probit link
SELECT r.simple AS penalty, round(r.df, 3) AS df, round(r.aic_c, 3) AS aic_c, r.is_best
FROM (SELECT unnest(pentrace_agg(grade, [age1, bp, treated],
          {'penalties': [0, 2, 8], 'model': 'ordinal', 'link': 'probit'})) AS r
      FROM pen_demo);

-- The best penalty per cohort
SELECT cohort, simple AS best_penalty, round(df, 3) AS df
FROM pentrace_by('pen_demo', cohort, outcome, [age1, bp, treated], options := {'penalties': [0, 1, 4, 16, 64]})
WHERE is_best
ORDER BY cohort;
```

## penalized_ols_fit_agg

**Signature:**

```text
penalized_ols_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP | STRUCT]) -> STRUCT
```

Penalized least squares `min ||y − β₀ − Xβ||² + β'Pβ` with rms's penalty
matrix `P` (`lm.pfit`); without a penalty it is ordinary least squares.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `penalty` | `DOUBLE` or `STRUCT` | `0` | One penalty for every column type, or `{'simple': ..., 'nonlinear': ..., 'interaction': ..., 'nonlinear_interaction': ...}` (`ols(penalty = list(...))`) |
| `var_penalty` | `VARCHAR` | `'simple'` | Covariance of the coefficients: `'simple'` (`σ̂² (X'X + P)⁻¹`) or `'sandwich'` (`σ̂² (X'X + P)⁻¹ X'X (X'X + P)⁻¹`), as rms `var.penalty` |
| `terms`, `nonlinear`, `interactions`, `categorical` | | | See [Terms and penalty types](#terms-and-penalty-types) |

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Slopes, in the order of `x` |
| `intercept` | `DOUBLE` | Intercept (not penalised) |
| `std_errors`, `intercept_std_error` | `DOUBLE[]`, `DOUBLE` | Square roots of the diagonal of `vcov` |
| `vcov` | `DOUBLE[]` | Coefficient covariance by `var_penalty`, row-major `(k+1) × (k+1)`, intercept first |
| `var_penalty` | `VARCHAR` | `'simple'` or `'sandwich'` |
| `r_squared` | `DOUBLE` | `1 − SSE / SST` |
| `sigma` | `DOUBLE` | rms `Sigma`, `√((SSE + β̂'Pβ̂) / n)` |
| `lr_chisq` | `DOUBLE` | rms `Model L.R.` |
| `effective_df` | `DOUBLE` | Effective d.f. of the slopes (rms `d.f.`) |
| `effective_df_diagonal` | `DOUBLE[]` | Effective d.f. of each coefficient, intercept first (rms `effective.df.diagonal`) |
| `terms` | `VARCHAR[]` | The term labels, in order of first appearance |
| `effective_df_by_term` | `DOUBLE[]` | Effective d.f. of each term (the sum over its columns), aligned with `terms` (rms `effective.df(fit)`) |
| `simple`, `nonlinear`, `interaction`, `nonlinear_interaction` | `DOUBLE` | The penalty used, after the fallbacks |
| `n_observations`, `n_features` | `BIGINT` | Rows and columns used |
| `model_type` | `VARCHAR` | `'penalized_ols'` |

The t-tests and adjusted R² that rms prints for a penalized fit (with `n − d.f. − 1`
residual degrees of freedom) are not part of the result.

**Examples:**

```sql
-- Differential penalties (spline nonlinearity penalised more), with the effective d.f. of each term
SELECT m.terms, list_transform(m.effective_df_by_term, lambda d: round(d, 3)) AS df_by_term,
       round(m.effective_df, 3) AS df, round(m.sigma, 4) AS sigma
FROM (SELECT penalized_ols_fit_agg(outcome, [age1, age2, age3, bp, treated, site_b, site_c],
          {'terms': ['age', 'age', 'age', 'bp', 'treated', 'site', 'site'],
           'nonlinear': [false, true, true, false, false, false, false],
           'categorical': ['site'], 'penalty': {'simple': 2, 'nonlinear': 8}}) AS m
      FROM pen_demo);

-- Sandwich covariance; standard errors of the slopes
SELECT list_transform(m.std_errors, lambda s: round(s, 4)) AS se
FROM (SELECT penalized_ols_fit_agg(outcome, [age1, bp, treated], {'penalty': 4, 'var_penalty': 'sandwich'}) AS m
      FROM pen_demo);

-- Per group
SELECT cohort, list_transform(coefficients, lambda b: round(b, 4)) AS coefficients, round(effective_df, 3) AS df
FROM penalized_ols_fit_by('pen_demo', cohort, outcome, [age1, bp, treated], options := {'penalty': 4})
ORDER BY cohort;
```

## Validation

`test/sql/reference/pentrace_reference.test` (generated by
`validation/generators/make_reference_pentrace.R`) compares every row of
`rms::pentrace` (OLS with a spline, a linear and a nonlinear interaction and a
categorical term; binary and ordinal `lrm`) and the full `rms::ols(penalty = )`
fit (coefficients, both covariances, `Model L.R.`, `d.f.`, `R2`, `Sigma`,
`effective.df.diagonal`). For a non-logistic link the reference is
`orm(family = "probit", penalty = λ)` fitted at each penalty, because
`rms::pentrace` refits an `orm` model with the logistic `lrm.fit` whatever its
family.
