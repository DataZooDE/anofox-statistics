# Generalized Least Squares

Linear regression for repeated measures and longitudinal data whose errors are
**correlated within a subject** and may have **unequal variances**: R's
`nlme::gls` in SQL.

```text
y = X b + e,   Var(e_g) = sigma^2 A_g R_g A_g   for every subject g
```

`R_g` is the within-subject correlation structure (AR(1), compound symmetry,
continuous-time AR(1), spatial-type functions of the time distance, unstructured) and
the diagonal `A_g` a variance function (`varIdent`, `varPower`, `varExp`). Subjects are
independent of each other. The correlation and variance parameters maximise the REML
(default) or ML log-likelihood. Use it where a mixed model's random intercept is the
wrong shape for the dependence, for example an AR(1) decay over visits.

> The estimator lives upstream in `anofox-regression` (`solvers::gls::GlsRegressor`);
> this extension is a wrapper. Results match `nlme::gls` (`test/sql/reference/gls_reference.test`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `gls_fit_agg` | Aggregate | Fit a GLS model |
| `gls_augment_agg` | Window Aggregate | Per-row fitted values, their standard errors and residuals |
| `gls_fit_by` | Table Macro | One fit per group, the model fields as columns |
| `gls_tidy_by` | Table Macro | Coefficients and variance-covariance parameters as `terms` rows |
| `gls_glance_by` | Table Macro | Fit statistics as `summary` rows |
| `gls_augment_by` | Table Macro | Per-row results as `obs` rows |

The examples on this page use this table:

```sql
-- 20 patients, up to 6 weekly visits, two treatment arms
CREATE OR REPLACE TABLE visits AS
SELECT 'c' || (i // 60) AS clinic,
       'p' || lpad((i // 6)::VARCHAR, 2, '0') AS patient,
       ['placebo', 'active'][(i // 6) % 2 + 1] AS arm,
       (i % 6 + 1) AS week,
       0.5 + ((i * 13) % 11) / 4.0 AS dose,
       (2 + 0.8 * (0.5 + ((i * 13) % 11) / 4.0) - 0.2 * (i % 6 + 1) + ((i // 6) % 5) * 0.3
        + (((i * 37) % 17) - 8) / 8.0 * (1 + (i // 6) % 2))::DOUBLE AS score
FROM range(120) r(i)
WHERE i NOT IN (14, 75);
```

## gls_fit_agg

**Signature:**

```text
gls_fit_agg(y DOUBLE, x DOUBLE[], subject ANY, time DOUBLE [, options]) -> STRUCT
gls_fit_agg(y DOUBLE, x DOUBLE[], subject ANY, time DOUBLE, variance_covariate ANY, options) -> STRUCT
```

* `subject` (any type) groups the rows whose errors are correlated (`form = ~ time | subject`).
* `time` places a row within its subject: the lag of `ar1` (integer times), the
  distance of `car1` and the spatial structures, the position of `unstructured` (each
  distinct time is a position). `independent` and `compound_symmetry` do not use it, but
  it still orders each subject's rows.
* `variance_covariate` feeds the variance function: a stratum of any type for
  `'ident'`, a number for `'power'` / `'exp'`. It is ignored for `'constant'`.

**Options** (MAP or STRUCT):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| correlation | VARCHAR | 'independent' | `independent`, `compound_symmetry` (alias `cs`; `corCompSymm`), `ar1` (`corAR1`, integer times), `car1` (`corCAR1`), `exponential` (`corExp`), `gaussian` (`corGaus`), `linear` (`corLin`), `spherical` (`corSpher`), `unstructured` (alias `symm`; `corSymm`) |
| nugget | BOOLEAN | false | Estimate a nugget (spatial structures only) |
| variance | VARCHAR | 'constant' | `constant`, `ident` (`varIdent(form = ~ 1 \| v)`), `power` (`varPower(form = ~ v)`, `v` non-zero), `exp` (`varExp(form = ~ v)`) |
| method | VARCHAR | 'reml' | `reml` or `ml` (compare models with different fixed effects by ML) |
| reml | BOOLEAN | true | Same as `method` (as `glmm_fit_agg`) |
| fit_intercept | BOOLEAN | true | Include an intercept |
| compute_inference | BOOLEAN | false | t tests of the coefficients (N - p df, as `summary.gls`) and the approximate var-cov intervals (`intervals.gls`) |
| confidence_level | DOUBLE | 0.95 | Level of both intervals |
| max_iterations | INTEGER | 500 | L-BFGS iterations for the variance-covariance parameters |
| variogram | INTEGER | 0 | Number of classes of the empirical semivariogram (`nlme::Variogram`, `collapse = "quantiles"`); 0 = none |
| variogram_residuals | VARCHAR | 'pearson' | Residuals of the variogram: `pearson`, `normalized` or `response` |

**Returns** the stable model STRUCT: `coefficients` (without the intercept),
`intercept`, `correlation`, `variance`, `method`, `sigma`, `correlation_params`,
`variance_params`, `variance_strata`, `variance_reference`, `n_observations`,
`n_groups` (subjects), `n_features`, `n_params`, `iterations`, `converged`, the
inference lists `std_errors`, `t_values`, `p_values`, `conf_low`, `conf_high` and
`intercept_*` (NULL without `compute_inference`), the var-cov intervals
`correlation_conf_low` / `_high`, `variance_conf_low` / `_high`, `sigma_conf_low` /
`_high` (NULL without `compute_inference`, or when the Hessian of the log-likelihood is
not negative definite), `variogram` (LIST of `dist`, `variog`, `n_pairs`),
`log_likelihood`, `aic`, `bic`, `model_type` ('gls'), `family` ('gaussian'), `link`,
`vcov` (row-major, intercept first) and `df_residual`.

The parameters are on their natural scale (`coef(fit$modelStruct, unconstrained = FALSE)`):

| correlation | `correlation_params` |
|-------------|----------------------|
| `compound_symmetry` | `rho` |
| `ar1`, `car1` | `phi` |
| spatial | `range`, then `nugget` with `'nugget': true` |
| `unstructured` | the correlations `r12, r13, ..., r23, ...` of the sorted distinct times |

`variance_params` holds `delta` for `power` / `exp`, and for `ident` the SD ratio of
every stratum in `variance_strata` to the reference stratum `variance_reference`.

```sql
-- AR(1) errors over the visits
SELECT r.intercept, r.coefficients, r.correlation_params[1] AS phi, r.sigma, r.aic
FROM (SELECT gls_fit_agg(score, [dose, week], patient, week, {'correlation': 'ar1'}) AS r FROM visits);

-- with t tests and the interval of phi
SELECT r.std_errors, r.p_values, r.correlation_conf_low[1] AS phi_lo, r.correlation_conf_high[1] AS phi_hi
FROM (SELECT gls_fit_agg(score, [dose, week], patient, week,
                         {'correlation': 'ar1', 'compute_inference': true}) AS r FROM visits);

-- AR(1) and a separate residual SD per arm (varIdent), fitted by ML
SELECT r.variance_reference, r.variance_strata, r.variance_params, r.log_likelihood
FROM (SELECT gls_fit_agg(score, [dose, week], patient, week, arm,
                         {'correlation': 'ar1', 'variance': 'ident', 'method': 'ml'}) AS r FROM visits);

-- compound symmetry with an SD that grows as |dose|^delta
SELECT (gls_fit_agg(score, [dose, week], patient, week, dose,
                    {'correlation': 'compound_symmetry', 'variance': 'power'})).variance_params
FROM visits;

-- the empirical semivariogram of the Pearson residuals
SELECT unnest((gls_fit_agg(score, [dose, week], patient, week, {'variogram': 10})).variogram, recursive := true)
FROM visits;

-- one model per clinic
SELECT clinic, (gls_fit_agg(score, [dose], patient, week, {'correlation': 'ar1'})).correlation_params
FROM visits GROUP BY clinic ORDER BY clinic;
```

**Row order and the varIdent reference.** The fit does not depend on the order in which
DuckDB delivers the rows: they are handed to the solver sorted by subject (by its text
form) and time. Two consequences: `normalized_residual` decorrelates each subject's
rows in time order, and the varIdent reference stratum is the stratum of the earliest
row of the first subject. `nlme` takes the stratum of the first row of the data, which
is the same row when the data are sorted by subject and time. `variance_reference`
names it.

**NULL handling.** A row with a NULL `y`, `x` (or a NULL element of `x`), `subject`,
`time` or, when the variance function uses it, `variance_covariate` is left out. Data
that cannot support the fit return NULL: no rows, fewer rows than coefficients,
compound symmetry without a subject with two rows, an unstructured correlation with
a single time, or a variance-covariance optimisation that fails (for example AR(1)
with tied times within a subject).

**Errors.** Unknown options or option values, `nugget` without a spatial correlation,
a variance function without a `variance_covariate` (or with a NULL one), a
non-numeric covariate for `power` / `exp`, a zero `power` covariate, non-integer
times for `ar1` and an inconsistent number of features raise an error.

## gls_augment_agg

```text
gls_augment_agg(y, x, subject, time [, variance_covariate], options) OVER (...) -> LIST(STRUCT)
```

A window aggregate that fits the same model as `gls_fit_agg` over its frame and
returns one STRUCT per frame row, in frame order: `fitted` (`x'b`, for every row with
`x`), `se_fit` (`sqrt(x' V x)` with `V = vcov`), `residual` (`y - fitted`),
`pearson_residual` (`resid(fit, "pearson")`), `normalized_residual`
(`resid(fit, "normalized")`) and `is_training`. A row the fit could not use keeps its
`fitted` / `se_fit` when it has `x` (a prediction) and gets NULL residuals. Pair it with
`ROW_NUMBER()` over the same window:

```sql
SELECT patient, week, (a).fitted, (a).se_fit, (a).normalized_residual
FROM (SELECT patient, week, l[rn] AS a FROM (
    SELECT patient, week, row_number() OVER w AS rn,
           gls_augment_agg(score, [dose, week], patient, week, {'correlation': 'ar1'}) OVER w AS l
    FROM visits
    WINDOW w AS (ORDER BY patient, week ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)))
ORDER BY patient, week
LIMIT 5;
```

## gls_fit_by, gls_tidy_by, gls_glance_by, gls_augment_by

```text
gls_fit_by(source, group_col, y_col, x_cols, subject_col, time_col [, options := ...] [, variance_col := ...])
gls_tidy_by(...    [, names := [...]])
gls_glance_by(...)
gls_augment_by(...)
```

All four fit one model per `group_col`; `variance_col := <column>` is the variance
covariate. `gls_fit_by` returns one row per group with every field of the
`gls_fit_agg` STRUCT. The other three emit the integration contract's schemas:

* `gls_tidy_by` (`terms`): `effect = 'coefficient'` rows with the t tests (inference is
  always computed), then `'correlation'` rows (`phi`, `rho`, `range` / `nugget` or
  `cor_<k>`), `'variance'` rows (`sd_ratio_<stratum>` or `delta`) and the `'sigma'` row,
  each with its var-cov interval. `names := [...]` labels the coefficients.
* `gls_glance_by` (`summary`): `n_observations`, `n_groups`, `n_features`, `n_params`,
  `sigma`, `log_likelihood`, `aic`, `bic`, `df_residual`, `iterations`, `converged`.
* `gls_augment_by` (`obs`): every source row with `fitted`, `residual`, `std_residual`
  (the Pearson residual), `is_training`, then `se_fit` and `normalized_residual`.

```sql
SELECT clinic, correlation_params, sigma, aic
FROM gls_fit_by('visits', clinic, score, [dose, week], patient, week, options := {'correlation': 'ar1'});

SELECT clinic, effect, term, estimate, conf_low, conf_high
FROM gls_tidy_by('visits', clinic, score, [dose, week], patient, week,
                 options := {'correlation': 'ar1', 'variance': 'ident'}, variance_col := arm,
                 names := ['dose', 'week']);

SELECT clinic, metric, value
FROM gls_glance_by('visits', clinic, score, [dose, week], patient, week, options := {'correlation': 'cs'});

SELECT patient, week, fitted, se_fit, normalized_residual
FROM gls_augment_by('visits', clinic, score, [dose, week], patient, week, options := {'correlation': 'ar1'})
LIMIT 5;
```

## Choosing a structure

Fit the candidates by REML with the same fixed effects and compare their AIC:

```sql
SELECT 'independent' AS cor, (gls_fit_agg(score, [dose, week], patient, week)).aic AS aic FROM visits
UNION ALL
SELECT 'compound_symmetry', (gls_fit_agg(score, [dose, week], patient, week, {'correlation': 'cs'})).aic FROM visits
UNION ALL
SELECT 'ar1', (gls_fit_agg(score, [dose, week], patient, week, {'correlation': 'ar1'})).aic FROM visits
ORDER BY aic;
```

To compare fixed effects use `'method': 'ml'` for both models, for example with
[`lr_test`](../glm/likelihood.md).
