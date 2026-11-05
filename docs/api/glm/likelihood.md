# Likelihood-ratio and score tests, profile-likelihood intervals

Model comparison and interval estimation from the likelihood, as in R's
`anova(fit0, fit1, test = "LRT")`, `anova(fit0, fit1, test = "Rao")` and
`confint(glm_fit)`. The statistics come from anofox-regression's
`inference::likelihood` module.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `lr_test` | Scalar | Likelihood-ratio test of two nested fitted models, or of one model against its null model |
| `glm_score_test_agg` | Aggregate | Rao score test for adding predictors to a GLM, evaluated at the reduced fit |
| `glm_score_test_by` | Table macro | `glm_score_test_agg` per group |
| `profile_ci` option | GLM fit option | Profile-likelihood confidence intervals on `poisson_fit_agg`, `binomial_fit_agg`, `logistic_fit_agg`, `negbinom_fit_agg`, `gamma_fit_agg` and `tweedie_fit_agg` |

The examples on this page use this table:

```sql
-- Insurance claims per policy: age band, a bonus class, an urban flag and the exposure
CREATE OR REPLACE TABLE claims AS
SELECT i AS policy,
       'R' || (i % 3) AS region,
       (i % 5)::DOUBLE AS age_band,
       ((i * 7) % 4)::DOUBLE AS bonus,
       (i % 2)::DOUBLE AS urban,
       ln(1 + (i % 4) * 0.5) AS log_exposure,
       (((i * 11) % 7) + (i % 2) * 2 + (i % 5) // 2)::DOUBLE AS n_claims,
       (((i * 7919) % 101) < 15 + 8 * (i % 5) + 15 * (i % 2))::INTEGER::DOUBLE AS lapsed
FROM range(300) r(i);
```

## lr_test

**Signatures:**

```text
lr_test(model_full STRUCT, model_reduced STRUCT) -> STRUCT
lr_test(model STRUCT) -> STRUCT
```

Pass two model STRUCTs returned by the same `*_fit_agg` (or `*_fit`) function.
The reduced model must be nested in the full one: its predictors a subset of
the full model's, fitted on the same rows.

- **The statistic.** For binomial and Poisson GLMs it is the deviance
  difference (`anova(test = "LRT")`); for GLMs with an estimated dispersion
  (Gamma, Tweedie) the deviance difference divided by the full model's
  dispersion, as `anova.glm` does; for every other model (OLS, WLS, the
  Negative Binomial, AFT, GLMM, ALM, ...) `-2 (log L_reduced - log L_full)`, as
  `lmtest::lrtest`. The Negative Binomial uses the log-likelihood because the
  two fits estimate different `theta`.
- **Degrees of freedom.** `df1` is the difference in estimated parameters:
  the coefficients that are not NaN (a column dropped as constant or aliased
  is not a parameter) plus a fitted intercept. Nuisance parameters common to
  both fits (dispersion, AFT scale, NB theta, GLMM variance components) cancel.
- **One argument.** `lr_test(model)` tests the model against its null
  (intercept-only) model on `k - 1` degrees of freedom: `null_deviance -
  deviance` for GLMs, `2 (log L - null_log_likelihood)` for AFT fits (the
  "Model L.R." of `rms`). Models without a null likelihood (OLS) raise an error;
  use the F test of `ols_fit_agg` / [`ols_anova_agg`](../regression/ols.md#nested-model-f-test-ols_anova_agg--ols_anova_by) there.
- **Checks.** Both models must have the same `model_type`, the same `family`
  and the same `n_observations`, and `model_full` must have more parameters;
  otherwise the call raises an error. Nesting of the predictors themselves
  cannot be seen from the fitted STRUCTs and is the caller's responsibility.
  Compare GLMM fits with `reml := false` (ML), as R's `anova(lmer)` refits.
- **NULL.** A NULL model (e.g. an empty group) gives NULL.

The result follows the integration contract's `test` schema:

| Field | Type | Description |
|-------|------|-------------|
| `method` | `VARCHAR` | `'likelihood ratio'` |
| `statistic` | `DOUBLE` | LR chi-square |
| `df1` | `DOUBLE` | Difference in estimated parameters |
| `p_value` | `DOUBLE` | Upper tail of `chi^2(df1)` |
| `dispersion` | `DOUBLE` | Dispersion the deviance difference was divided by (1 for fixed-dispersion and log-likelihood tests) |
| `n_parameters_full`, `n_parameters_reduced` | `BIGINT` | Estimated parameters of the two models |
| `n_observations` | `BIGINT` | Rows of the fits |
| `alternative`, `df2`, `estimate`, `conf_low`, `conf_high`, `conf_level`, `effect_size`, `effect_name`, `n` | | Unified test fields (NULL where they do not apply; `n` = rows) |

**Examples:**

```sql
-- Does the bonus class add to a Poisson model of claim counts? (anova(..., test = "LRT"))
SELECT round(t.statistic, 4) AS lr_chisq, t.df1, round(t.p_value, 4) AS p_value
FROM (SELECT lr_test(poisson_fit_agg(n_claims, [age_band, bonus, urban]),
                     poisson_fit_agg(n_claims, [age_band])) AS t
      FROM claims);

-- Per region, and the model LR test of a logistic model against the null model
SELECT region,
       round((lr_test(poisson_fit_agg(n_claims, [age_band, bonus]),
                      poisson_fit_agg(n_claims, [age_band]))).p_value, 4) AS p_bonus,
       round((lr_test(logistic_fit_agg(lapsed, [age_band, urban]))).statistic, 4) AS model_lr
FROM claims GROUP BY region ORDER BY region;
```

## glm_score_test_agg / glm_score_test_by

**Signatures:**

```text
glm_score_test_agg(y DOUBLE, x_reduced DOUBLE[], x_add DOUBLE[] [, options MAP]) -> STRUCT
glm_score_test_by(source VARCHAR, group_col, y_col, x_reduced := [], x_add := [...], options := NULL) -> TABLE
```

Rao's score (Lagrange multiplier) test of adding the `x_add` columns to the
GLM of `y` on `x_reduced`. Only the reduced model is fitted, so the test is
cheap and works where the larger model is hard to fit. It reproduces
`anova(glm(y ~ x_reduced), glm(y ~ x_reduced + x_add), test = "Rao")`, and the
per-column `z_values` are those of `statmod::glm.scoretest`.

- **Rows.** A row is used only when `y`, both lists and every list element are
  non-NULL and finite. `x_reduced := []` is the intercept-only model.
- **Dispersion.** 1 for the binomial, Poisson and Negative Binomial families;
  for the Gaussian, Gamma and Tweedie families the Pearson estimate of the
  larger model, as `anova(test = "Rao")` (the larger model is fitted only for
  that estimate). `z_values` use the same dispersion.
- **Degenerate data.** An empty group gives NULL; collinear `x_add` columns
  leave the statistic undefined (NULL).

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `family` | VARCHAR | `'gaussian'` | `'gaussian'`, `'binomial'`, `'poisson'`, `'negbinom'`, `'gamma'`, `'tweedie'` |
| `link` | VARCHAR | canonical | Binomial: `'logit'`, `'probit'`, `'cloglog'`; Poisson: `'log'`, `'identity'`, `'sqrt'`; Gaussian: `'identity'`; the others `'log'` |
| `power` | DOUBLE | `1.5` | Tweedie variance power in `[1, 2]` |
| `theta` | DOUBLE | estimated | Negative Binomial shape; when omitted, the estimate of the reduced `glm.nb` fit |
| `fit_intercept` | BOOLEAN | `true` | Include an intercept in both models |
| `offset` | INTEGER | none | 1-based index into `x_reduced` of an offset column (coefficient fixed at 1) |

**Result:** `method` (`'Rao score'`), `statistic` (score chi-square), `df1`
(number of added columns), `p_value`, `z_values` (one per `x_add` column),
`dispersion`, `theta` (NULL except for the Negative Binomial), `n_observations`,
then the unified test fields. The table macro returns one row per group with
the group column, `id`, `model_id` / `model_name` (`'glm'`), `method`,
`statistic`, `p_value`, `df1`, `z_values`, `dispersion`, `theta` and
`n_observations`.

**Examples:**

```sql
-- Score test for bonus and urban in a Poisson model with an exposure offset
SELECT round(t.statistic, 4) AS score_chisq, t.df1, round(t.p_value, 4) AS p_value,
       [round(z, 3) FOR z IN t.z_values] AS z
FROM (SELECT glm_score_test_agg(n_claims, [age_band, log_exposure], [bonus, urban],
                                {'family': 'poisson', 'offset': 2}) AS t
      FROM claims);

-- Per region, logistic model of lapse
SELECT region, round(statistic, 4) AS score_chisq, round(p_value, 4) AS p_value
FROM glm_score_test_by('claims', region, lapsed, x_reduced := [age_band], x_add := [urban, bonus],
                       options := {'family': 'binomial'});
```

## profile_ci option

`{'profile_ci': true}` on `poisson_fit_agg`, `binomial_fit_agg`,
`logistic_fit_agg`, `negbinom_fit_agg`, `gamma_fit_agg` and `tweedie_fit_agg`
replaces the Wald intervals by profile-likelihood intervals, R's
`confint(glm_fit)` (`MASS::confint.glm` before R 4.4): `conf_low`,
`conf_high`, their `ci_lower` / `ci_upper` aliases and `intercept_conf_low` /
`intercept_conf_high`, at `confidence_level`. The STRUCT keeps its shape, so
[`tidy`](../regression/model_tools.md) shows the profile intervals.

- Each coefficient is profiled on R's grid and the signed root deviance is
  inverted with R's spline, so the bounds agree with `confint` to about 1e-6.
  A bound the profile does not reach is NULL (R's `NA`).
- The profile uses the rows, columns and `offset` of the fit; a column the fit
  dropped keeps a NaN interval, like its other inference values. The Negative
  Binomial is profiled at the fitted `theta` (as `confint(glm.nb)`); Gamma and
  Tweedie scale the deviance by the estimated dispersion.
- `profile_ci` implies `compute_inference` (standard errors, z and p values
  stay the Wald ones). It needs an unpenalized fit: combined with `glm_lambda >
  0` or a `prior` it raises an error.
- It costs one IRLS fit per grid point (about 20 per coefficient).

**Example:**

```sql
-- Profile (confint) versus Wald intervals of a logistic model
SELECT [round(v, 4) FOR v IN p.conf_low] AS profile_low, [round(v, 4) FOR v IN p.conf_high] AS profile_high,
       [round(v, 4) FOR v IN w.conf_low] AS wald_low, [round(v, 4) FOR v IN w.conf_high] AS wald_high
FROM (SELECT logistic_fit_agg(lapsed, [age_band, urban], {'profile_ci': true}) AS p,
             logistic_fit_agg(lapsed, [age_band, urban], {'compute_inference': true}) AS w
      FROM claims);
```
