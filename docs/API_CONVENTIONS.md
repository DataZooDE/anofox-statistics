# API Conventions

**Applies to:** anofox-statistics v0.10.0 and later
**Status:** Authoritative. The runnable SQL examples in this document are executed against the extension in CI by `scripts/validate_docs_sql.py`.

---

## 1. Function Naming Convention

### Pattern

```
{model}_{verb}[_{suffix}]
```

All functions are **unprefixed and uniform**. The `anofox_stats_` prefix that existed up to v0.9.x was removed in v0.10.0 (breaking change, see §5 and [MIGRATION.md](MIGRATION.md)).

### Model component

The statistical model or family name, using snake_case:

| Model component | Description |
|----------------|-------------|
| `ols` | Ordinary Least Squares |
| `ridge` | Ridge (L2-penalized) regression |
| `elasticnet` | Elastic-net (L1+L2) regression |
| `lars` | Least Angle Regression |
| `wls` | Weighted Least Squares |
| `huber` | Huber robust regression |
| `ransac` | RANSAC robust regression |
| `theil_sen` | Theil-Sen robust regression (was `theilsen` up to v0.9.x) |
| `rls` | Recursive Least Squares |
| `bls` | Bounded Least Squares |
| `nnls` | Non-Negative Least Squares |
| `pls` | Partial Least Squares |
| `isotonic` | Isotonic (monotonic) regression |
| `quantile` | Quantile regression |
| `poisson` | GLM with Poisson family |
| `binomial` | GLM with Binomial family (logit / probit / cloglog link) |
| `logistic` | Binary classification (Binomial GLM, logit link) |
| `negbinom` | GLM with Negative Binomial family |
| `gamma` | GLM with Gamma family |
| `tweedie` | GLM with Tweedie family |
| `alm` | Augmented Linear Model (24 error distributions) |
| `aft` | Accelerated Failure Time (survival) |
| `glmm` | Generalized linear mixed model |
| `eb_shrink` | Empirical-Bayes shrinkage |
| `aid` | Automatic Identification of Demand (demand-pattern classification and anomaly flags) |
| `t_test`, `pearson`, `spearman`, `kendall`, … | Hypothesis tests and correlation measures |
| `vif` | Variance Inflation Factor |

### Verb component

| Verb | Description |
|------|-------------|
| `fit` | Fit a model, returning a STRUCT with coefficients and diagnostics |
| `fit_predict` | Fit and return predictions |
| `predict` | Predict from a fitted model struct (`predict(model, x)`) or from coefficients (`predict(x_new, coefficients, intercept)`, also `linear_predict`); see [Model tools](api/regression/model_tools.md) |
| `fit_by`, `fit_predict_by` | Table macro that runs a fit per group of a table |

Hypothesis tests are named after the test itself (`t_test_agg`, `chisq_test_agg`, `mann_whitney_u_agg`).

### Suffix component

| Suffix | When used |
|--------|-----------|
| `_agg` | DuckDB aggregate function (use with `GROUP BY`, or with `OVER (...)` where noted) |
| `_by` | Table macro taking a table name |
| (none) | Scalar function, or window aggregate (`*_fit_predict`) |

### Full examples

| Function | Type | Description |
|----------|------|-------------|
| `ols_fit(y, X[, options])` | Scalar | OLS fit on literal arrays (`X` is column-major: one inner array per feature) |
| `ols_fit_agg(y, [x1, x2][, options])` | Aggregate | OLS fit over the rows of a table or group |
| `ols_fit_predict(y, [x1, x2][, options]) OVER (ORDER BY t ROWS BETWEEN ... AND CURRENT ROW)` | Window aggregate | Fits on the window frame and returns `{yhat, yhat_lower, yhat_upper}` for the last row of the frame |
| `ols_fit_predict_agg(y, [x1, x2][, options])` | Aggregate | Fits on rows with non-NULL `y`; returns a LIST of per-row predictions |
| `ols_fit_predict_by('table', group_col, y_col, [x1, x2][, options])` | Table macro | Per-group fit + predict, one output row per input row |
| `wls_fit_agg(y, [x1, x2], weight[, options])` | Aggregate | Weighted fit; the weight is a positional argument |
| `theil_sen_fit_agg(y, [x])` | Aggregate | Theil-Sen fit |
| `poisson_fit_agg(y, [x1, x2][, options])` | Aggregate | Poisson GLM fit |
| `t_test_agg(value, group[, options])` | Aggregate | Two-sample t-test |
| `vif(X)` | Scalar | Variance Inflation Factors |
| `nnls_fit_agg(y, [x1, x2])` | Aggregate | Non-Negative Least Squares fit |

---

## 2. Option-Map Keys

Options are passed as a constant MAP or STRUCT literal as the last argument:

```sql
SELECT (ols_fit_agg(y, [x], {'fit_intercept': true, 'compute_inference': true})).p_values
FROM (VALUES (1.0, 1.0), (2.1, 2.0), (2.9, 3.0), (4.2, 4.0)) t(y, x);
```

### Key convention

Option keys are `snake_case` and case-insensitive. Keys that a function does not support raise an `InvalidInputException` at bind time instead of being silently ignored:

```sql skip
-- Raises: "unknown option 'intercept_mode'; valid keys: ..."
SELECT ols_fit_agg(y, [x], {'intercept_mode': true}) FROM tbl;
```

### Common option keys

The table lists the most common keys; each reference page in [docs/api/](api/) lists exactly the keys that function reads.

| Key (aliases) | Type | Default | Applies to | Description |
|-----|------|---------|-----------|-------------|
| `fit_intercept` (`intercept`) | BOOLEAN | `true` | All regression | Fit a constant intercept term |
| `compute_inference` (`inference`) | BOOLEAN | `false` | OLS, Ridge, WLS, Huber, RANSAC, Theil-Sen, GLMs, ALM, AFT, GLMM | Add `std_errors`, `t_values`/`z_values`, `p_values`, `ci_lower`, `ci_upper` to the result |
| `confidence_level` (`confidence`) | DOUBLE | `0.95` | Functions with intervals | Confidence level for intervals |
| `alpha` (`lambda`) | DOUBLE | `1.0` (Ridge, Elastic Net), `0.0` (LARS) | Ridge, Elastic Net, LARS | Regularization strength; `alpha = 0` gives the unpenalized fit |
| `l1_ratio` | DOUBLE | `0.5` | Elastic Net | Mix of L1 vs L2, in [0, 1] |
| `lambda_scaling` | VARCHAR | `'raw'` | Ridge, Elastic Net | `'raw'` or `'glmnet'` (divide penalty by n) |
| `solver` | VARCHAR | `'svd'` | OLS, WLS, Ridge | `'qr'`, `'svd'` or `'cholesky'` |
| `hc_type` | VARCHAR | `'none'` | OLS, WLS | Heteroscedasticity-consistent SEs: `'none'`, `'hc0'`, `'hc1'`, `'hc2'`, `'hc3'` |
| `max_iterations` (`max_iter`) | INTEGER | function-specific | Iterative solvers | Maximum iterations |
| `tolerance` (`tol`) | DOUBLE | function-specific | Iterative solvers | Convergence tolerance |
| `epsilon` | DOUBLE | `1.35` | Huber | Huber threshold |
| `max_trials` | INTEGER | `100` | RANSAC | Maximum RANSAC trials |
| `residual_threshold` | DOUBLE | MAD of `y` | RANSAC | Inlier threshold |
| `min_samples` | INTEGER | `n_features + 1` | RANSAC | Sample size per trial |
| `random_state` (`seed`) | INTEGER | — | RANSAC, Theil-Sen | Seed for reproducible subsampling |
| `forgetting_factor` | DOUBLE | `1.0` | RLS | Exponential forgetting factor |
| `link` (`poisson_link`) | VARCHAR | `'log'` | Poisson | `'log'`, `'identity'`, `'sqrt'` |
| `binomial_link` | VARCHAR | `'logit'` | Binomial | `'logit'`, `'probit'`, `'cloglog'` |
| `power` (`tweedie_power`) | DOUBLE | `1.5` | Tweedie | Tweedie variance power |
| `threshold` | DOUBLE | `0.5` | Logistic | Classification threshold used for `accuracy` |
| `glm_lambda` | DOUBLE | `0.0` | GLMs | L2 penalty on GLM coefficients |
| `prior`, `feature_names` | MAP / LIST | — | GLMs | Coefficient priors, see [api/glm/priors.md](api/glm/priors.md) |
| `family` | VARCHAR | `'gaussian'` | GLMM only | `'gaussian'`, `'poisson'`, `'binomial'`, `'negbinomial'`, `'gamma'`, `'tweedie'` |
| `distribution` (`dist`) | VARCHAR | model-specific | ALM, AFT | Error / survival distribution |
| `null_policy` | VARCHAR | `'drop'` | `*_fit_predict`, `*_fit_predict_agg` | `'drop'` or `'drop_y_zero_x'`, see [NULL_SEMANTICS.md](NULL_SEMANTICS.md) |

There is no `family` key for the single-family GLM aggregates: the family is chosen by the function (`poisson_fit_agg`, `binomial_fit_agg`, `negbinom_fit_agg`, `gamma_fit_agg`, `tweedie_fit_agg`, `logistic_fit_agg`). WLS weights are a positional argument (`wls_fit_agg(y, x, weight)`), and the `wls_fit_predict_by` macro takes a `weight_col` argument; neither is an option key. Prediction intervals from the fit-predict functions always use `confidence_level`; there is no `interval_type` option.

Option values outside their valid range raise `InvalidInputException`.

---

## 3. Return-Struct Field Names

Result structs use `snake_case` field names. The **standard field set** for linear regression families (OLS, Ridge, WLS, Huber, RANSAC, Theil-Sen) is:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Fitted coefficients (excluding intercept) |
| `intercept` | DOUBLE | Intercept term (0 if `fit_intercept: false`) |
| `r_squared` | DOUBLE | Coefficient of determination R² |
| `adj_r_squared` | DOUBLE | Adjusted R² |
| `residual_std_error` | DOUBLE | Residual standard error |
| `n_observations` | BIGINT | Number of observations used |
| `n_features` | BIGINT | Number of features (predictors) |
| `std_errors` | DOUBLE[] | Standard errors (NULL without `compute_inference: true`) |
| `t_values` | DOUBLE[] | t-statistics (NULL without `compute_inference: true`) |
| `p_values` | DOUBLE[] | Two-sided p-values (NULL without `compute_inference: true`) |
| `ci_lower`, `ci_upper` | DOUBLE[] | Deprecated aliases of `conf_low`, `conf_high` |
| `f_statistic`, `f_pvalue` | DOUBLE | Overall F-test (NULL without `compute_inference: true`) |
| `conf_low`, `conf_high` | DOUBLE[] | Confidence-interval bounds (NULL without `compute_inference: true`) |
| `conf_level` | DOUBLE | Confidence level of the intervals |
| `intercept_std_error`, `intercept_statistic`, `intercept_p_value`, `intercept_conf_low`, `intercept_conf_high` | DOUBLE | The intercept's inference |
| `log_likelihood`, `aic`, `bic` | DOUBLE | As R's `logLik`, `AIC`, `BIC`; NULL for Huber, RANSAC, Theil-Sen (no likelihood) |
| `model_type`, `family`, `link` | VARCHAR | What was fitted, e.g. `'ols'`, `'gaussian'`, `'identity'`; `family` is NULL without a likelihood |

The shape of the struct does not depend on the options (#152): fields that were
not computed are NULL. Elastic Net, LARS and RLS have no coefficient inference
(their inference fields are always NULL, and they have no `ci_lower`,
`ci_upper`, `f_statistic`, `f_pvalue`). Robust models add their own fields (`scale`, `n_outliers` for Huber; `residual_threshold`, `n_inliers`, `n_trials` for RANSAC). Coefficients of constant (zero-variance) or aliased columns are reported as `NaN`, see [NULL_SEMANTICS.md](NULL_SEMANTICS.md).

### Per-family exceptions (intentional — do NOT force z → t)

**GLM families (Poisson, Binomial, Logistic, Negative Binomial, Gamma, Tweedie):**

GLMs use `z_values` instead of `t_values` because the Wald statistic under GLM asymptotic theory follows a standard-normal distribution.

| Field | Type | Description |
|-------|------|-------------|
| `coefficients`, `intercept` | DOUBLE[], DOUBLE | Coefficients on the link scale |
| `deviance` | DOUBLE | Residual deviance |
| `null_deviance` | DOUBLE | Null-model deviance |
| `pseudo_r_squared` | DOUBLE | Deviance-based pseudo R² (`1 - deviance / null_deviance`) |
| `aic` | DOUBLE | Akaike Information Criterion |
| `dispersion` | DOUBLE | Dispersion estimate (Negative Binomial: the overdispersion `alpha`) |
| `n_observations`, `n_features` | BIGINT | Counts |
| `iterations` | INTEGER | IRLS iterations used |
| `converged` | BOOLEAN | Whether IRLS converged |
| `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper` | DOUBLE[] | NULL without `compute_inference: true` (`ci_lower`/`ci_upper` are deprecated aliases of `conf_low`/`conf_high`) |
| `family`, `link` | VARCHAR | e.g. `'poisson'`, `'log'` |
| shared fields | | `conf_low`, `conf_high`, `conf_level`, `intercept_*` inference, `log_likelihood`, `bic`, `model_type` (`'glm'`), as in the linear models |

GLM results do not include `r_squared`. `logistic_fit_agg` replaces `dispersion` with `accuracy` and `threshold`.

**AFT survival models:**

| Field | Type | Description |
|-------|------|-------------|
| `scale` | DOUBLE | Scale parameter |
| `log_likelihood`, `null_log_likelihood` | DOUBLE | Log-likelihood of the fitted and intercept-only model |
| `aic`, `bic` | DOUBLE | Information criteria |
| `n_observations`, `n_events`, `n_censored`, `n_features` | BIGINT | Counts |
| `iterations`, `converged` | INTEGER, BOOLEAN | Optimizer status |
| `z_values` (and `std_errors`, `p_values`, `ci_lower`, `ci_upper`, `intercept_std_error`, `log_scale_std_error`) | | NULL without `compute_inference: true` |
| shared fields | | `conf_low`, `conf_high`, `conf_level`, the other `intercept_*` inference, `model_type` (`'aft'`), `family` (the distribution), `link` (`'log'`) |

**ALM (Augmented Linear Model):**

ALM returns `coefficients`, `intercept`, `log_likelihood`, `aic`, `bic`, `scale`, `n_observations`, `n_features`, `iterations`, `std_errors`, `t_values`, `p_values`, `ci_lower`, `ci_upper` (NULL without `compute_inference: true`) and the shared fields (`model_type` `'alm'`, `family` = the distribution). It has no `r_squared`.

---

## 4. Error Messages

Errors always name the function:

```
{function_name}: {problem}
```

### Exception taxonomy

| Exception class | Raised when |
|----------------|------------|
| `InvalidInputException` | User data, shape or option problems: dimension mismatch, all-non-finite input, unsupported option key, option value out of range |
| `InternalException` | Numerical failures: singular matrix, convergence failure, allocation failure |

Aggregates over too few rows to fit (for example fewer than `n_features + 1` rows in a group) return `NULL` for that group rather than raising, so one degenerate group does not abort a `GROUP BY` query.

### Unsupported option keys

```sql skip
-- Raises: "unknown option 'typo_key'; valid keys: ..."
SELECT ols_fit_agg(y, [x], {'typo_key': true}) FROM tbl;
```

### Window frames

A `*_fit_predict` window function fits on the rows of the window frame and predicts for the **last row of the frame**. Use frames that end at `CURRENT ROW` over a unique ordering (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` or `ROWS BETWEEN k PRECEDING AND CURRENT ROW`). Frames ending before the current row (`... AND 1 PRECEDING`) do not produce a one-step-ahead forecast, `OVER (PARTITION BY g)` without `ORDER BY` gives every row the same prediction, and `RANGE` frames with ties are not supported. For one prediction per row of a whole group, use `*_fit_predict_agg` or `*_fit_predict_by`.

When the frame has too few rows to fit, the function returns `NULL` for that row rather than raising an error: the first rows of an expanding window have insufficient data.

---

## 5. Breaking Changes in v0.10.0

See [MIGRATION.md](MIGRATION.md) for the complete list.

### Dropped `anofox_stats_` prefix

Up to v0.9.x every function was registered both with and without the `anofox_stats_` prefix. From v0.10.0 only the unprefixed names exist.

**Migration:** remove the `anofox_stats_` prefix from every function call.

```sql skip
-- Before (v0.9.x):
SELECT anofox_stats_ols_fit_agg(y, [x1, x2]) FROM tbl;

-- After (v0.10.0+):
SELECT ols_fit_agg(y, [x1, x2]) FROM tbl;
```

### `theilsen` renamed to `theil_sen`

```sql skip
-- Before (v0.9.x):
SELECT theilsen_fit_agg(y, [x]) FROM tbl;

-- After (v0.10.0+):
SELECT theil_sen_fit_agg(y, [x]) FROM tbl;
```

### Use `.r_squared`, not `.r2`

The return-struct field is named `r_squared`; `.r2` is not a valid field path.

```sql
SELECT (ols_fit([1.0, 2.0, 3.1, 3.9], [[1.0, 2.0, 3.0, 4.0]])).r_squared;
```

### No deprecated aliases

No backward-compatibility aliases are provided for the removed names.

---

## 6. Documentation Rules

`scripts/validate_docs_sql.py` executes every fenced `sql` block in `README.md`, `guides/*.md`, `docs/*.md` and `docs/api/**/*.md` against the built extension (blocks marked `sql skip` are excluded). Examples must:

1. Use unprefixed function names.
2. Use `r_squared` (not `r2`) for the coefficient of determination.
3. Use `theil_sen_*` (not `theilsen_*`).
4. Use `z_values` for GLM and AFT results (not `t_values`).
5. Pass only option keys the function supports.
