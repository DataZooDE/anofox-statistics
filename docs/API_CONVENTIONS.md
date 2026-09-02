# API Conventions

**Applies to:** anofox-statistics v0.3.0 and later
**Status:** Authoritative — Phase 6 doc-SQL validation checks examples against this document

---

## 1. Function Naming Convention

### Pattern

```
{model}_{verb}[_{suffix}]
```

All functions are **unprefixed and uniform**. The `anofox_stats_` prefix that existed in pre-v0.3.0 versions has been dropped (breaking change — see §5).

### Model component

The statistical model or family name, using snake_case:

| Model component | Description |
|----------------|-------------|
| `ols` | Ordinary Least Squares |
| `ridge` | Ridge (L2-penalized) regression |
| `elasticnet` | Elastic-net (L1+L2) regression |
| `wls` | Weighted Least Squares |
| `huber` | Huber robust regression |
| `ransac` | RANSAC robust regression |
| `rls` | Recursive Least Squares |
| `bls` | Bounded Least Squares |
| `nnls` | Non-Negative Least Squares |
| `theil_sen` | Theil-Sen robust regression (note: was `theilsen` pre-v0.3.0) |
| `glm` | Generalized Linear Model |
| `poisson` | GLM with Poisson family |
| `logistic` | GLM with Binomial/Logistic family |
| `gamma` | GLM with Gamma family |
| `nb` | GLM with Negative Binomial family |
| `aft` | Accelerated Failure Time (survival) |
| `aid` | Anomaly / Influence Detection |
| `t_test` | Student's t-test |
| `pearson` | Pearson correlation |
| `spearman` | Spearman correlation |
| `kendall` | Kendall correlation |
| `distance_cor` | Distance correlation |
| `icc` | Intraclass correlation |
| `chisq_test` | Chi-squared test |
| `chisq_gof` | Chi-squared goodness-of-fit |
| `fisher_exact` | Fisher's exact test |
| `g_test` | G-test (likelihood ratio) |
| `mcnemar` | McNemar's test |
| `tost` | Two One-Sided Tests (equivalence) |
| `vif` | Variance Inflation Factor |

### Verb component

The verb describes what the function does:

| Verb | Description |
|------|-------------|
| `fit` | Fit a model, returning a STRUCT with coefficients and diagnostics |
| `fit_predict` | Fit and return predictions (in-sample or with new X) |
| `predict` | Predict from previously computed coefficients |
| `test` | Hypothesis test, returning a result STRUCT |

### Suffix component

| Suffix | When used |
|--------|-----------|
| `_agg` | DuckDB aggregate function (use with GROUP BY or OVER) |
| (none) | Scalar or table function |

### Full examples

| Function | Type | Description |
|----------|------|-------------|
| `ols_fit(y, X)` | Scalar | OLS fit on literal arrays |
| `ols_fit_agg(y, x_col)` | Aggregate | OLS fit across groups |
| `ols_fit_predict(y, X, X_new)` | Table | OLS fit + predict from table-function call |
| `ols_fit_predict_agg(y, x_col)` | Window aggregate | Rolling OLS predictions |
| `theil_sen_fit(y, X)` | Scalar | Theil-Sen fit on literal arrays |
| `theil_sen_fit_agg(y, x_col)` | Aggregate | Theil-Sen fit across groups |
| `poisson_fit_agg(y, x_col)` | Aggregate | GLM Poisson fit |
| `t_test_agg(x, y)` | Aggregate | Two-sample t-test |
| `vif(y, X)` | Scalar | Variance Inflation Factors |
| `bls_fit_agg(y, x_col)` | Aggregate | Bounded Least Squares fit |
| `nnls_fit_agg(y, x_col)` | Aggregate | Non-Negative Least Squares fit |

---

## 2. Option-Map Keys

Options are passed as a DuckDB `MAP` literal, e.g.:

```sql
SELECT ols_fit_agg(y, [x1, x2], {'fit_intercept': true, 'compute_inference': true}) FROM tbl;
```

### Key convention

All option keys are `snake_case` matching the Rust core. Unknown keys are rejected at bind time:

```sql
-- This raises: "unknown option 'intercept_mode'; valid keys: fit_intercept, ..."
SELECT ols_fit_agg(y, [x], {'intercept_mode': true}) FROM tbl;
```

### Common option keys

| Key | Type | Default | Applies to | Description |
|-----|------|---------|-----------|-------------|
| `fit_intercept` | BOOLEAN | `true` | All regression | Fit a constant intercept term |
| `intercept` | BOOLEAN | — | All regression | Accepted alias for `fit_intercept` |
| `compute_inference` | BOOLEAN | `false` | OLS, Ridge, WLS | Compute std errors, t-values, p-values |
| `confidence_level` | DOUBLE | `0.95` | All with CIs | Confidence level for intervals; must be in (0, 1) |
| `alpha` | DOUBLE | — | Ridge, Elastic-net | Regularization strength (L2 penalty); must be > 0 |
| `l1_ratio` | DOUBLE | `0.5` | Elastic-net | Mix of L1 vs L2; must be in [0, 1] |
| `lambda` | DOUBLE | — | Ridge | Alias for `alpha` |
| `max_iterations` | INTEGER | — | Iterative solvers | Maximum iterations |
| `tolerance` | DOUBLE | — | Iterative solvers | Convergence tolerance |
| `hc_type` | VARCHAR | `'HC3'` | OLS robust SEs | Heteroscedasticity-consistent SE type |
| `weight_col` | VARCHAR | — | WLS | Column name for observation weights |
| `huber_epsilon` | DOUBLE | `1.35` | Huber | Epsilon threshold |
| `max_trials` | INTEGER | `100` | RANSAC | Maximum RANSAC trials |
| `residual_threshold` | DOUBLE | — | RANSAC | Inlier threshold |
| `min_samples` | INTEGER | — | RANSAC | Minimum inlier sample size |
| `link` | VARCHAR | — | GLM | Link function override |
| `family` | VARCHAR | — | GLM | Distribution family |
| `distribution` | VARCHAR | — | AFT | Survival distribution |
| `interval_type` | VARCHAR | `'confidence'` | Prediction | `'confidence'` or `'prediction'` |

Option value ranges are enforced at bind time. Providing a value outside the documented range raises `InvalidInputException` immediately.

---

## 3. Return-Struct Field Names

Result structs use `snake_case` field names. The **standard field set** for regression families is:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Fitted coefficients (excluding intercept) |
| `intercept` | DOUBLE | Intercept term (or 0 if `fit_intercept: false`) |
| `std_errors` | DOUBLE[] | Standard errors of coefficients (when `compute_inference: true`) |
| `t_values` | DOUBLE[] | t-statistics for each coefficient |
| `p_values` | DOUBLE[] | Two-sided p-values |
| `r_squared` | DOUBLE | Coefficient of determination R² |
| `adj_r_squared` | DOUBLE | Adjusted R² |
| `residual_std_error` | DOUBLE | Residual standard error |
| `n_obs` | BIGINT | Number of observations used |
| `n_features` | BIGINT | Number of features (predictors) |
| `ci_lower` | DOUBLE[] | Lower confidence-interval bounds |
| `ci_upper` | DOUBLE[] | Upper confidence-interval bounds |

### Per-family exceptions (intentional — do NOT force z → t)

**GLM families (Poisson, Logistic, Gamma, Negative Binomial):**

GLM uses `z_values` instead of `t_values` because the Wald statistic under GLM asymptotic theory follows a standard-normal (z) distribution, not a t distribution. This is the correct statistical convention for these families.

| Field | Type | Description |
|-------|------|-------------|
| `z_values` | DOUBLE[] | Wald z-statistics (replaces `t_values`) |
| `log_likelihood` | DOUBLE | Log-likelihood at convergence |
| `deviance` | DOUBLE | Model deviance |
| `null_deviance` | DOUBLE | Null-model deviance |
| `aic` | DOUBLE | Akaike Information Criterion |
| `bic` | DOUBLE | Bayesian Information Criterion |
| `n_iterations` | INTEGER | Iterations to convergence |

Note: GLM does **not** include `r_squared` (not a meaningful statistic for non-Gaussian families).

**AFT survival models:**

AFT survival analysis uses `z_values` for Wald statistics (survival convention) and omits `r_squared`.

| Field | Type | Description |
|-------|------|-------------|
| `z_values` | DOUBLE[] | Wald z-statistics |
| `log_likelihood` | DOUBLE | Log-likelihood at convergence |
| `aic` | DOUBLE | Akaike Information Criterion |
| `scale` | DOUBLE | Scale parameter |

**ALM / Additive models:**

ALM (Additive Linear Models) uses a different core field set: omits `r_squared`, carries `log_likelihood`, `aic`, `bic`, and `scale`.

---

## 4. Error Messages

When a function receives invalid input, it throws an exception with the format:

```
{function_name}: {problem}; expected {shape} (got {actual})
```

### Exception taxonomy

| Exception class | Raised when |
|----------------|------------|
| `InvalidInputException` | User data/shape problems — dimension mismatch, insufficient rows (`n < n_features + 1`), all-non-finite input, constant/zero-variance column, unknown option key, option value out of range |
| `FunctionException` | Numerical failures — singular matrix (non-invertible), convergence failure, internal panic |

### Unknown option keys

Unknown option-map keys are rejected at bind time:

```sql
-- Raises: "unknown option 'typo_key'; valid keys: fit_intercept, compute_inference, ..."
SELECT ols_fit_agg(y, [x], {'typo_key': true}) FROM tbl;
```

### Degenerate window frames

When a `_fit_predict_agg` function is used with `OVER (... ROWS BETWEEN ...)` and the window frame has fewer than `n_features + 1` rows, the function returns `NULL` for that row (rather than raising an error). This is standard rolling-regression behavior — degenerate frames at the start of a partition simply have insufficient data to fit.

---

## 5. Breaking Changes in v0.3.0

### Dropped `anofox_stats_` prefix

All functions previously registered under the `anofox_stats_` prefix are now registered under unprefixed names only. There are no deprecated aliases.

**Migration:** Remove the `anofox_stats_` prefix from every function call.

```sql
-- Before (v0.2.x):
SELECT anofox_stats_ols_fit_agg(y, [x1, x2]) FROM tbl;

-- After (v0.3.0+):
SELECT ols_fit_agg(y, [x1, x2]) FROM tbl;
```

### `theilsen` renamed to `theil_sen`

The Theil-Sen estimator functions were previously named `theilsen_*`. They are now `theil_sen_*` (underscore inserted for consistency).

```sql
-- Before:
SELECT anofox_stats_theilsen_fit_agg(y, [x]) FROM tbl;

-- After:
SELECT theil_sen_fit_agg(y, [x]) FROM tbl;
```

### `.r2` field removed; use `.r_squared`

The return-struct field was always named `r_squared` in the C++ type builder; some older test examples used `.r2` which was not a valid field path. The correct field is `.r_squared`.

```sql
-- Correct:
SELECT (ols_fit([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]])).r_squared;
```

### No deprecated aliases

No backward-compatibility aliases are provided. All callers must update to the new names.

---

## 6. Validation Rules

### Phase 6 doc-SQL validation checks

When Phase 6's documentation-SQL validator runs, it checks every SQL example in `docs/` against the live extension. Examples must:

1. Use unprefixed function names (no `anofox_stats_` prefix).
2. Use `r_squared` (not `r2`) for the coefficient of determination.
3. Use `theil_sen_*` (not `theilsen_*`).
4. Use `z_values` for GLM and AFT results (not `t_values`).
5. Pass only known option keys (no typos silently ignored).
