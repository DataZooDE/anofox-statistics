# Diagnostic Functions

Model diagnostics: multicollinearity (VIF, condition indices), separation of
binary and count responses, information criteria (AIC/BIC),
residual normality (Jarque-Bera) and residual analysis. Most come in a scalar
form that takes arrays and an aggregate form that takes one row at a time.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `vif` | Scalar | Variance inflation factors from feature arrays |
| `vif_agg` | Aggregate | Variance inflation factors from rows |
| `aic` | Scalar | Akaike information criterion from RSS |
| `bic` | Scalar | Bayesian information criterion from RSS |
| `jarque_bera` | Scalar | Jarque-Bera normality test on an array |
| `jarque_bera_agg` | Aggregate | Jarque-Bera normality test on rows |
| `residuals_diagnostics` | Scalar | Raw, standardized and studentized residuals, leverage, Cook's distance |
| `residuals_diagnostics_agg` | Aggregate | Same, from rows |
| `condition_diagnostic_agg` | Aggregate | Belsley-Kuh-Welsch condition indices and variance-decomposition proportions |
| `separation_check_agg` | Aggregate | Per-predictor separation check of a binary response |
| `count_sparsity_agg` | Aggregate | All-zero responses on an active indicator (count response) |
| `detect_separation_agg` | Aggregate | Exact separation check (does the logistic MLE exist?) |

## Variance Inflation Factor

### vif

```text
vif(x DOUBLE[][]) -> DOUBLE[]
```

`x` holds one inner list per feature (feature-major). Returns one VIF per
feature.

```sql
SELECT vif([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [2.0, 4.1, 6.2, 7.9, 10.1],   -- almost 2 * the first feature
    [5.0, 3.0, 4.0, 1.0, 2.0]
]) AS vif_values;
```

### vif_agg

```text
vif_agg(x DOUBLE[]) -> DOUBLE[]
```

```sql
CREATE OR REPLACE TABLE diag_demo AS
SELECT
    i AS id,
    i::DOUBLE AS x1,
    2.0 * i + (i % 3) * 0.1 AS x2,
    ((i * 7) % 5)::DOUBLE AS x3,
    1.0 + 0.5 * i + ((i * 7) % 5) + 0.3 * sin(i) AS y
FROM range(1, 31) t(i);

SELECT vif_agg([x1, x2, x3]) AS vif_values FROM diag_demo;
```

| VIF | Interpretation |
|-----|----------------|
| 1 | No correlation with the other features |
| 1-5 | Moderate correlation |
| > 5 | High correlation (warning) |
| > 10 | Very high correlation (problematic) |

## Condition Indices

### condition_diagnostic_agg

```text
condition_diagnostic_agg(x DOUBLE[] [, options]) -> STRUCT
```

Belsley-Kuh-Welsch collinearity diagnostics of the design `[1, x]` (R's
`perturb::colldiag`, `olsrr::ols_eigen_cindex`). Every column of the design,
the intercept included, is scaled to unit length; the SVD `X = U D V'` of that
design gives the condition indices `d_max / d_k` and the variance-decomposition
proportions `pi_kj = (v_jk^2 / d_k^2) / sum_k (v_jk^2 / d_k^2)`: the share of
the variance of coefficient `j` that is associated with component `k`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fit_intercept` | BOOLEAN | true | Add the intercept column to the design |

| Field | Type | Description |
|-------|------|-------------|
| `condition_number` | DOUBLE | `kappa(X, exact = TRUE)` of the unscaled design (depends on units) |
| `condition_number_xtx` | DOUBLE | Condition number of X'X (`condition_number^2`) |
| `scaled_condition_number` | DOUBLE | Largest condition index of the unit-length design |
| `singular_values` | DOUBLE[] | Singular values of the unit-length design, descending |
| `condition_indices` | DOUBLE[] | `d_max / d_k`, one per component, increasing |
| `variance_proportions` | DOUBLE[][] | One list per component (aligned with `condition_indices`), one proportion per coefficient (intercept first) |
| `severity` | VARCHAR | `well_conditioned` (< 30), `moderate` (< 100), `high` (< 1000) or `severe`, from `scaled_condition_number` |
| `warning` | VARCHAR | Advice when the severity is not `well_conditioned`, else NULL |
| `n_observations` | BIGINT | Rows used |

A row is skipped when the list, one of its elements, or a value is NULL or not
finite. Fewer than two rows, or an empty `x`, give NULL. An exactly collinear
design has an infinite condition number.

```sql
SELECT
    r.scaled_condition_number,
    r.severity,
    r.condition_indices,
    r.variance_proportions[len(r.variance_proportions)] AS last_component
FROM (SELECT condition_diagnostic_agg([x1, x2, x3]) AS r FROM diag_demo) s;
```

Two or more coefficients with a proportion above 0.5 on a component whose
condition index exceeds 30 are involved in a near dependency (here `x1` and
`x2`, since `x2` is almost `2 * x1`).

## Separation

Separation makes the maximum-likelihood estimate of a logistic (or Poisson)
model diverge: a linear combination of predictors predicts the response
perfectly. Check for it before reading the coefficients of
[logistic_fit_agg](../glm/logistic.md), or use a penalized fit.

### detect_separation_agg

```text
detect_separation_agg(y DOUBLE, x DOUBLE[] [, options]) -> STRUCT
```

The exact check of Konis (2007), as R's `detectseparation::detect_separation`:
a linear program that finds a direction `b` with `(2y - 1) x'b >= 0` for every
row. `y` must be 0 or 1. It sees separation by a combination of predictors.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fit_intercept` | BOOLEAN | true | Add the intercept column to the design |

| Field | Type | Description |
|-------|------|-------------|
| `separated` | BOOLEAN | The classes are (quasi-)completely separated: the logistic MLE does not exist |
| `infinite` | DOUBLE[] | `inf` / `-inf` for each coefficient that diverges, `0` otherwise (intercept first), as `detectseparation`'s `coefficients` |
| `direction` | DOUBLE[] | The separating direction found (`|b_j| <= 1`, all zero without separation) |
| `n_observations` | BIGINT | Rows used |

```sql
CREATE OR REPLACE TABLE sep_demo AS
SELECT * FROM (VALUES
    (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (3.0, 1.0, 0.0), (4.0, 1.0, 0.0),
    (1.0, 4.0, 0.0), (2.0, 4.0, 1.0), (3.0, 4.0, 1.0), (4.0, 4.0, 1.0),
    (2.0, 2.0, 0.0), (3.0, 3.0, 1.0), (1.0, 3.0, 0.0), (4.0, 2.0, 1.0)
) t(x1, x2, y);

-- y = 1 exactly when x1 + x2 > 5: separated by the combination.
SELECT detect_separation_agg(y, [x1, x2]) AS sep FROM sep_demo;
```

### separation_check_agg

```text
separation_check_agg(y DOUBLE, x DOUBLE[]) -> STRUCT
```

A per-predictor heuristic for a binary (0/1) `y`: a predictor shows
`complete` separation when a threshold on it splits the classes, `quasi`
separation when at most two observations overlap, and a `monotonic_response`
when every repeated value of it has a single class. It does not see separation
by a combination of predictors (use `detect_separation_agg`).

| Field | Type | Description |
|-------|------|-------------|
| `has_separation` | BOOLEAN | Some predictor is flagged |
| `separated_predictors` | BIGINT[] | 1-based positions in `x` of the flagged predictors |
| `separation_types` | VARCHAR[] | `complete`, `quasi` or `monotonic_response`, aligned with `separated_predictors` |
| `warning` | VARCHAR | Advice when a predictor is flagged, else NULL |
| `n_observations` | BIGINT | Rows used |

```sql
SELECT separation_check_agg(y, [x1, x2]) AS chk FROM sep_demo;
```

### count_sparsity_agg

```text
count_sparsity_agg(y DOUBLE, x DOUBLE[]) -> STRUCT
```

The count-data analogue for Poisson / negative binomial models: when more than
90% of the non-negative counts `y` are zero, an indicator predictor (at most
three distinct values) whose active rows all have a zero count drives its
coefficient to `-inf`. Returns the same fields as `separation_check_agg`.

```sql
SELECT count_sparsity_agg(y, [z, ind]) AS chk
FROM (
    SELECT CASE WHEN i = 0 THEN 4.0 ELSE 0.0 END AS y,
           CASE WHEN i >= 15 THEN 1.0 ELSE 0.0 END AS ind,
           (i % 7)::DOUBLE AS z
    FROM range(20) t(i)
) s;
```

All three skip rows with a NULL or non-finite `y`, a NULL list or a NULL
element, and return NULL on no rows or an empty `x`. A `y` outside the allowed
values is an error.

## Information Criteria

### aic

```text
aic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `rss` | `DOUBLE` | Residual sum of squares |
| `n` | `BIGINT` | Number of observations |
| `k` | `BIGINT` | Number of parameters, including the intercept |

AIC = n · ln(RSS / n) + 2k. Lower is better.

```sql
SELECT aic(100.0, 50, 3) AS aic_value;
```

### bic

```text
bic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

BIC = n · ln(RSS / n) + k · ln(n). Lower is better; penalizes extra parameters
more than AIC once n > 7.

```sql
SELECT bic(100.0, 50, 3) AS bic_value;
```

| Criterion | Best for |
|-----------|----------|
| AIC | Prediction, when the true model may not be among the candidates |
| BIC | Model identification; consistent as n grows |

## Normality

### jarque_bera

```text
jarque_bera(values DOUBLE[]) -> STRUCT
```

```sql
SELECT jarque_bera([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 3.0, 2.0, 8.0, 1.0]) AS jb;
```

### jarque_bera_agg

```text
jarque_bera_agg(value DOUBLE) -> STRUCT
```

```sql
-- Normality of OLS residuals
WITH fit AS (SELECT ols_fit_agg(y, [x1, x3]) AS m FROM diag_demo)
SELECT (jarque_bera_agg(y - (m.intercept + list_dot_product(m.coefficients, [x1, x3])))).p_value AS p_value
FROM diag_demo, fit;
```

**Returns** (both forms):

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `DOUBLE` | Jarque-Bera statistic |
| `p_value` | `DOUBLE` | p-value (chi-squared, 2 df) |
| `skewness` | `DOUBLE` | Sample skewness |
| `kurtosis` | `DOUBLE` | Sample excess kurtosis (0 for a normal distribution) |
| `n` | `BIGINT` | Number of values used |

NULL values are ignored by the aggregate.

## Residual Analysis

### residuals_diagnostics

```text
residuals_diagnostics(y DOUBLE[], y_hat DOUBLE[]) -> STRUCT
residuals_diagnostics(y DOUBLE[], y_hat DOUBLE[], x DOUBLE[][],
                      residual_std_error DOUBLE, include_studentized BOOLEAN) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Observed values |
| `y_hat` | `DOUBLE[]` | Fitted values |
| `x` | `DOUBLE[][]` | Feature columns (feature-major); needed for leverage |
| `residual_std_error` | `DOUBLE` | Residual standard error of the model; needed for standardized residuals |
| `include_studentized` | `BOOLEAN` | Also compute studentized residuals |

With only `y` and `y_hat`, just the raw residuals are filled; the other fields
are NULL.

```sql
SELECT residuals_diagnostics(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.1, 1.9, 3.2, 3.8, 5.1],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.2,
    true
) AS diagnostics;
```

### residuals_diagnostics_agg

```text
residuals_diagnostics_agg(y DOUBLE, y_hat DOUBLE [, x DOUBLE[]]) -> STRUCT
```

The aggregate computes raw residuals and, when `x` is given, standardized and
studentized residuals, leverage and Cook's distance (σ is estimated from the
residuals as `sqrt(RSS / (n - k - 1))`).

```sql
SELECT residuals_diagnostics_agg(y, y_hat, [x]) AS diagnostics
FROM (VALUES (1.0, 1.1, 1.0), (2.0, 1.9, 2.0), (3.0, 3.2, 3.0),
             (4.0, 3.8, 4.0), (5.0, 5.1, 5.0)) t(y, y_hat, x);
```

**Returns** (both forms):

| Field | Type | Description |
|-------|------|-------------|
| `raw` | `DOUBLE[]` | Raw residuals `y - y_hat` |
| `standardized` | `DOUBLE[]` | `raw / residual_std_error` (NULL unless `residual_std_error` is given) |
| `studentized` | `DOUBLE[]` | Internally studentized residuals `raw / (σ · √(1 - h))` (NULL unless requested) |
| `leverage` | `DOUBLE[]` | Diagonal of the hat matrix (NULL unless `x` is given) |
| `cooks_distance` | `DOUBLE[]` | Cook's distance `raw² / ((k + 1) σ²) · h / (1 - h)²`, as R's `cooks.distance()` (NULL unless `x` is given and leverage is computed) |

## Detecting problems

- **High leverage**: leverage > 2(k + 1) / n suggests an influential point;
  check its studentized residual.
- **Outliers**: |studentized residual| > 3.
- **Non-normal residuals**: small Jarque-Bera p-value.
- **Multicollinearity**: VIF > 5 (moderate) or > 10 (severe), or a condition
  index > 30 with two proportions > 0.5 (`condition_diagnostic_agg`); consider
  [Ridge](../regression/ridge.md) or dropping features.
- **Separation**: `detect_separation_agg(...).separated` before trusting a
  logistic fit.

## See Also

- [Time-series diagnostics](timeseries.md) - ACF/PACF, Ljung-Box, Durbin-Watson, Breusch-Pagan, White
- [OLS](../regression/ols.md) - Standard regression
- [Ridge](../regression/ridge.md) - Regularization for multicollinearity
- [Hypothesis tests](../statistics/hypothesis.md) - Statistical tests
