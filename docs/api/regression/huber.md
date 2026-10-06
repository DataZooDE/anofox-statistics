# Huber Regression

Huber M-estimator regression. Residuals smaller than `epsilon` times the
robust scale are treated quadratically (like OLS), larger ones linearly, so a
few outliers cannot dominate the fit. Fitted by iteratively reweighted least
squares with a small L2 penalty `alpha`.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `huber_fit` | Scalar | Fit on complete arrays in a single call |
| `huber_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `huber_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `huber_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `huber_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#huber_fit_predict_by) |

## huber_fit

**Signature:**

```text
huber_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
-- y = 2x + 1 with one gross outlier at x = 4
SELECT huber_fit(
    [3.0, 5.1, 6.9, 40.0, 11.0, 13.1, 14.9, 17.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
) AS fit;
```

## huber_fit_agg

**Signature:**

```text
huber_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE huber_demo AS
SELECT
    i AS id,
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    i::DOUBLE AS x,
    1.0 + 2.0 * i + sin(i) + CASE WHEN i % 10 = 0 THEN 60.0 ELSE 0.0 END AS y
FROM range(1, 41) t(i);

SELECT
    (ols_fit_agg(y, [x])).coefficients AS ols,
    (huber_fit_agg(y, [x])).coefficients AS huber,
    (huber_fit_agg(y, [x])).n_outliers AS n_outliers
FROM huber_demo;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `epsilon` | DOUBLE | `1.35` | Threshold (in units of the robust scale) between quadratic and linear loss; must be > 1. Smaller is more robust |
| `alpha` | DOUBLE | `0.0001` | L2 penalty on the coefficients |
| `max_iterations` (alias `max_iter`) | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` (alias `tol`) | DOUBLE | `1e-5` | Convergence tolerance |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add standard errors, t-values, p-values, confidence intervals and the F-test |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Confidence level for intervals |

The window, fit-predict aggregate and table-macro forms also accept
`null_policy` (see [Fit-predict aggregates](fit_predict_agg.md#null_policy));
`compute_inference` has no effect there. Option keys the function does not
support raise an error.

```sql
-- More aggressive down-weighting of outliers, with inference
SELECT (huber_fit_agg(y, [x], {'epsilon': 1.1, 'compute_inference': true})).p_values AS p_values
FROM huber_demo;
```

## huber_fit_predict / huber_fit_predict_agg / huber_fit_predict_by

```text
huber_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...) -> STRUCT(yhat, yhat_lower, yhat_upper)
huber_fit_predict_agg(y DOUBLE, x DOUBLE[] [, split_col VARCHAR] [, options MAP]) -> STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)[]
huber_fit_predict_by(source VARCHAR, group_col, y_col, x_cols [, options] [, split]) -> TABLE
```

```sql
-- One prediction per row, model fitted per group on rows with known y
SELECT * FROM huber_fit_predict_by('huber_demo', grp, y, [x]) LIMIT 5;
```

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | One coefficient per feature |
| `intercept` | `DOUBLE` | Intercept |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `adj_r_squared` | `DOUBLE` | Adjusted R² |
| `residual_std_error` | `DOUBLE` | Residual standard error |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |
| `scale` | `DOUBLE` | MAD-based robust scale estimate of the residuals |
| `n_outliers` | `BIGINT` | Observations with \|residual\| > `epsilon` × `scale` |

With `compute_inference = true` these fields are added: `std_errors`,
`t_values`, `p_values`, `ci_lower`, `ci_upper` (`DOUBLE[]`), `f_statistic` and
`f_pvalue` (`DOUBLE`).

## NULL handling

The aggregate skips rows where `y` or `x` is NULL, or where `x` contains a NULL
element. `huber_fit` drops positions with NaN or infinite values. In the
fit-predict forms, rows with NULL `y` are not used for fitting but still get a
prediction.

## See Also

- [RANSAC](ransac.md) - Consensus-based robust regression
- [Theil-Sen](theil_sen.md) - Median-of-slopes robust regression
- [OLS](ols.md) - Non-robust baseline
