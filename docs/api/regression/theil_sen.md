# Theil-Sen Regression

Theil-Sen is a robust, non-parametric estimator. With one feature the slope is
the median of all pairwise slopes `(y_j - y_i) / (x_j - x_i)`; with several
features it fits least squares on subsets of `n_subsamples` observations and
takes the spatial (geometric) median of the resulting coefficient vectors. Its
breakdown point is about 29%.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `theil_sen_fit` | Scalar | Fit on complete arrays in a single call |
| `theil_sen_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `theil_sen_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `theil_sen_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `theil_sen_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#theil_sen_fit_predict_by) |

## theil_sen_fit

**Signature:**

```text
theil_sen_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
-- One outlier barely moves the median-of-slopes estimate
SELECT theil_sen_fit(
    [3.0, 5.0, 7.0, 9.0, 30.0, 13.0, 15.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
) AS fit;
```

## theil_sen_fit_agg

**Signature:**

```text
theil_sen_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE theil_sen_demo AS
SELECT
    i AS id,
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    i::DOUBLE AS x,
    0.5 + 1.5 * i + cos(i) + CASE WHEN i % 8 = 0 THEN -40.0 ELSE 0.0 END AS y
FROM range(1, 41) t(i);

SELECT
    grp,
    (theil_sen_fit_agg(y, [x])).coefficients AS slope,
    (theil_sen_fit_agg(y, [x])).intercept AS intercept
FROM theil_sen_demo
GROUP BY grp
ORDER BY grp;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_subsamples` | INTEGER | `n_features + 1` (with intercept) | Observations per subset; larger values trade robustness for efficiency |
| `max_subpopulation` | INTEGER | `10000` | Maximum number of subsets; if there are more combinations, a random sample of this size is used. Must be > 0 |
| `max_iterations` (alias `max_iter`) | INTEGER | `300` | Maximum iterations of the spatial-median solver |
| `tolerance` (alias `tol`) | DOUBLE | `1e-3` | Convergence tolerance of the spatial-median solver |
| `random_state` (alias `seed`) | INTEGER | `0` | Seed used when subsets are sampled |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add standard errors, t-values, p-values, confidence intervals and the F-test |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Confidence level for intervals |

The window, fit-predict aggregate and table-macro forms also accept
`null_policy` (see [Fit-predict aggregates](fit_predict_agg.md#null_policy));
`compute_inference` has no effect there. Option keys the function does not
support raise an error.

## theil_sen_fit_predict / theil_sen_fit_predict_agg / theil_sen_fit_predict_by

```text
theil_sen_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...) -> STRUCT(yhat, yhat_lower, yhat_upper)
theil_sen_fit_predict_agg(y DOUBLE, x DOUBLE[] [, split_col VARCHAR] [, options MAP]) -> STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)[]
theil_sen_fit_predict_by(source VARCHAR, group_col, y_col, x_cols [, options] [, split] [, order_by]) -> TABLE
```

```sql
SELECT grp, x, y, round(yhat, 2) AS yhat, is_training
FROM theil_sen_fit_predict_by('theil_sen_demo', grp, y, [x])
LIMIT 5;
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

With `compute_inference = true` these fields are added: `std_errors`,
`t_values`, `p_values`, `ci_lower`, `ci_upper` (`DOUBLE[]`), `f_statistic` and
`f_pvalue` (`DOUBLE`).

## NULL handling

The aggregate skips rows where `y` or `x` is NULL, or where `x` contains a NULL
element. `theil_sen_fit` drops positions with NaN or infinite values. In the
fit-predict forms, rows with NULL `y` are not used for fitting but still get a
prediction.

## See Also

- [Huber](huber.md) - M-estimator
- [RANSAC](ransac.md) - Consensus-based robust regression
- [Quantile](quantile.md) - Median regression
