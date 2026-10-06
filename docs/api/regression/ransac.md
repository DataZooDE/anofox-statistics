# RANSAC Regression

RANSAC (RANdom SAmple Consensus) repeatedly fits a model on small random
subsets, counts how many observations lie within `residual_threshold` of each
candidate fit (the inliers), keeps the candidate with the largest consensus
set, and refits on its inliers. It tolerates a large share of gross outliers.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ransac_fit` | Scalar | Fit on complete arrays in a single call |
| `ransac_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `ransac_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `ransac_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `ransac_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#ransac_fit_predict_by) |

## ransac_fit

**Signature:**

```text
ransac_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
-- y = 2x + 1 with two gross outliers
SELECT ransac_fit(
    [3.0, 5.0, 7.0, 50.0, 11.0, 13.0, -20.0, 17.0, 19.0, 21.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
    {'residual_threshold': 1.0, 'random_state': 42}
) AS fit;
```

## ransac_fit_agg

**Signature:**

```text
ransac_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE ransac_demo AS
SELECT
    i AS id,
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    i::DOUBLE AS x,
    -- 20% of the rows are contaminated
    CASE WHEN i % 5 = 0 THEN 100.0 - i ELSE 1.0 + 2.0 * i + 0.1 * sin(i) END AS y
FROM range(1, 51) t(i);

SELECT
    (ransac_fit_agg(y, [x], {'random_state': 7})).coefficients AS coefficients,
    (ransac_fit_agg(y, [x], {'random_state': 7})).n_inliers AS n_inliers
FROM ransac_demo;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `residual_threshold` | DOUBLE | MAD of `y` | Maximum absolute residual for an observation to count as an inlier; must be finite and > 0 |
| `min_samples` | INTEGER | `n_features + 1` (with intercept) | Observations drawn per random subset |
| `max_trials` | INTEGER | `100` | Maximum number of random subsets |
| `stop_probability` | DOUBLE | `0.99` | Stop early once an outlier-free subset has been drawn with this probability |
| `stop_n_inliers` | INTEGER | unset | Stop early once a candidate has at least this many inliers |
| `random_state` (alias `seed`) | INTEGER | `0` | Seed for the subset sampler; fits are deterministic for a given seed |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add standard errors, t-values, p-values, confidence intervals and the F-test |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Confidence level for intervals |

The window, fit-predict aggregate and table-macro forms also accept
`null_policy` (see [Fit-predict aggregates](fit_predict_agg.md#null_policy));
`compute_inference` has no effect there. Option keys the function does not
support raise an error.

## ransac_fit_predict / ransac_fit_predict_agg / ransac_fit_predict_by

```text
ransac_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...) -> STRUCT(yhat, yhat_lower, yhat_upper)
ransac_fit_predict_agg(y DOUBLE, x DOUBLE[] [, split_col VARCHAR] [, options MAP]) -> STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)[]
ransac_fit_predict_by(source VARCHAR, group_col, y_col, x_cols [, options] [, split]) -> TABLE
```

```sql
SELECT grp, x, y, round(yhat, 2) AS yhat
FROM ransac_fit_predict_by('ransac_demo', grp, y, [x], {'random_state': 1})
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
| `n_observations` | `BIGINT` | Rows passed to the fit |
| `n_features` | `BIGINT` | Number of features |
| `residual_threshold` | `DOUBLE` | Inlier threshold actually used |
| `n_inliers` | `BIGINT` | Size of the final consensus set |
| `n_trials` | `BIGINT` | Random subsets evaluated |

With `compute_inference = true` these fields are added: `std_errors`,
`t_values`, `p_values`, `ci_lower`, `ci_upper` (`DOUBLE[]`), `f_statistic` and
`f_pvalue` (`DOUBLE`).

## NULL handling

The aggregate skips rows where `y` or `x` is NULL, or where `x` contains a NULL
element. `ransac_fit` drops positions with NaN or infinite values. In the
fit-predict forms, rows with NULL `y` are not used for fitting but still get a
prediction.

## See Also

- [Huber](huber.md) - M-estimator, mild outliers
- [Theil-Sen](theil_sen.md) - Median-of-slopes robust regression
- [OLS](ols.md) - Non-robust baseline
