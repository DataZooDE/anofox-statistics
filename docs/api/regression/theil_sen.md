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
| `random_state` (alias `seed`) | INTEGER | `0` | Seed used when subsets are sampled and for the bootstrap resamples |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add coefficient confidence intervals, see [Inference](#inference) |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Confidence level for intervals |
| `ci_method` | VARCHAR | `'exact'` for one feature, `'bootstrap'` otherwise | `'exact'`: Sen's distribution-free interval (one feature with an intercept only); `'bootstrap'`: pairs bootstrap |
| `n_bootstrap` (alias `n_boot`) | INTEGER | `999` | Bootstrap resamples, at least 2 |

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

## Inference

With `compute_inference = true` the fit carries confidence intervals for the
coefficients, computed in one of two ways (`ci_method`):

- **`'exact'`** (the default with one feature and an intercept): the
  distribution-free interval of Sen (1968), the interval R's
  `trend::sens.slope` reports. The `N` pairwise slopes are sorted and the
  bounds are the slopes at ranks `round((N − C)/2)` and `round((N + C)/2 + 1)`,
  with `C = z_{(1+level)/2} · sqrt(Var S)` and `Var S` the null variance of
  Kendall's score `S` of `(x, y)`, corrected for ties in `x` and in `y`.
  `t_values` holds Kendall's continuity-corrected `z = sign(S)(|S| − 1) / sqrt(Var S)`
  and `p_values` its two-sided normal p-value for slope = 0 (R `trend::sens.slope`;
  with ties in `x`, `cor.test(x, y, method = "kendall", continuity = TRUE)`).
  The interval has no standard error (`std_errors` is NaN) and there is none for
  the intercept. With very few rows the rank falls outside `1..N` and the
  bound is NaN (with 4 rows at 95%, for example).
- **`'bootstrap'`** (the default otherwise, and for the intercept): the pairs
  bootstrap. `n_bootstrap` resamples of the rows are refitted with the same
  options; `std_errors` and `intercept_std_error` are the bootstrap standard
  deviations and the intervals are percentile intervals (R `quantile` type 7).
  The result is deterministic for a given `seed` and does not depend on the
  row order or the number of threads. Resamples that cannot be fitted (e.g.
  all `x` equal) are skipped. There is no test statistic: `t_values` and
  `p_values` are NaN. The bootstrap refits the model `n_bootstrap` times, so it
  costs about that many fits; lower `n_bootstrap` for large groups.

Theil-Sen has no F-test: `f_statistic` and `f_pvalue` are NaN. A coverage
simulation (400 data sets, n = 40, Laplace noise, 90% level) gave 88.5% for the
exact interval, 88.5% / 87.0% for the one-feature bootstrap slope / intercept
and 91.0% / 91.2% for a two-feature bootstrap.

```sql
-- Sen's exact interval and Kendall's test for the slope
SELECT
    grp,
    round((m).conf_low[1], 3) AS slope_low,
    round((m).conf_high[1], 3) AS slope_high,
    round((m).p_values[1], 6) AS p_value
FROM (SELECT grp, theil_sen_fit_agg(y, [x], {'compute_inference': true}) AS m FROM theil_sen_demo GROUP BY grp)
ORDER BY grp;

-- Bootstrap standard errors and intervals, also for the intercept
SELECT
    round((m).std_errors[1], 3) AS slope_se,
    round((m).intercept_conf_low, 3) AS intercept_low,
    round((m).intercept_conf_high, 3) AS intercept_high
FROM (
    SELECT theil_sen_fit_agg(y, [x], {'compute_inference': true, 'ci_method': 'bootstrap', 'n_bootstrap': 199, 'seed': 1}) AS m
    FROM theil_sen_demo
);
```

`theil_sen_tidy_by` computes the intervals by default and carries them into
the `terms` rows (`std_error`, `statistic`, `p_value`, `conf_low`, `conf_high`).

**Behaviour change (#222):** before, `compute_inference` returned these
fields but left them empty (NaN / NULL): the solver had no inference. See the
[changelog](../../../CHANGELOG.md).

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

Like every model STRUCT the result also has the inference fields
`std_errors`, `t_values`, `p_values`, `conf_low` / `conf_high` (aliases
`ci_lower` / `ci_upper`), `conf_level`, `f_statistic`, `f_pvalue` and the
`intercept_*` fields, NULL without `compute_inference`; see
[Inference](#inference) for what each method fills.

## NULL handling

The aggregate skips rows where `y` or `x` is NULL, or where `x` contains a NULL
element. `theil_sen_fit` drops positions with NaN or infinite values. In the
fit-predict forms, rows with NULL `y` are not used for fitting but still get a
prediction.

## See Also

- [Huber](huber.md) - M-estimator
- [RANSAC](ransac.md) - Consensus-based robust regression
- [Quantile](quantile.md) - Median regression
