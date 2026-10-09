# Table Macros

Table macros wrap the aggregate functions so that a per-group analysis is a
single `SELECT * FROM <macro>(...)`. The `*_fit_predict_by` macros fit one
model per group and return **one row per input row**, with every source column
passed through and the prediction columns appended.

| Macro | Wraps | Output |
|-------|-------|--------|
| `ols_fit_predict_by` | `ols_fit_predict_agg` | one row per input row |
| `ridge_fit_predict_by` | `ridge_fit_predict_agg` | one row per input row |
| `elasticnet_fit_predict_by` | `elasticnet_fit_predict_agg` | one row per input row |
| `wls_fit_predict_by` | `wls_fit_predict_agg` | one row per input row |
| `rls_fit_predict_by` | `rls_fit_predict_agg` | one row per input row |
| `huber_fit_predict_by` | `huber_fit_predict_agg` | one row per input row |
| `ransac_fit_predict_by` | `ransac_fit_predict_agg` | one row per input row |
| `theil_sen_fit_predict_by` | `theil_sen_fit_predict_agg` | one row per input row |
| `bls_fit_predict_by` | `bls_fit_predict_agg` | one row per input row |
| `alm_fit_predict_by` | `alm_fit_predict_agg` | one row per input row |
| `poisson_fit_predict_by` | `poisson_fit_predict_agg` | one row per input row |
| `binomial_fit_predict_by` | `binomial_fit_agg` + `predict` | one row per input row, no intervals |
| `logistic_fit_predict_by` | `logistic_fit_agg` + `predict` | one row per input row, no intervals |
| `negbinom_fit_predict_by` | `negbinom_fit_agg` + `predict` | one row per input row, no intervals |
| `gamma_fit_predict_by` | `gamma_fit_agg` + `predict` | one row per input row, no intervals |
| `tweedie_fit_predict_by` | `tweedie_fit_agg` + `predict` | one row per input row, no intervals |
| `pls_fit_predict_by` | `pls_fit_predict_agg` | one row per input row, no intervals |
| `quantile_fit_predict_by` | `quantile_fit_predict_agg` | one row per input row, no intervals |
| `isotonic_fit_predict_by` | `isotonic_fit_predict_agg` | one row per input row, no intervals |
| `glmm_fit_by` | `glmm_fit_agg` | one row per group (random effects) |
| `eb_shrink_by` | `eb_shrink_agg` | one row per input row (shrunken estimates) |
| `aid_by` | `aid_agg` | one row per group (demand classification) |
| `aid_anomaly_by` | `aid_anomaly_agg` | one row per input row (anomaly flags) |

## Common interface of the *_fit_predict_by macros

```text
<method>_fit_predict_by(
    source VARCHAR,          -- table or view name, as a string
    group_col,               -- column to group by
    y_col,                   -- response column; NULL marks a row to predict
    x_cols,                  -- feature columns as a list, e.g. [x1, x2]
    options := NULL,         -- optional MAP/STRUCT of model options
    split := NULL,           -- optional column with 'train' / other values
    order_by := NULL         -- optional column that orders the rows of each group
) -> TABLE
```

Some macros differ:

```text
wls_fit_predict_by(source, group_col, y_col, x_cols, weight_col, options := NULL, split := NULL, order_by := NULL)
isotonic_fit_predict_by(source, group_col, y_col, x_col, options := NULL, split := NULL, order_by := NULL)   -- single x column

-- GLM macros (fit the model per group, then predict(model, x) on every row); no order_by
binomial_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
logistic_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
negbinom_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
gamma_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
tweedie_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
```

`options` and `split` can be passed positionally or by name
(`options := {...}`, `split := split_col`); pass `order_by` by name
(`order_by := t`).

**`order_by`:** the macros built on the fit-predict aggregates attach each
prediction to its row through the order of the rows inside the group. Without
`order_by` that order is the scan order of `source`; with `order_by := col` it
is `col` (ties fall back to scan order), so the row-to-prediction alignment is
deterministic. For order-dependent models (RLS, whose recursive updates follow
the row order) it also fixes the order the rows are fed to the fit. The GLM
macros join each row to its group's model and need no `order_by`.

**Output columns:** all columns of `source` (original names), followed by

| Column | Type | Description |
|--------|------|-------------|
| `yhat` | `DOUBLE` | Prediction (GLM macros: on the response scale) |
| `yhat_lower` | `DOUBLE` | Lower prediction-interval bound (not for PLS, quantile, isotonic; always NULL for the GLM macros other than Poisson) |
| `yhat_upper` | `DOUBLE` | Upper prediction-interval bound (not for PLS, quantile, isotonic; always NULL for the GLM macros other than Poisson) |
| `is_training` | `BOOLEAN` | Whether the row was used to fit its group's model |

Rows are returned ordered by `group_col`.

**Training rows:** without `split`, rows with a non-NULL `y` train the model
and rows with NULL `y` are predicted. With `split`, rows whose split value is
`'train'` or NULL (and whose `y` is not NULL) are trained on; rows with any
other split value are predicted only.

**Prediction intervals:** `yhat_lower`/`yhat_upper` are leverage-aware
(`yhat ± t(n − p) · s · sqrt(1 + x₀ᵀ M x₀)`, see
[Methodology](../../METHODOLOGY.md#prediction-intervals)). They are NULL when no
interval exists for the group (zero residual degrees of freedom, singular
design) and equal to `yhat` for an exact fit.

**Options:** each macro accepts the options of the model it wraps (see the
model pages linked below), plus `confidence_level` (default `0.95`) and
`null_policy` (default `'drop'`, see [null_policy](#null_policy)) for all
macros except PLS, quantile, isotonic and the GLM macros (binomial, logistic,
negbinom, gamma, tweedie), which take their model's options. Option keys the model does not
support raise an error.

### Demo data

The examples below use this table:

```sql
CREATE OR REPLACE TABLE macro_demo AS
SELECT
    i AS t,
    CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS region,
    i::DOUBLE AS price,
    (i % 4)::DOUBLE AS promo,
    1.0 + (i % 3) AS weight,
    -- the last 6 periods have no sales yet
    CASE WHEN i <= 34 THEN 50.0 + 1.2 * i + 3.0 * (i % 4) + sin(i) END AS sales,
    round(5.0 + 0.3 * i + (i % 4))::DOUBLE AS visits,
    CASE WHEN i <= 26 THEN 'train' ELSE 'test' END AS split
FROM range(1, 41) t(i);
```

## ols_fit_predict_by

Options: [OLS](../regression/ols.md#options).

```sql
-- Per-region OLS; rows with NULL sales receive forecasts
SELECT region, t, sales, round(yhat, 2) AS yhat, is_training
FROM ols_fit_predict_by('macro_demo', region, sales, [price, promo], order_by := t)
WHERE NOT is_training;

-- 99% prediction intervals and an explicit train/test split
SELECT region, t, sales, round(yhat, 2) AS yhat, round(yhat_lower, 2) AS lo, round(yhat_upper, 2) AS hi
FROM ols_fit_predict_by('macro_demo', region, sales, [price, promo],
    options := {'confidence_level': 0.99}, split := split)
WHERE NOT is_training AND sales IS NOT NULL;
```

## ridge_fit_predict_by

Options: [Ridge](../regression/ridge.md#options) (`alpha` default `1.0`, ...).

```sql
SELECT * FROM ridge_fit_predict_by('macro_demo', region, sales, [price, promo], {'alpha': 0.5})
LIMIT 5;
```

## elasticnet_fit_predict_by

Options: [Elastic Net](../regression/elasticnet.md#options) (`alpha`, `l1_ratio`, ...).

```sql
SELECT * FROM elasticnet_fit_predict_by('macro_demo', region, sales, [price, promo],
    {'alpha': 0.1, 'l1_ratio': 0.7})
LIMIT 5;
```

## wls_fit_predict_by

Options: [WLS](../regression/wls.md#options). The weight column is the fifth
positional argument.

```sql
SELECT * FROM wls_fit_predict_by('macro_demo', region, sales, [price, promo], weight)
LIMIT 5;
```

## rls_fit_predict_by

Options: [RLS](../regression/rls.md#options) (`forgetting_factor`, `initial_p_diagonal`, ...).

```sql
-- RLS is order-dependent: feed each region's rows in time order
SELECT * FROM rls_fit_predict_by('macro_demo', region, sales, [price, promo],
    {'forgetting_factor': 0.95}, order_by := t)
LIMIT 5;
```

## huber_fit_predict_by

Options: [Huber](../regression/huber.md#options) (`epsilon`, `alpha`, ...).

```sql
SELECT * FROM huber_fit_predict_by('macro_demo', region, sales, [price, promo], {'epsilon': 1.5})
LIMIT 5;
```

## ransac_fit_predict_by

Options: [RANSAC](../regression/ransac.md#options) (`residual_threshold`, `max_trials`, `random_state`, ...).

```sql
SELECT * FROM ransac_fit_predict_by('macro_demo', region, sales, [price, promo], {'random_state': 42})
LIMIT 5;
```

## theil_sen_fit_predict_by

Options: [Theil-Sen](../regression/theil_sen.md#options).

```sql
SELECT * FROM theil_sen_fit_predict_by('macro_demo', region, sales, [price, promo])
LIMIT 5;
```

## bls_fit_predict_by

Options: [BLS](../regression/bls.md#bls_fit_agg) (`lower_bound`, `upper_bound`,
`fit_intercept` default `false`, `max_iterations`, `tolerance`). Without
bounds the fit is non-negative least squares.

```sql
SELECT * FROM bls_fit_predict_by('macro_demo', region, sales, [price, promo],
    {'lower_bound': 0.0, 'upper_bound': 5.0, 'fit_intercept': true})
LIMIT 5;
```

## alm_fit_predict_by

Augmented linear model with a selectable error distribution. Options:
[ALM](../glm/alm.md) (`distribution`, `loss`, `quantile`, `role_trim`,
`max_iterations`, `tolerance`, `fit_intercept`).

```sql
-- Laplace errors: robust, median-type regression
SELECT * FROM alm_fit_predict_by('macro_demo', region, sales, [price, promo],
    {'distribution': 'laplace'})
LIMIT 5;
```

## poisson_fit_predict_by

Poisson GLM for counts. Options: [Poisson](../glm/poisson.md) (`link`:
`'log'`, `'identity'`, `'sqrt'`; `max_iterations`, `tolerance`,
`fit_intercept`).

```sql
SELECT * FROM poisson_fit_predict_by('macro_demo', region, visits, [promo])
LIMIT 5;
```

## binomial_fit_predict_by

Binomial GLM for success rates in [0, 1]. Options:
[Binomial](../glm/binomial.md) (`binomial_link`: `'logit'`, `'probit'`,
`'cloglog'`; `max_iterations`, `tolerance`, `fit_intercept`, ...). `yhat` is
the predicted rate; `yhat_lower`/`yhat_upper` are NULL.

```sql
-- source must be a table or view name, so derive the response in a view
CREATE OR REPLACE VIEW macro_demo_glm AS
SELECT *, promo / 3.0 AS rate, (promo >= 2)::DOUBLE AS high FROM macro_demo;

SELECT region, t, round(rate, 3) AS rate, round(yhat, 3) AS yhat, is_training
FROM binomial_fit_predict_by('macro_demo_glm', region, rate, [price])
ORDER BY region, t
LIMIT 5;
```

## logistic_fit_predict_by

Logistic regression for a binary (0/1) response. Options:
[Logistic](../glm/logistic.md). `yhat` is the predicted probability.

```sql
SELECT region, t, high, round(yhat, 3) AS p_high, is_training
FROM logistic_fit_predict_by('macro_demo_glm', region, high, [price], split := split)
WHERE NOT is_training
ORDER BY region, t
LIMIT 5;
```

## negbinom_fit_predict_by

Negative binomial GLM for over-dispersed counts (log link). Options:
[Negative Binomial](../glm/negbinom.md) (`theta`, ...). `yhat` is the
expected count.

```sql
SELECT region, t, visits, round(yhat, 2) AS yhat
FROM negbinom_fit_predict_by('macro_demo', region, visits, [price])
ORDER BY region, t
LIMIT 5;
```

## gamma_fit_predict_by

Gamma GLM for strictly positive, right-skewed responses (log link). Options:
[Gamma](../glm/gamma.md). `yhat` is the expected value.

```sql
SELECT region, t, sales, round(yhat, 2) AS yhat, is_training
FROM gamma_fit_predict_by('macro_demo', region, sales, [price, promo])
WHERE NOT is_training
ORDER BY region, t;
```

## tweedie_fit_predict_by

Tweedie GLM for non-negative responses with exact zeros (log link). Options:
[Tweedie](../glm/tweedie.md) (`power`, default `1.5`). `yhat` is the expected
value.

```sql
SELECT region, t, promo, round(yhat, 3) AS yhat
FROM tweedie_fit_predict_by('macro_demo', region, promo, [price], {'power': 1.5})
ORDER BY region, t
LIMIT 5;
```

## pls_fit_predict_by

Options: [PLS](../regression/pls.md#options) (`n_components` default `1`,
`fit_intercept`). Returns `yhat` and `is_training` only.

```sql
SELECT * FROM pls_fit_predict_by('macro_demo', region, sales, [price, promo], {'n_components': 2})
LIMIT 5;
```

## quantile_fit_predict_by

Options: [Quantile](../regression/quantile.md#options) (`tau` default `0.5`,
`fit_intercept`). Returns `yhat` and `is_training` only.

```sql
SELECT * FROM quantile_fit_predict_by('macro_demo', region, sales, [price, promo], {'tau': 0.9})
LIMIT 5;
```

## isotonic_fit_predict_by

Options: [Isotonic](../regression/isotonic.md#options) (`increasing` default
`true`). Takes a single `x_col`, not a list. Returns `yhat` and `is_training`
only.

```sql
SELECT * FROM isotonic_fit_predict_by('macro_demo', region, sales, price)
LIMIT 5;
```

## glmm_fit_by

```text
glmm_fit_by(source VARCHAR, group_col, y_col, x_cols, options := NULL) -> TABLE
```

Fits **one** mixed-effects model across all groups (random intercept per
group) and returns one row per group: `group`, `ranef`, `ranef_se`, `n`,
`fixed_intercept`, `fixed_coefficients`, `var_group`, `var_residual`, `icc`.
See [GLMM](../glm/glmm.md) for the options.

```sql
SELECT "group", round(ranef, 3) AS ranef, n, round(icc, 3) AS icc
FROM glmm_fit_by('macro_demo', region, price, [promo]);
```

## eb_shrink_by

```text
eb_shrink_by(source VARCHAR, estimate_col, se_col, options := NULL) -> TABLE
```

Empirical-Bayes shrinkage of existing per-group estimates toward their
precision-weighted mean. Returns all source columns plus `shrunken`,
`shrunken_se`, `weight`, `mu`, `tau_squared`. See
[Empirical-Bayes shrinkage](../glm/eb_shrink.md).

```sql
CREATE OR REPLACE TABLE macro_estimates AS
SELECT * FROM (VALUES ('a', 1.2, 0.3), ('b', 0.4, 0.5), ('c', 2.1, 0.8), ('d', 0.9, 0.2))
    t(segment, estimate, se);

SELECT segment, estimate, round(shrunken, 3) AS shrunken
FROM eb_shrink_by('macro_estimates', estimate, se);
```

## aid_by

```text
aid_by(source VARCHAR, group_col, y_col, options := NULL) -> TABLE
```

Demand classification per group (one row per group). Options:
`intermittent_threshold`, `outlier_method`. See [AID](../aid/aid.md).

```sql
SELECT region, demand_type, is_intermittent, zero_proportion
FROM aid_by('macro_demo', region, promo);
```

## aid_anomaly_by

```text
aid_anomaly_by(source VARCHAR, group_col, order_col, y_col, options := NULL) -> TABLE
```

Per-observation anomaly flags (`stockout`, `new_product`, `obsolete_product`,
`high_outlier`, `low_outlier`), returned with the group and order columns. See
[AID](../aid/aid.md).

```sql
SELECT region, t, stockout, high_outlier
FROM aid_anomaly_by('macro_demo', region, t, promo)
WHERE stockout OR high_outlier;
```

## null_policy

| Value | Training rows | Predicted rows |
|-------|---------------|----------------|
| `'drop'` (default) | `y` is not NULL | all rows |
| `'drop_y_zero_x'` | `y` is not NULL and no feature equals 0 | all rows |

## See Also

- [Fit-predict aggregates](../regression/fit_predict_agg.md) - The underlying aggregates
- [Window fit-predict](../regression/fit_predict_window.md) - Expanding/rolling window predictions
- [ALM](../glm/alm.md), [Poisson](../glm/poisson.md), [AID](../aid/aid.md)
