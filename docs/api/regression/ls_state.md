# Least-Squares States (store, combine, fit later)

A least-squares state is a compact summary of a group's rows: the row counts, the
weighted means and the centred co-moments of `y` and the features. It is a BLOB of
`O(p²)` bytes whatever the number of rows (144 bytes for two features). States
built per partition can be stored in a table, merged later and fitted without the
rows, which makes refits incremental:

- a yearly model from 365 daily states instead of a scan of every row;
- append today's state to the stored ones and refit;
- one stored state for a whole grid of ridge penalties.

The statistics come from anofox-regression (`SufficientStats`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ls_state_agg(y, x[, weight])` | Aggregate | State of a group's rows (BLOB) |
| `ls_state_combine_agg(state)` | Aggregate | Merge states into one, as if `ls_state_agg` had seen all their rows |
| `ls_state_info(state)` | Scalar | `STRUCT(n_observations, n_features, weighted)` |
| `ols_fit_from_state(state[, options])` | Scalar | OLS fit, same STRUCT as `ols_fit_agg` |
| `wls_fit_from_state(state[, options])` | Scalar | WLS fit (weights from `ls_state_agg(y, x, weight)`), same STRUCT as `wls_fit_agg` |
| `ridge_fit_from_state(state[, options])` | Scalar | Ridge fit, same STRUCT as `ridge_fit_agg`; the penalty is applied here |

## Example

```sql
CREATE OR REPLACE TABLE sales AS
SELECT DATE '2026-01-01' + (i % 30)::INTEGER AS day, sin(i * 0.37) * 3 + 100 AS price, cos(i * 1.3) AS promo,
       50 - 0.4 * (sin(i * 0.37) * 3 + 100) + 2.5 * cos(i * 1.3) + sin(i * 7.1) AS units
FROM range(30000) r(i);

-- One state per day (store this table; it holds 30 small BLOBs).
CREATE OR REPLACE TABLE daily_states AS
SELECT day, ls_state_agg(units, [price, promo]) AS state FROM sales GROUP BY day;

-- The month's model from the daily states, without the rows.
SELECT round((m).coefficients[1], 4) AS price_effect, round((m).r_squared, 4) AS r2, (m).n_observations AS n
FROM (SELECT ols_fit_from_state(ls_state_combine_agg(state), {'compute_inference': true}) AS m FROM daily_states);

-- Ridge with any penalty from the same states.
SELECT (ridge_fit_from_state(ls_state_combine_agg(state), {'alpha': 10.0})).coefficients AS coef FROM daily_states;
```

## What a state can and cannot give

A fit from a state returns what the direct fit on the same rows returns: the
coefficients (aliased columns `NaN`, as in R's `lm`), the classical standard errors,
t / p values and confidence intervals with `compute_inference`, R², adjusted R², the F
test, the log-likelihood and information criteria, and the coefficient covariance
(`vcov`), so `predict(model, x, {'interval': 'prediction'})` works on it. The values
equal the direct fit up to floating-point summation order (about 1e-12 relative).

Outputs that need the individual rows are not available, and the option raises an
error instead of returning an approximation:

| Option | Why |
|--------|-----|
| `hc_type` (HC0–HC3 sandwich) | needs every residual: `Σ eᵢ² xᵢxᵢᵀ` |
| `compute_loo` (PRESS, LOO R²) | needs every row's leverage |

Residuals, fitted values and `*_augment_agg` likewise need the rows.

**Options:** `fit_intercept`, `compute_inference`, `confidence_level`, `solver`, plus
`alias_tolerance` for OLS / WLS and `alpha` (`lambda`) / `lambda_scaling` for ridge.
NULL options mean no options.

## Determinism

`ls_state_agg` adds the rows in a canonical order, and `ls_state_combine_agg` merges
the states in the order of their bytes. So a state's bytes, and every fit from it, do
not depend on the row order or the number of threads (#226).

## NULL handling

Rows with a NULL `y`, a NULL `x` list or element, or a NULL weight are skipped. A group
without rows gives a NULL state, a NULL state gives a NULL fit, and
`ls_state_combine_agg` skips NULL states. States with different feature counts cannot
be merged, and a BLOB that is not a state raises an error.

## See Also

- [OLS](ols.md), [WLS](wls.md), [Ridge](ridge.md): the direct fits
- [Model tools](model_tools.md): `predict`, `tidy`, `glance` on the fitted STRUCT
