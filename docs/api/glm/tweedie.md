# Tweedie GLM

Regression for non-negative continuous responses with a point mass at zero —
insurance losses, rainfall, intermittent sales revenue. With a variance power
between 1 and 2 the Tweedie distribution is a compound Poisson–Gamma. Fitted
with a log link by IRLS.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `tweedie_fit_agg` | Aggregate | Fit a Tweedie GLM (log link) |
| `tweedie_fit_predict_by` | Table macro | Per-group fit, expected values for every row, see [Table macros](../macros/table_macros.md#tweedie_fit_predict_by) |

## tweedie_fit_agg

**Signature:**

```text
tweedie_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Response; non-negative, zeros allowed |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `power` (alias `tweedie_power`) | DOUBLE | `1.5` | Variance power `p` in `[1, 2]`: `Var(y) = phi * mu^p`. `1` is Poisson-like, `2` is Gamma. |
| `fit_intercept` | BOOLEAN | `true` | Include an intercept |
| `max_iterations` | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `compute_inference` | BOOLEAN | `false` | Standard errors, z-tests, p-values, intervals |
| `confidence_level` | DOUBLE | `0.95` | Interval level |
| `glm_lambda` | DOUBLE | `0.0` | L2 (ridge) penalty strength |
| `offset` | INTEGER | none | 1-based index into `x` of an offset column (coefficient fixed at 1, removed from the design) |
| `feature_names`, `prior`, `vcov` | | | See [Explicit priors](priors.md) |

The link is always log, regardless of `power`.

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | DOUBLE[] | Coefficients on the log scale; `NaN` for an aliased column |
| `intercept` | DOUBLE | Intercept |
| `deviance` | DOUBLE | Residual deviance |
| `null_deviance` | DOUBLE | Intercept-only deviance |
| `pseudo_r_squared` | DOUBLE | `1 - deviance / null_deviance` |
| `aic` | DOUBLE | Akaike information criterion |
| `dispersion` | DOUBLE | Pearson estimate of `phi`; scales the standard errors |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | IRLS iterations performed |
| `converged` | BOOLEAN | Whether IRLS reached the tolerance |
| `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper` | DOUBLE[] | Only with `compute_inference` |
| `family` | VARCHAR | `'tweedie'`; used by [`predict`](../regression/model_tools.md#predict) |
| `link` | VARCHAR | `'log'` |

**Example:**

```sql
-- Policy losses: roughly 40% of policies have no loss at all
CREATE OR REPLACE TABLE losses AS
SELECT i AS policy_id,
       (i % 5)::DOUBLE AS risk_score,
       CASE WHEN (i * 13) % 7 < 3 THEN 0
            ELSE exp(0.2 + 0.25 * (i % 5)) * (0.5 + ((i * 37) % 100) / 100.0)
       END AS loss
FROM range(300) r(i);

SELECT tweedie_fit_agg(loss, [risk_score], {'power': 1.5}) AS fit FROM losses;

-- Compare variance powers (power must be a constant, so fit each separately)
SELECT 1.3 AS power, (tweedie_fit_agg(loss, [risk_score], {'power': 1.3})).deviance AS deviance FROM losses
UNION ALL
SELECT 1.5, (tweedie_fit_agg(loss, [risk_score], {'power': 1.5})).deviance FROM losses
UNION ALL
SELECT 1.7, (tweedie_fit_agg(loss, [risk_score], {'power': 1.7})).deviance FROM losses;
```

## Prediction

Apply a fitted model to new rows with
[`predict(model, x)`](../regression/model_tools.md#predict). The model's `link`
field maps the linear predictor back to the response scale; pass
`{'type': 'link'}` for the log mean instead. [`tidy`](../regression/model_tools.md#tidy)
and [`glance`](../regression/model_tools.md#glance) give the per-term table and
the fit summary.

```sql
WITH fit AS (SELECT tweedie_fit_agg(loss, [risk_score], {'power': 1.5}) AS m FROM losses)
SELECT m.family, m.link,
       round(predict(m, [4.0]), 4) AS response,   -- expected loss at risk score 4
       round(predict(m, [4.0], {'type': 'link'}), 4) AS linear_predictor
FROM fit;
```

### tweedie_fit_predict_by

```text
tweedie_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
```

Fits `tweedie_fit_agg` on each group's training rows (`y` not NULL and `split`
NULL or `'train'`) and applies `predict(model, x)` to every row. Returns all
source columns plus `yhat` (response scale), `yhat_lower` and `yhat_upper`
(NULL: no interval for this GLM) and `is_training`, ordered by the group
column. A group whose fit fails gets NULL `yhat`. See
[Table macros](../macros/table_macros.md#tweedie_fit_predict_by).

```sql
CREATE OR REPLACE VIEW losses_seg AS
SELECT *,
       CASE WHEN policy_id % 2 = 0 THEN 'even' ELSE 'odd' END AS segment,
       CASE WHEN policy_id % 10 < 8 THEN 'train' ELSE 'test' END AS split
FROM losses;

SELECT segment, policy_id, loss, round(yhat, 3) AS yhat, is_training
FROM tweedie_fit_predict_by('losses_seg', segment, loss, [risk_score], {'power': 1.5}, split := split)
WHERE NOT is_training
ORDER BY segment, policy_id
LIMIT 4;
```

## Choosing the power

Common choices are `1.5` (the default) for insurance pure premiums and
`1.1`–`1.3` when the data look closer to over-dispersed counts. `power` must be
a constant per query; to compare candidates, fit each and compare them on
held-out data (deviances at different powers are not directly comparable).

## NULL and Invalid Input Handling

- Rows where `y` or the `x` list is NULL are skipped; a NULL element inside `x`
  or any non-finite value drops the row. `n_observations` counts the rows used.
- Fewer than two usable rows, a negative `y`, a `power` outside `[1, 2]`, or a
  failed fit returns `NULL`.

## See Also

- [Gamma GLM](gamma.md) — strictly positive responses (`power = 2`)
- [Poisson GLM](poisson.md) — counts (`power = 1`)
- [Mixed-effects GLMs](glmm.md)
