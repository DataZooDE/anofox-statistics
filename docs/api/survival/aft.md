# AFT Survival Regression

Accelerated failure time models for duration data with **right censoring** — rows
where the event has not happened yet.

Duration data almost always contains open cases: orders not yet delivered,
customers not yet churned. Fitting a Gamma or lognormal GLM on the observed
durations treats those as if the event had occurred at the cut-off, which
attenuates covariate effects and understates the spread.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `aft_fit_agg` | Aggregate | Fit an AFT model with right censoring |
| `aft_cdf` | Scalar | `P(T <= t)` for a fitted model |
| `aft_quantile` | Scalar | The `p`-quantile of `T` |

The examples on this page use this table:

```sql
-- Purchase-order lead times, observed until day 15; open orders are censored
CREATE OR REPLACE TABLE po_lines AS
SELECT order_id, supplier_rating, order_qty,
       least(raw_days, 15.0) AS days,
       CASE WHEN raw_days > 15 THEN 0.0 ELSE 1.0 END AS delivered
FROM (
    SELECT i AS order_id,
           (1 + i % 5)::DOUBLE AS supplier_rating,
           (10 + (i % 7) * 5)::DOUBLE AS order_qty,
           exp(1.5 + 0.2 * (1 + i % 5) + 0.01 * (10 + (i % 7) * 5))
               * (0.4 + ((i * 37) % 100) / 80.0) AS raw_days
    FROM range(300) r(i)
);
```

## aft_fit_agg

**Signature:**

```text
aft_fit_agg(time DOUBLE, x DOUBLE[], event DOUBLE [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `time` | DOUBLE | Event or censoring time, strictly positive |
| `x` | DOUBLE[] | Feature values for the row |
| `event` | DOUBLE | 1 = event observed, 0 = right-censored |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dist (alias `distribution`) | VARCHAR | 'weibull' | `weibull`, `lognormal`, `loglogistic`, `exponential` |
| fit_intercept | BOOLEAN | true | Include an intercept |
| max_iterations | INTEGER | 100 | Newton iterations |
| tolerance | DOUBLE | 1e-9 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Standard errors, z-tests, intervals |
| confidence_level | DOUBLE | 0.95 | Interval level |
| feature_names, prior, vcov | | | See [Explicit priors](../glm/priors.md) |

**Returns:**

```text
STRUCT(coefficients DOUBLE[], intercept DOUBLE, scale DOUBLE,
       log_likelihood DOUBLE, null_log_likelihood DOUBLE, aic DOUBLE, bic DOUBLE,
       n_observations BIGINT, n_events BIGINT, n_censored BIGINT,
       n_features BIGINT, iterations INTEGER, converged BOOLEAN
     [, std_errors DOUBLE[], z_values DOUBLE[], p_values DOUBLE[],
        ci_lower DOUBLE[], ci_upper DOUBLE[],
        intercept_std_error DOUBLE, log_scale_std_error DOUBLE])
```

**Example:**

```sql
-- Delivery lead times; `delivered = 0` marks orders still open.
SELECT aft_fit_agg(days, [supplier_rating, order_qty], delivered, {
    'dist': 'weibull',
    'compute_inference': true
}) AS fit
FROM po_lines;
```

## The model

```
log T = x'beta + sigma * W
```

`W` follows a fixed standard distribution. An observed event contributes its
density; a censored row contributes its survival, `P(T > t)`, which is the
information "we know only that it lasted at least this long".

| `dist` | Distribution of `T` | `W` | Notes |
|--------|--------------------|-----|-------|
| `weibull` | Weibull | standard extreme value | Monotone hazard |
| `lognormal` | Lognormal | standard normal | Non-monotone hazard |
| `loglogistic` | Log-logistic | standard logistic | Heavier tail than lognormal |
| `exponential` | Exponential | standard extreme value | Weibull with `scale` fixed at 1 |

## Interpreting coefficients

Coefficients are on the **log-time scale**. A coefficient of `0.3` means a
one-unit increase in that feature multiplies the expected duration by
`exp(0.3) = 1.35` — a 35% longer time to event. Positive means slower.

`scale` is `sigma`. Smaller means the durations cluster more tightly around the
fitted median. It is exactly 1.0 for `exponential`, where it is not estimated,
and `log_scale_std_error` is `NaN` there.

## aft_cdf / aft_quantile

Stateless scalar helpers that evaluate a fitted AFT model at a given linear
predictor.

**Signatures:**

```text
aft_cdf(t DOUBLE, eta DOUBLE, scale DOUBLE, dist VARCHAR) -> DOUBLE
aft_quantile(p DOUBLE, eta DOUBLE, scale DOUBLE, dist VARCHAR) -> DOUBLE
```

| Parameter | Description |
|-----------|-------------|
| `t` | Time at which to evaluate `P(T <= t)` |
| `p` | Probability in (0, 1) for the quantile |
| `eta` | Linear predictor `intercept + x'beta` from `aft_fit_agg` |
| `scale` | `scale` field from `aft_fit_agg` |
| `dist` | Distribution used for the fit: `weibull`, `lognormal`, `loglogistic`, `exponential` |

A NULL in any argument returns NULL; an unknown distribution name raises an
error.

```sql
-- P(delivery within 10 days) and the median predicted duration per order
WITH fit AS (
    SELECT aft_fit_agg(days, [supplier_rating, order_qty], delivered) AS f
    FROM po_lines
)
SELECT p.order_id,
       aft_cdf(10.0,
               f.intercept + f.coefficients[1] * p.supplier_rating
                           + f.coefficients[2] * p.order_qty,
               f.scale, 'weibull') AS p_within_10,
       aft_quantile(0.5,
               f.intercept + f.coefficients[1] * p.supplier_rating
                           + f.coefficients[2] * p.order_qty,
               f.scale, 'weibull') AS median_days
FROM po_lines p, fit
ORDER BY p.order_id
LIMIT 5;

-- Direct evaluation
SELECT aft_cdf(30.0, 2.5, 0.5, 'weibull') AS cdf,
       aft_quantile(0.5, 2.5, 0.5, 'weibull') AS median;
```

## Degenerate inputs

These return `NULL` rather than `NaN` coefficients:

| Situation | Reason |
|-----------|--------|
| Every row censored | The model is not identified |
| A non-positive time | `log T` is undefined |
| An `event` value other than 0 or 1 | Ambiguous |
| Fewer observations than parameters | Under-determined |

Rows with a `NULL` or non-finite value in `time`, `event` or any feature are
dropped from the fit; `n_observations` reports how many were actually used.

## Use Cases

- **Supplier lead times** — open purchase orders are censored, not fast.
- **Time to churn** — customers who have not churned yet.
- **Time to failure** — units still running at the end of the observation window.
- **Any duration with a cut-off** — anything where the observation window closes
  before every case resolves.

## See Also

- [Explicit priors](../glm/priors.md) — priors and `vcov` work on AFT coefficients
- [Gamma GLM](../glm/gamma.md) — for positive durations with no censoring
