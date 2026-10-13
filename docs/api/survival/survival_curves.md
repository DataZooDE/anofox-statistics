# Survival Curves

Non-parametric and model-based survival curves for right-censored durations.
Results match R's `survival::survfit`, `survival::survdiff` and
`survival::survreg` (with `psurvreg`).

## Functions

| Function | Type | Description | R reference |
|----------|------|-------------|-------------|
| `kaplan_meier_agg` | Aggregate | Kaplan-Meier curve with Greenwood confidence intervals | `survfit(Surv(time, event) ~ 1)` |
| `kaplan_meier_by` | Table macro | One Kaplan-Meier curve per stratum (`curve` schema) | `survfit(Surv(time, event) ~ strata)` |
| `log_rank_test_agg` | Aggregate | Log-rank test of equal survival across groups | `survdiff(Surv(time, event) ~ group)` |
| `aft_survival` | Scalar | `S(t \| x)` of a fitted AFT model | `1 - psurvreg(t, lp, scale, dist)` |

`event` is BOOLEAN or numeric: true / non-zero when the event was observed,
false / 0 when the row is right-censored. Rows with a NULL time, event or group
are skipped; empty input gives NULL.

The examples on this page use this table:

```sql
CREATE OR REPLACE TABLE churn AS
SELECT customer_id, plan,
       least(raw_months, 24.0) AS months,
       raw_months <= 24 AS churned,
       (customer_id % 4)::DOUBLE AS support_calls
FROM (
    SELECT i AS customer_id, CASE WHEN i % 3 = 0 THEN 'annual' ELSE 'monthly' END AS plan,
           round(exp(2.0 + 0.6 * (i % 3 = 0)::INTEGER - 0.15 * (i % 4))
                 * (0.3 + ((i * 37) % 100) / 60.0), 1) AS raw_months
    FROM range(200) r(i)
);
```

## kaplan_meier_agg

```sql skip
kaplan_meier_agg(time DOUBLE, event [, conf_level DOUBLE [, conf_type VARCHAR]])
    -> STRUCT(time DOUBLE, surv DOUBLE, conf_low DOUBLE, conf_high DOUBLE,
              n_risk BIGINT, n_event BIGINT, n_censor BIGINT, std_err DOUBLE)[]
```

One row per distinct observed time (event or censoring), as the rows of an R
`survfit` object: the number at risk just before `time`, the events and
censorings at `time`, the survival estimate `surv` and its pointwise confidence
interval. Ties follow R: rows censored at an event time are still at risk at
that time.

| Argument | Default | Description |
|----------|---------|-------------|
| `conf_level` | `0.95` | Confidence level of `conf_low` / `conf_high` |
| `conf_type` | `'log'` | `'log'` (R's default, `exp(log S -/+ z se)` capped at 1), `'log-log'`, or `'plain'` (`S -/+ z S se` clipped to [0, 1]) |

`std_err` is the Greenwood standard error of `surv` (R's
`summary(survfit)$std.err`). The interval collapses to `surv` before the first
event and is NULL once `surv` reaches 0 (R gives `NA`).

```sql
SELECT plan, unnest(r, recursive := true)
FROM (SELECT plan, kaplan_meier_agg(months, churned) AS r FROM churn GROUP BY plan)
ORDER BY plan
LIMIT 5;
```

## kaplan_meier_by

```sql skip
kaplan_meier_by(source, strata_col, time_col, event_col [, conf_level := 0.95] [, conf_type := 'log'])
```

One Kaplan-Meier curve per stratum in the integration contract's `curve`
schema, ready to plot as step curves: the stratum column, `model_id` (the stratum),
`series` (the stratum), `curve_type` (`'km'`), `x` (time), `y` (survival),
`y_low` / `y_high` (pointwise interval), `threshold` (NULL), `weight` (number at
risk), `n_censor` (censor marks), then `n_risk`, `n_event` and `std_err`.

```sql
SELECT series, x, y, y_low, y_high, n_censor
FROM kaplan_meier_by('churn', plan, months, churned, conf_type := 'log-log')
WHERE x <= 3;
```

## log_rank_test_agg

```sql skip
log_rank_test_agg(time DOUBLE, event, group VARCHAR | BIGINT) -> STRUCT(<unified test fields>)
```

The k-group log-rank test (`survdiff`, `rho = 0`): `statistic` is the
chi-squared, `df1` the number of groups with expected events minus one, `n` the
number of rows. Fewer than two groups gives NULL.

```sql
SELECT unnest(log_rank_test_agg(months, churned, plan)) FROM churn;
```

## aft_survival

```sql skip
aft_survival(model STRUCT, x DOUBLE[], times DOUBLE[]) -> STRUCT(time DOUBLE, surv DOUBLE)[]
```

The model-based survival curve `S(t | x) = P(T > t | x)` of the model returned
by [`aft_fit_agg`](aft.md), for one covariate row `x` (one value per
coefficient), at each of `times`. `S = 1` for `t <= 0`. Compare it with the
Kaplan-Meier curve of the same data to check the fitted distribution, or plot
one curve per covariate profile (survminer's `ggadjustedcurves`).

```sql
WITH fit AS (
    SELECT aft_fit_agg(months, [support_calls], churned::DOUBLE, {'dist': 'weibull'}) AS m FROM churn
)
SELECT calls, unnest(aft_survival(m, [calls], [3, 6, 12, 24]), recursive := true)
FROM fit, (VALUES (0.0), (3.0)) v(calls)
ORDER BY calls;
```

## See Also

- [AFT survival regression](aft.md) - `aft_fit_agg`, `aft_cdf`, `aft_quantile`
- [Hypothesis tests](../statistics/hypothesis.md) - the unified test fields
