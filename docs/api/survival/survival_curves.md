# Survival Curves

Non-parametric and model-based survival curves for right-censored durations.
Results match R's `survival::survfit`, `survival::survdiff` and
`survival::survreg` (with `psurvreg`).

## Functions

| Function | Type | Description | R reference |
|----------|------|-------------|-------------|
| `kaplan_meier_agg` | Aggregate | Kaplan-Meier curve with Greenwood confidence intervals; options for weights, left truncation, robust SEs, Nelson-Aalen / Fleming-Harrington cumulative hazard | `survfit(Surv(time, event) ~ 1)` |
| `kaplan_meier_by` | Table macro | One Kaplan-Meier curve per stratum (`curve` schema) | `survfit(Surv(time, event) ~ strata)` |
| `log_rank_test_agg` | Aggregate | Log-rank test of equal survival across groups; options `rho` (Peto-Peto) and `strata` | `survdiff(Surv(time, event) ~ group, rho)` |
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

### Options: weights, left truncation, robust variance, cumulative hazard

```sql skip
kaplan_meier_agg(time DOUBLE, event, options MAP/STRUCT)
kaplan_meier_agg(time DOUBLE, event, columns DOUBLE[], options MAP/STRUCT)
    -> STRUCT(time DOUBLE, surv DOUBLE, conf_low DOUBLE, conf_high DOUBLE,
              n_risk DOUBLE, n_event DOUBLE, n_censor DOUBLE, std_err DOUBLE,
              cumhaz DOUBLE, std_chaz DOUBLE)[]
```

With an options argument the curve comes from anofox-regression's `survfit`
(R `survfit(Surv(entry, time, event) ~ 1, weights, cluster, ctype, robust)`). Per-row
case weights, entry times and cluster ids are columns of the `columns` list, named by
their 1-based position in the options (as `cox_fit_agg`'s `strata` / `entry` /
`cluster` are columns of `x`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `conf_level` | DOUBLE | 0.95 | Confidence level |
| `conf_type` | VARCHAR | `'log'` | `'log'`, `'log-log'` or `'plain'` |
| `hazard` | VARCHAR | `'nelson-aalen'` | Cumulative hazard: `'nelson-aalen'` (`ctype = 1`) or `'fleming-harrington'` (`ctype = 2`, tie-corrected) |
| `robust` | BOOLEAN | survfit's default | Infinitesimal-jackknife standard errors; by default on when the weights are not all integers or a cluster is given |
| `timefix` | BOOLEAN | true | Merge times that differ by rounding error (`Surv(timefix = TRUE)`) |
| `weight` | INTEGER | none | Position in `columns` of a case weight (`survfit(weights = )`) |
| `entry` | INTEGER | none | Position in `columns` of the entry time: the row is at risk on `(entry, time]` (left truncation, `Surv(entry, time, event)`) |
| `cluster` | INTEGER | none | Position in `columns` of a cluster id: robust variance summed within clusters (`survfit(cluster = )`) |

The result adds `cumhaz` and its standard error `std_chaz`, and reports the counts
as DOUBLE (weighted sums). `std_err` is the standard error of `surv` (Greenwood, or
robust). A NULL in `columns` (or a NULL list) leaves the row out. Without weights,
entry times, clusters or `robust`, the curve equals the one of the positional form
(which stays on anofox-statistics), except that `timefix` merges times that differ
only by rounding error, as R does. Strata are `GROUP BY` (or `kaplan_meier_by`).

`cluster` is `survfit(cluster = )`. With `survfit(id = )` R additionally stops
counting a subject's intermediate `(start, stop]` rows as censorings (and drops
those time points); here every non-event row counts in `n_censor`.

```sql
-- Weighted curve with the Fleming-Harrington hazard
SELECT p.time, p.surv, p.std_err, p.cumhaz
FROM (SELECT unnest(kaplan_meier_agg(months, churned, [1.0 + support_calls],
                                     {'weight': 1, 'hazard': 'fleming-harrington'})) AS p FROM churn)
LIMIT 4;
```

```sql
-- Left truncation: customers observed only from month 1 on (delayed entry)
SELECT p.time, p.n_risk, p.surv
FROM (SELECT unnest(kaplan_meier_agg(months, churned, [0.5], {'entry': 1})) AS p
      FROM churn WHERE months > 0.5)
LIMIT 3;
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

### Options: G-rho family and strata

```sql skip
log_rank_test_agg(time DOUBLE, event, group, options MAP/STRUCT)
log_rank_test_agg(time DOUBLE, event, group, columns DOUBLE[], options MAP/STRUCT)
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rho` | DOUBLE | 0 | Weight each event time by `S(t-)^rho` (pooled Kaplan-Meier within the stratum): 0 is the log-rank test, 1 the Peto-Peto modification of the Gehan-Wilcoxon test (`survdiff(rho = )`) |
| `strata` | INTEGER | none | Position in `columns` of a stratum code: observed, expected and variance summed over strata (`survdiff(... + strata(s))`) |

With `rho = 0` and no strata the test is the one of the three-argument form
(anofox-statistics); otherwise it comes from anofox-regression's `survdiff`.
`method` is `'Log-rank test'`, `'Peto-Peto test'` (`rho = 1`) or `'G-rho test'`.
A NULL stratum leaves the row out.

```sql
SELECT (log_rank_test_agg(months, churned, plan, {'rho': 1})).p_value AS peto_peto,
       (log_rank_test_agg(months, churned, plan, [support_calls], {'strata': 1})).p_value AS stratified
FROM churn;
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
- [Cox proportional hazards](cox.md) - `cox_fit_agg`, `cox_survival` (model-based curves of a Cox fit)
- [Hypothesis tests](../statistics/hypothesis.md) - the unified test fields
