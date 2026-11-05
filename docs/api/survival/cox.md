# Cox Proportional Hazards Regression

Semi-parametric regression for right-censored and counting-process survival data:
the hazard of row `i` is `h0(t) * exp(x_i'beta)` with an unspecified baseline hazard
`h0`. Results match R's `survival::coxph`, `cox.zph`, `survfit.coxph` and
`residuals.coxph` (the fit is anofox-regression's `survival::cox`).

## Functions

| Function | Type | Description | R reference |
|----------|------|-------------|-------------|
| `cox_fit_agg` | Aggregate | Fit the model: log hazard ratios, hazard ratios with intervals, LR / Wald / score tests, concordance, baseline hazard | `coxph(Surv(time, event) ~ x)` |
| `cox_zph_agg` | Aggregate | Proportional-hazards test per covariate and global | `cox.zph(fit)` |
| `cox_survival` | Scalar | Predicted survival `S(t \| x)` from a fitted model | `survfit(fit, newdata)` |
| `cox_augment_agg` | Window aggregate | Per-row linear predictor and residuals | `residuals(fit, type = ...)` |
| `cox_fit_by` | Table macro | One model STRUCT per group, as columns | |
| `cox_tidy_by` / `cox_glance_by` | Table macros | Terms and fit statistics per group (integration contract `terms` / `summary`) | `broom::tidy` / `glance` |
| `cox_augment_by` | Table macro | Every source row with its residuals (contract `obs`) | `broom::augment` |

The examples on this page use this table:

```sql
-- A follow-up study: days to an event, censored at the end of follow-up
CREATE OR REPLACE TABLE patients AS
SELECT id, site, age, treated, entry_day,
       least(raw_days, 400.0) AS days,
       CASE WHEN raw_days <= 400 THEN 1.0 ELSE 0.0 END AS died,
       1.0 + (id % 3) AS weight
FROM (
    SELECT i AS id, (i % 3)::VARCHAR AS site,
           (40 + (i * 7) % 35)::DOUBLE AS age,
           (i % 2)::DOUBLE AS treated,
           ((i * 11) % 20)::DOUBLE AS entry_day,
           round(25 + 300 * exp(-0.03 * ((40 + (i * 7) % 35) - 55) + 0.6 * (i % 2))
                 * (0.05 + ((i * 37) % 101) / 101.0), 1) AS raw_days
    FROM range(300) r(i)
);
```

## cox_fit_agg

**Signature:**

```text
cox_fit_agg(time DOUBLE, x DOUBLE[], event DOUBLE [, weight DOUBLE] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `time` | DOUBLE | Event or censoring time (the stop time of a counting-process row) |
| `x` | DOUBLE[] | Covariates of the row (no intercept: the baseline hazard absorbs it) |
| `event` | DOUBLE | 1 = event observed, 0 = censored |
| `weight` | DOUBLE | Optional case weight (`coxph(weights = w)`); a NULL weight drops the row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ties` | VARCHAR | `'efron'` | Tied event times: `'efron'` or `'breslow'` (`coxph(ties = )`) |
| `compute_inference` | BOOLEAN | true | Standard errors, z, p, intervals and `vcov` (cheap for a Cox fit, so on by default) |
| `confidence_level` | DOUBLE | 0.95 | Level of the Wald intervals |
| `robust` | BOOLEAN | false | Sandwich covariance, each row its own cluster (`coxph(robust = TRUE)`) |
| `cluster` | INTEGER | none | 1-based position in `x` of a cluster id: cluster-robust covariance (`+ cluster(id)`) |
| `strata` | INTEGER | none | 1-based position in `x` of a stratum code: separate baseline hazards (`+ strata(s)`) |
| `entry` | INTEGER | none | 1-based position in `x` of the entry (start) time: counting-process data `(entry, time]` (`Surv(start, stop, event)`) for left truncation and time-varying covariates |
| `offset` | INTEGER | none | 1-based position in `x` of an offset added to the linear predictor (`+ offset(o)`) |
| `ridge` | DOUBLE | 0 | Ridge penalty `ridge/2 * sum(beta^2)` (`ridge(..., theta = ridge, scale = FALSE)`) |
| `max_iterations` | INTEGER | 20 | Newton-Raphson iterations |
| `tolerance` | DOUBLE | 1e-9 | Convergence on the relative change of the log partial likelihood |
| `timefix` | BOOLEAN | true | Merge times that differ by rounding error, as `Surv(timefix = TRUE)` |

The `cluster`, `strata`, `entry` and `offset` columns are taken out of `x`: the
coefficients belong to the remaining columns, in order (as the GLMs' `offset`
option).

**Returns** (the stable model STRUCT; inference fields are NULL with
`compute_inference = false`):

| Field | Description |
|-------|-------------|
| `coefficients` | Log hazard ratios |
| `hazard_ratios`, `hr_conf_low`, `hr_conf_high` | `exp(coef)` and `exp` of the Wald interval |
| `std_errors`, `z_values`, `p_values`, `conf_low`, `conf_high` | Wald inference on the reported covariance (robust when `robust` / `cluster`) |
| `naive_std_errors` | Model-based (inverse information) standard errors |
| `vcov` | Reported covariance, row-major |
| `log_likelihood`, `null_log_likelihood` | Log partial likelihood at the estimate and at `beta = 0` (`fit$loglik`) |
| `lr_statistic`, `wald_statistic`, `score_statistic` (+ `_p_value`), `test_df` | The three global tests of `summary(coxph)`; the Wald test uses the model-based covariance |
| `concordance` | Harrell's C (`fit$concordance`) |
| `pseudo_r_squared` | `rms::cph`'s R2 |
| `aic`, `bic` | `AIC(fit)`, `BIC(fit)` (BIC uses the number of events); NULL for a ridge fit |
| `n_observations`, `n_events`, `n_censored`, `n_features`, `n_strata`, `iterations`, `converged`, `ties`, `robust` | |
| `baseline` | `[{stratum, time, cumhaz}]`: the baseline cumulative hazard at covariates 0 (`basehaz(fit, centered = FALSE)`) at every distinct time, per stratum (`stratum` NULL unstratified); `cox_survival` reads it |
| `model_type` (`'cox'`), `family`, `link`, `conf_level`, ... | The shared tail of every model STRUCT |

```sql
SELECT round(m.hazard_ratios[1], 4) AS hr_age, round(m.hazard_ratios[2], 4) AS hr_treated,
       round(m.hr_conf_low[2], 4) AS lo, round(m.hr_conf_high[2], 4) AS hi,
       round(m.concordance, 4) AS c, m.n_events
FROM (SELECT cox_fit_agg(days, [age, treated], died) AS m FROM patients);
```

```sql
-- Breslow ties, case weights and a robust covariance
SELECT (cox_fit_agg(days, [age, treated], died, weight, {'ties': 'breslow', 'robust': true})).std_errors
FROM patients;

-- Stratified by site (x position 3) with delayed entry (x position 4)
SELECT (cox_fit_agg(days, [age, treated, site::DOUBLE, entry_day], died,
                    {'strata': 3, 'entry': 4})).coefficients
FROM patients;
```

Rows with a NULL or non-finite `time`, `event`, weight or covariate are left out
(`n_observations` counts the rows used). Every row censored, no rows, or a
singular information matrix (for example a constant covariate) give NULL; an
`event` other than 0 / 1, an entry time not before the stop time, a negative
weight or an unknown option raise an error.

Unlike R, fractional case weights do not switch to the robust covariance
automatically: ask for it with `'robust': true` (R's default in that case).

## cox_zph_agg

```text
cox_zph_agg(time, x, event [, weight] [, options]) -> STRUCT(term VARCHAR, statistic DOUBLE, df DOUBLE, p_value DOUBLE)[]
```

The Grambsch-Therneau score test of proportional hazards (`cox.zph`, survival
3.x): one row per covariate (`x1`, `x2`, ... in design order) and a final
`GLOBAL` row. Takes the options of `cox_fit_agg` plus `transform`: `'km'`
(default), `'rank'`, `'identity'` or `'log'`. A small p-value says the effect of
that covariate changes over time.

```sql
SELECT unnest(cox_zph_agg(days, [age, treated], died), recursive := true) FROM patients;
```

## cox_survival

```text
cox_survival(model STRUCT, x DOUBLE[], times DOUBLE[] [, stratum DOUBLE]) -> STRUCT(time DOUBLE, surv DOUBLE)[]
```

The predicted survival curve of the model returned by `cox_fit_agg` for one
covariate row `x` (one value per coefficient), at each of `times`:
`S(t | x) = exp(-H0(t) exp(x'beta))` with the stored Breslow / Efron baseline,
the point estimate of `survfit(fit, newdata)`. It is a step function: `S = 1`
before the first time, and the last value is carried beyond the last time
(`summary(..., extend = TRUE)`). A stratified model needs the stratum value.
The offset is not added, and no interval is returned (the baseline's variance
is not stored in the model).

```sql
WITH f AS (SELECT cox_fit_agg(days, [age, treated], died) AS m FROM patients)
SELECT treated, unnest(cox_survival(m, [55.0, treated], [90, 180, 365]), recursive := true)
FROM f, (VALUES (0.0), (1.0)) v(treated)
ORDER BY treated;
```

## cox_augment_agg / cox_augment_by

```text
cox_augment_agg(time, x, event [, weight] [, options]) OVER (...) -> STRUCT(...)[]
cox_augment_by(source, group_col, time_col, x_cols, event_col [, options := ...] [, order_by := ...])
```

`cox_augment_agg` fits the model over a window frame and returns one STRUCT per
row of the frame, in frame order: `linear_predictor` (`x'beta + offset`, not
centred: R's `predict(fit, type = "lp", reference = "zero")`), `risk`
(`exp(linear_predictor)`), `martingale_residual`, `deviance_residual`,
`score_residuals`, `dfbeta`, `dfbetas`, `schoenfeld_residuals` and
`scaled_schoenfeld_residuals` (event rows only, NULL otherwise), `stratum` and
`is_training`. Rows left out of the fit get NULLs.

`cox_augment_by` runs it per group and returns every source row in the
integration contract's `obs` schema: `fitted` is the linear predictor,
`residual` the martingale residual (`resid_type = 'martingale'`), followed by
the Cox residuals above.

```sql
-- Martingale residuals against age: a curve suggests a non-linear age effect
SELECT round(age / 10) * 10 AS age_band, round(avg(residual), 3) AS mean_martingale
FROM cox_augment_by('patients', site, days, [age, treated], died)
GROUP BY age_band ORDER BY age_band;
```

## cox_fit_by / cox_tidy_by / cox_glance_by

```sql
-- One model per site
SELECT site, hazard_ratios, concordance FROM cox_fit_by('patients', site, days, [age, treated], died);

-- Log hazard ratios with inference, in the `terms` schema
SELECT site, term, round(estimate, 4) AS log_hr, round(exp(estimate), 4) AS hr, round(p_value, 6) AS p
FROM cox_tidy_by('patients', site, days, [age, treated], died, names := ['age', 'treated'])
ORDER BY site, term;

SELECT * FROM cox_glance_by('patients', site, days, [age, treated], died) WHERE metric = 'aic';
```

`cox_tidy_by` reports the coefficients on the log scale (`estimate`); there is
no intercept row. The `*_by` macros take no weight column: pass weights to the
aggregates directly.

## See Also

- [AFT survival regression](aft.md) - parametric survival models
- [Survival curves](survival_curves.md) - Kaplan-Meier, log-rank test
