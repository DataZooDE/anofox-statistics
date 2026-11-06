# Model Validation and Calibration

Resampling estimates of how well a regression model will predict new data, after
Harrell's *Regression Modeling Strategies* (§5.3, §10.11): the optimism of the apparent
indexes, the overfitting-corrected calibration curve and the bootstrap covariance of the
coefficients. Results follow R's `rms::validate`, `rms::calibrate`, `rms::calibrate.cph`
and `rms::bootcov`; the resampling engine and every index are anofox-regression's
`validation` module.

## Functions

| Function | Type | Description | R reference |
|----------|------|-------------|-------------|
| `validate_agg` | Aggregate | Optimism-corrected indexes (bootstrap, cross-validation, .632) | `validate(fit, method, B)` |
| `calibrate_agg` | Aggregate | Overfitting-corrected calibration curve | `calibrate(fit, B, predy)` |
| `calibrate_cph_agg` | Aggregate | Calibration of a Cox model's survival at time `u` against Kaplan-Meier | `calibrate(cph_fit, cmethod = "KM", u, m, B)` |
| `bootcov_agg` | Aggregate | Bootstrap covariance, standard errors and percentile intervals of the coefficients (ordinary or cluster bootstrap) | `bootcov(fit, B, cluster)` |
| `validate_by` | Table macro | `validate_agg` per group, one row per index | |
| `calibrate_by` | Table macro | `calibrate_agg` per group (integration contract `curve`) | |
| `calibrate_cph_by` | Table macro | `calibrate_cph_agg` per group | |
| `bootcov_by` | Table macro | `bootcov_agg` per group (integration contract `terms`) | |

**Models.** `validate_agg`, `calibrate_agg` and `bootcov_agg` fit, with an intercept, a
linear model of `y` on the columns of `x` (`rms::ols`, `'model': 'ols'`, the default) or a
binary logistic model (`rms::lrm`, `'model': 'logistic'`, `y` in {0, 1}).
`calibrate_cph_agg` fits a Cox model (`rms::cph`).

**Resamples and seeds.** The resamples are drawn from `seed` (default 42) by upstream's
SplitMix64 generator. The rows are first sorted into a canonical order, so the same seed
gives the same result whatever the thread count or the order of the rows. The draws are
not R's: the same seed in R gives other resamples, so expect Monte-Carlo differences from
an `rms` run with `set.seed()` unless B is large. The reference tests replay these
resamples in `rms` and match it exactly.

**NULL handling.** Rows with a NULL or non-finite `y`, `x` element, `time` or `event` are
skipped. Data that cannot support the result (no rows, too few rows, a model that cannot
be fitted on all rows, more cross-validation groups than `n / 2`, no events) give NULL.
Invalid options raise an error.

The examples on this page use this table:

```sql
-- One row per patient: a continuous outcome, a binary outcome, follow-up and a clinic
CREATE OR REPLACE TABLE patients AS
SELECT id, site, clinic, age, sbp,
       10 + 0.3 * age - 0.05 * sbp + ((id * 7) % 13) / 2.0 AS score,
       CASE WHEN 0.06 * age + 0.02 * sbp + ((id * 37) % 23) / 6.0 > 7.5 THEN 1 ELSE 0 END AS died,
       least(raw_days, 400.0) AS days,
       CASE WHEN raw_days <= 400 THEN 1.0 ELSE 0.0 END AS event
FROM (
    SELECT i AS id, (i % 2)::VARCHAR AS site, i // 6 AS clinic,
           (40 + (i * 7) % 35)::DOUBLE AS age,
           (110 + (i * 13) % 50)::DOUBLE AS sbp,
           round(25 + 300 * exp(-0.03 * ((40 + (i * 7) % 35) - 55)) * (0.05 + ((i * 37) % 101) / 101.0), 1)
               AS raw_days
    FROM range(300) r(i)
);
```

## validate_agg

**Signature:**

```text
validate_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> LIST(STRUCT)
```

Each index is computed on the model fitted to all rows (`index_orig`, the apparent
index), on each training sample (`training`) and on the matching test rows (`test`); the
optimism is `training - test` (`.632`: `0.632 (index_orig - test)`, `test` the weighted
sum over the left-out rows) and `index_corrected = index_orig - optimism`.

| Field | Type | Description |
|-------|------|-------------|
| `index_name` | VARCHAR | ols: `R-square`, `MSE`, `g`, `Intercept`, `Slope`; logistic: `Dxy`, `R2`, `Intercept`, `Slope`, `Emax`, `D`, `U`, `Q`, `B` (Brier), `g`, `gp` |
| `index_orig` | DOUBLE | Apparent index |
| `training` | DOUBLE | Mean index on the training samples |
| `test` | DOUBLE | Mean index on the test rows |
| `optimism` | DOUBLE | Optimism |
| `index_corrected` | DOUBLE | Bias-corrected index |
| `n` | BIGINT | Resamples with a finite index |
| `n_failed` | BIGINT | Resamples whose fit failed (skipped) |
| `model` | VARCHAR | `'ols'` or `'logistic'` |
| `method` | VARCHAR | `'boot'`, `'crossvalidation'` or `'.632'` |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | VARCHAR | `'ols'` | `'ols'` (alias `'linear'`) or `'logistic'` (alias `'lrm'`) |
| `method` | VARCHAR | `'boot'` | `'boot'` (Efron-Gong optimism bootstrap), `'crossvalidation'` (alias `'cv'`) or `'.632'` |
| `B` | INTEGER | 40 | Bootstrap repetitions, or the number of cross-validation groups (`n_boot` is an alias) |
| `seed` | INTEGER | 42 | Seed of the resamples (`random_state` is an alias) |

**Example:**

```sql
-- Optimism-corrected discrimination and calibration slope of a logistic model
SELECT r.index_name, round(r.index_orig, 3) AS apparent, round(r.optimism, 3) AS optimism,
       round(r.index_corrected, 3) AS corrected
FROM (SELECT unnest(validate_agg(died, [age, sbp], {'model': 'logistic', 'B': 200, 'seed': 1})) AS r
      FROM patients)
WHERE r.index_name IN ('Dxy', 'R2', 'Slope', 'B');
```

```sql
-- 10-fold cross-validation of a linear model
SELECT r.index_name, r.index_corrected
FROM (SELECT unnest(validate_agg(score, [age, sbp], {'method': 'crossvalidation', 'B': 10})) AS r FROM patients);
```

## calibrate_agg

**Signature:**

```text
calibrate_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> LIST(STRUCT)
```

The apparent calibration curve is the lowess smooth (`f = 2/3`, no robustness
iterations) of `y` on the model's prediction (the probability for a logistic model),
evaluated on a grid `predy`; its optimism is estimated by refitting the model on every
resample.

| Field | Type | Description |
|-------|------|-------------|
| `predy` | DOUBLE | Grid point (predicted value) |
| `calibrated_orig` | DOUBLE | Apparent calibration at `predy` |
| `calibrated_corrected` | DOUBLE | Bias-corrected calibration |
| `optimism` | DOUBLE | Optimism at `predy` |
| `n` | BIGINT | Resamples with a value at `predy` |
| `mean_abs_error` | DOUBLE | Mean absolute error of the corrected curve over the predictions (`print.calibrate`'s "Mean absolute error") |
| `mean_squared_error` | DOUBLE | Mean squared error |
| `quantile90_abs_error` | DOUBLE | 0.9 quantile of the absolute error |
| `model`, `method` | VARCHAR | As `validate_agg` |

**Options:** `model`, `method`, `B` (default 40), `seed` as `validate_agg`, and `predy`
(DOUBLE[]): the grid; by default 50 equally spaced points between the 5th smallest and
the 5th largest prediction (which needs more than 10 rows). A grid point outside the
range of the predictions has no lowess value: its calibration is NULL.

**Example:**

```sql
-- Bias-corrected calibration of predicted risk at a few probabilities
SELECT round(r.predy, 2) AS predicted, round(r.calibrated_orig, 3) AS apparent,
       round(r.calibrated_corrected, 3) AS corrected
FROM (SELECT unnest(calibrate_agg(died, [age, sbp],
                    {'model': 'logistic', 'B': 100, 'predy': [0.3, 0.45, 0.6, 0.75]})) AS r
      FROM patients);
```

## calibrate_cph_agg

**Signature:**

```text
calibrate_cph_agg(time DOUBLE, x DOUBLE[], event DOUBLE, options MAP) -> LIST(STRUCT)
```

The Cox model's predicted survival at time `u` groups the subjects by its quantiles (about
`m` per group, or the given `cuts`); each group's Kaplan-Meier estimate at `u` is compared
with its mean prediction, and the difference (`index_orig = km - mean_predicted`) is
corrected for overfitting by refitting the Cox model on every resample (refits with fewer
than five events are skipped). `event` is 1 for an event and 0 for censoring.

| Field | Type | Description |
|-------|------|-------------|
| `group_index` | BIGINT | Group, 1-based, in increasing predicted survival |
| `cut_low`, `cut_high` | DOUBLE | Group boundaries on the predicted-survival scale |
| `mean_predicted` | DOUBLE | Mean predicted survival at `u` |
| `km` | DOUBLE | Kaplan-Meier survival at `u` |
| `km_corrected` | DOUBLE | `km - optimism` |
| `std_err` | DOUBLE | Standard error of `log km` (Greenwood) |
| `index_orig`, `training`, `test`, `optimism`, `index_corrected`, `n` | | The resampling table of `km - mean_predicted` |
| `u` | DOUBLE | The time horizon |
| `method` | VARCHAR | The resampling method |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `u` | DOUBLE | required | Time at which survival is calibrated |
| `m` | INTEGER | 50 | Subjects per group |
| `cuts` | DOUBLE[] | | Group boundaries on the survival scale (instead of `m`) |
| `method`, `B`, `seed` | | `'boot'`, 40, 42 | As `validate_agg` |
| `ties` | VARCHAR | `'efron'` | `'efron'` or `'breslow'` |
| `max_iterations` | INTEGER | 20 | Newton iterations of the Cox fits (`rms::cph` uses 10) |
| `tolerance` | DOUBLE | 1e-9 | Convergence tolerance (`rms::cph` uses 1e-4) |

**Example:**

```sql
-- Is predicted 200-day survival calibrated?
SELECT r.group_index, round(r.mean_predicted, 3) AS predicted, round(r.km, 3) AS observed,
       round(r.km_corrected, 3) AS observed_corrected
FROM (SELECT unnest(calibrate_cph_agg(days, [age, sbp], event, {'u': 200, 'm': 75, 'B': 50})) AS r
      FROM patients);
```

## bootcov_agg

**Signature:**

```text
bootcov_agg(y DOUBLE, x DOUBLE[] [, cluster ANY] [, options MAP]) -> LIST(STRUCT)
```

Refits the model on `B` bootstrap samples of the rows, or with `cluster` (any type, one
id per row) on samples of whole clusters (`bootcov(cluster = id)`, for repeated
measures). The options argument is required with a cluster.

| Field | Type | Description |
|-------|------|-------------|
| `term` | VARCHAR | `(Intercept)`, then `x1`, `x2`, ... (or `feature_names`) |
| `estimate` | DOUBLE | Coefficient of the fit on all rows |
| `std_error` | DOUBLE | Bootstrap standard error |
| `conf_low`, `conf_high` | DOUBLE | Percentile interval (type-7 quantiles of the replicates) |
| `conf_level` | DOUBLE | Interval level |
| `boot_mean` | DOUBLE | Mean of the replicates |
| `vcov` | DOUBLE[] | The term's row of the bootstrap covariance (divisor `n_boot - 1`) |
| `n_boot` | BIGINT | Successful replicates |
| `n_failed` | BIGINT | Replicates whose fit failed (skipped) |
| `model` | VARCHAR | `'ols'` or `'logistic'` |

**Options:** `model`, `B` (default 200, as `rms::bootcov`), `seed` (42), `conf_level`
(0.95), `feature_names`.

**Example:**

```sql
-- Bootstrap standard errors of a logistic model, resampling whole clinics
SELECT r.term, round(r.estimate, 4) AS estimate, round(r.std_error, 4) AS boot_se,
       round(r.conf_low, 4) AS conf_low, round(r.conf_high, 4) AS conf_high
FROM (SELECT unnest(bootcov_agg(died, [age, sbp], clinic,
                    {'model': 'logistic', 'B': 200, 'feature_names': ['age', 'sbp']})) AS r
      FROM patients);
```

## Table macros

```text
validate_by(source, group_col, y_col, x_cols, options := NULL)
calibrate_by(source, group_col, y_col, x_cols, options := NULL)
calibrate_cph_by(source, group_col, time_col, x_cols, event_col, options := NULL)
bootcov_by(source, group_col, y_col, x_cols, options := NULL, cluster := NULL, names := NULL)
```

Each runs its aggregate per group of `source` and returns one row per element of the
aggregate's list, after the group column. `calibrate_by` uses the integration contract's
`curve` schema (`id`, `model_id` `'calibrate'`, `model_name` the model, `curve_type`
`'calibrate'`, `x` = `predy`, `y` = `calibrated_corrected`, ..., then the
`calibrate_agg` fields and `row_order`); `bootcov_by` the `terms` schema (`model_id`
`'bootcov'`, `statistic` and `p_value` NULL, then `boot_mean`, `vcov`, `n_boot`,
`n_failed`, `row_order`).

```sql
SELECT site, index_name, round(index_corrected, 3) AS corrected
FROM validate_by('patients', site, died, [age, sbp], options := {'model': 'logistic', 'B': 100})
WHERE index_name IN ('Dxy', 'Slope');
```

```sql
SELECT site, round(x, 2) AS predicted, round(y, 3) AS corrected
FROM calibrate_by('patients', site, score, [age, sbp], options := {'B': 50})
WHERE row_order % 10 = 1;
```

```sql
SELECT site, group_index, round(km - mean_predicted, 3) AS apparent_error
FROM calibrate_cph_by('patients', site, days, [age, sbp], event, options := {'u': 200, 'm': 50, 'B': 20});
```

```sql
SELECT site, term, round(std_error, 4) AS boot_se
FROM bootcov_by('patients', site, score, [age, sbp], options := {'B': 100}, names := ['age', 'sbp']);
```
