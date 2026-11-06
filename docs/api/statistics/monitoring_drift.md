# Monitoring: Reconstruction-Error Drift and Alert Bands

Model monitoring without labels, after NannyML. Two building blocks:

- **Multivariate drift by PCA reconstruction error** (NannyML's
  `DataReconstructionDriftCalculator`): fit a PCA on the standardised reference
  rows and keep the components that explain a set share of the variance (default
  0.65). A row's reconstruction error is the euclidean norm of its standardised
  residual outside those components. If the relations between the columns change,
  the mean error of a chunk moves outside the band seen on the reference chunks,
  even when each column's marginal distribution looks the same.
- **Reference-band alert thresholds** for any chunked metric (a drift
  statistic, a PSI, an estimated or realised performance metric): the band is
  derived from the metric's values on the reference chunks (mean ± k·SD, fixed
  limits, or limits relative to the reference mean), with optional limits on the
  metric's range.

The computations come from upstream anofox-statistics `monitoring`
(`fit_reconstruction`, `ReconstructionModel`, `reconstruction_thresholds`,
`threshold_band`), which is validated against NannyML 0.13.1. The extension
checks the results against NannyML on its synthetic car-loan dataset
(`test/sql/reference/reconstruction_reference.test`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `pca_reconstruction_fit_agg` | Aggregate | Fit the reconstruction model on reference rows; returns the model STRUCT |
| `pca_reconstruction_error` | Scalar | Reconstruction error of one row under a model |
| `pca_reconstruction_chunk_agg` | Aggregate | Mean reconstruction error of a chunk, with its sampling error and confidence band |
| `pca_reconstruction_threshold_agg` | Aggregate | Alert band of the reconstruction error from the reference chunks' errors |
| `pca_reconstruction_drift_by` | Table macro | The whole NannyML calculation: one row per chunk, with thresholds and alert |
| `alert_band_agg` | Aggregate | Alert band of any metric from its reference values |
| `alert_band_by` | Table macro | Thresholds and alerts for every chunk of a chunked metric |

**Reference period.** Both table macros take the reference designation as a
BOOLEAN expression `is_reference`, evaluated on every source row (for example
`period = 'reference'` or `ts < DATE '2024-01-01'`). Rows where it is NULL count
as analysis rows.

**Missing values.** A NULL element in `x` is a missing value. As NannyML's
`SimpleImputer(strategy = "mean")`, it is replaced by the reference column mean
(`'impute_nan': false` makes it an error instead). Rows whose whole list is NULL
are skipped. All columns must be numeric: encode categorical columns before you
pass them.

**Alerts.** A chunk alerts when its value is strictly below the lower or above
the upper threshold, as in NannyML. A missing threshold (NULL) never triggers an
alert, a NaN value never alerts, and a NULL value gives a NULL alert.

## pca_reconstruction_fit_agg

**Signature:**

```text
pca_reconstruction_fit_agg(x DOUBLE[] [, options MAP]) -> STRUCT
```

Standardises the columns with the reference means and population standard
deviations (scikit-learn's `StandardScaler`; a constant column keeps scale 1),
fits a PCA and keeps the fewest components whose explained-variance share exceeds
`variance` (scikit-learn's `PCA(n_components = 0.65)`).

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `variance` | DOUBLE | 0.65 | Share of the variance the kept components explain, in (0, 1) |
| `n_components` | INTEGER | - | A fixed number of components (instead of `variance`), at most `min(n - 1, d)` |
| `scale` | BOOLEAN | true | Standardise the columns; `false` only centres them |
| `impute_nan` | BOOLEAN | true | Replace a missing value by the reference column mean |

**Returns:** `STRUCT(center DOUBLE[], scale DOUBLE[] (NULL when unscaled), sdev
DOUBLE[], proportion_of_variance DOUBLE[], loadings DOUBLE[][] (one list per
kept component), n_components BIGINT, n_features BIGINT, n_observations BIGINT,
impute_nan BOOLEAN)`. With fewer than two rows the result is NULL. You can store
the model in a table and use it later: `pca_reconstruction_error` and
`pca_reconstruction_chunk_agg` rebuild it from its fields (upstream
`ReconstructionModel::from_parts`).

**Example:**

```sql
CREATE TABLE loans AS
SELECT i AS id, i // 100 AS chunk, CASE WHEN i < 500 THEN 'reference' ELSE 'analysis' END AS period,
       20000 + 5000 * sin(i) AS car_value,
       0.3 + 0.1 * cos(0.7 * i) AS debt_to_income,
       36 + 12 * sin(1.3 * i) + CASE WHEN i >= 800 THEN 10 * cos(i) ELSE 0 END AS loan_length
FROM range(1000) t(i);

SELECT (m).n_components, list_transform((m).proportion_of_variance, lambda p: round(p, 3)) AS share
FROM (SELECT pca_reconstruction_fit_agg([car_value, debt_to_income, loan_length]) AS m
      FROM loans WHERE period = 'reference');
```

## pca_reconstruction_error

**Signature:**

```text
pca_reconstruction_error(model STRUCT, x DOUBLE[]) -> DOUBLE
```

Reconstruction error of one row under a model. The result is NULL when the
model or `x` is NULL, and it is an error when `x` does not have `n_features`
values.

**Example:**

```sql
WITH m AS (SELECT pca_reconstruction_fit_agg([car_value, debt_to_income, loan_length]) AS model
           FROM loans WHERE period = 'reference')
SELECT id, round(pca_reconstruction_error(model, [car_value, debt_to_income, loan_length]), 4) AS error
FROM loans, m
WHERE id IN (0, 1, 999)
ORDER BY id;
```

## pca_reconstruction_chunk_agg

**Signature:**

```text
pca_reconstruction_chunk_agg(model STRUCT, x DOUBLE[]) -> STRUCT
```

Mean reconstruction error of the group's rows, as NannyML reports it for a
chunk: `STRUCT(n BIGINT, reconstruction_error DOUBLE, std DOUBLE (divisor n - 1;
NULL for one row), sampling_error DOUBLE (std / sqrt(n)), lower_confidence DOUBLE
(max(error - 3 sampling_error, 0)), upper_confidence DOUBLE)`. The model must be
the same on every row of the group, for example from a scalar subquery or a
join.

## pca_reconstruction_threshold_agg

**Signature:**

```text
pca_reconstruction_threshold_agg(value DOUBLE [, threshold MAP]) -> STRUCT(lower_threshold, upper_threshold)
```

The alert band of the reconstruction error, computed from its values on the
reference chunks. The default is NannyML's `StandardDeviationThreshold(3, 3)`. A
lower threshold at or below 0 is dropped (NULL), as NannyML does. The
`threshold` options are `method`, `k`, `lower` and `upper`, the same as for
`alert_band_agg`, but without value limits.

## pca_reconstruction_drift_by

**Signature:**

```text
pca_reconstruction_drift_by(source VARCHAR, chunk_col, x_cols, is_reference
                            [, options := MAP] [, threshold := MAP]) -> TABLE
```

Runs NannyML's calculation on `source`:

1. Fits `pca_reconstruction_fit_agg(x_cols, options)` on the rows where
   `is_reference` is true.
2. Computes `pca_reconstruction_chunk_agg` for every `(chunk_col, is_reference)`
   group.
3. Derives the band with `pca_reconstruction_threshold_agg(..., threshold)` from
   the reference chunks.

Returns one row per `(chunk_col, is_reference)`, reference chunks first, with
the columns `chunk_col`, `is_reference`, `n`, `reconstruction_error`,
`sampling_error`, `lower_confidence`, `upper_confidence`, `lower_threshold`,
`upper_threshold` and `alert`. If a chunk holds both reference and analysis
rows, it appears twice. Without reference rows, the values are NULL.

**Example:**

```sql
SELECT chunk, is_reference, round(reconstruction_error, 4) AS error,
       round(lower_threshold, 4) AS lower, round(upper_threshold, 4) AS upper, alert
FROM pca_reconstruction_drift_by('loans', chunk, [car_value, debt_to_income, loan_length],
                                 period = 'reference', threshold := {'k': 3});
```

## alert_band_agg

**Signature:**

```text
alert_band_agg(value DOUBLE [, options MAP]) -> STRUCT(lower_threshold DOUBLE, upper_threshold DOUBLE)
```

Alert thresholds of a metric, computed from its values on the reference chunks.
It follows NannyML's `StandardDeviationThreshold` / `ConstantThreshold` and
`calculate_threshold_values` (upstream `threshold_band`). A NULL threshold means
there is no threshold on that side. NULL values are skipped and NaN values are
ignored. The `std` and `relative` methods need at least one reference value
(otherwise the result is NULL). `constant` does not use the values.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | VARCHAR | `'std'` | `'std'`: mean − lower·sd and mean + upper·sd (population sd); `'constant'` (alias `'fixed'`): `lower` and `upper` are the thresholds; `'relative'`: mean − lower·\|mean\| and mean + upper·\|mean\| |
| `k` | DOUBLE | 3 | Sets both multipliers (`std`) or shares (`relative`) |
| `lower`, `upper` | DOUBLE | 3 (`std`), none | One side's multiplier, share or constant; NULL disables that side |
| `lower_limit`, `upper_limit` | DOUBLE | none | The metric's range (e.g. 0 and 1 for accuracy). A threshold beyond the limit is set to the limit |
| `override_with_none` | BOOLEAN | false | Drop a threshold beyond its limit (NULL) instead of clamping it |

**Example:**

```sql
CREATE TABLE chunk_metrics AS SELECT * FROM (VALUES
    (1, 0.91), (2, 0.93), (3, 0.92), (4, 0.95), (5, 0.90), (6, 0.84), (7, 0.99)) t(chunk, accuracy);

SELECT round((b).lower_threshold, 4) AS lower, round((b).upper_threshold, 4) AS upper
FROM (SELECT alert_band_agg(accuracy, {'lower_limit': 0, 'upper_limit': 1}) AS b
      FROM chunk_metrics WHERE chunk <= 5);
```

## alert_band_by

**Signature:**

```text
alert_band_by(source VARCHAR, chunk_col, metric_col, is_reference [, options := MAP]) -> TABLE
```

Takes `source` with one row per chunk. It computes `alert_band_agg(metric_col,
options)` over the rows where `is_reference` is true, and returns every row with
`chunk_col`, `value`, `lower_threshold`, `upper_threshold`, `alert` and
`is_reference`, reference chunks first. Use it on the output of any chunked
calculation, for example a per-chunk `psi_agg` or a realised metric.

**Example:**

```sql
SELECT chunk, value, round(lower_threshold, 4) AS lower, round(upper_threshold, 4) AS upper, alert
FROM alert_band_by('chunk_metrics', chunk, accuracy, chunk <= 5, options := {'k': 2});

SELECT chunk, alert
FROM alert_band_by('chunk_metrics', chunk, accuracy, chunk <= 5,
                   options := {'method': 'constant', 'lower': 0.88});
```
