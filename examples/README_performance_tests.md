# Performance Testing Guide

This directory contains comprehensive performance tests for the anofox_statistics extension, with equivalent implementations in both SQL (DuckDB) and R to enable direct comparison.

## Overview

The performance testing suite consists of:

1. **Data Generation**: Creates test datasets as parquet files
2. **SQL Tests**: Tests DuckDB extension functions on the data
3. **R Tests**: Performs equivalent analysis using R's `lm()` function
4. **Results Comparison**: All results saved as parquet for analysis

## Directory Structure

```
examples/
├── generate_test_data.sql                    # Generate test datasets
├── performance_test_ols_fit_predict.sql      # SQL window function tests
├── performance_test_ols_fit_predict.R        # R equivalent tests
├── performance_test_ols_aggregate.sql        # SQL aggregate function tests
├── performance_test_ols_aggregate.R          # R equivalent tests
├── data/                                     # Generated test data (parquet)
│   ├── performance_data_fit_predict.parquet
│   └── performance_data_aggregate.parquet
└── results/                                  # Test results (parquet)
    ├── sql_predictions_expanding_single.parquet
    ├── sql_predictions_fixed_single.parquet
    ├── sql_predictions_expanding_multi.parquet
    ├── sql_predictions_fixed_multi.parquet
    ├── sql_group_models.parquet
    ├── sql_group_models_full.parquet
    ├── r_predictions_expanding_single.parquet
    ├── r_predictions_fixed_single.parquet
    ├── r_predictions_expanding_multi.parquet
    ├── r_predictions_fixed_multi.parquet
    ├── r_group_models.parquet
    └── r_group_models_full.parquet
```

## Quick Start

### 1. Generate Test Data

First, generate the test datasets that will be used by both SQL and R scripts:

```bash
# Using DuckDB CLI
duckdb < examples/generate_test_data.sql
```

This creates two parquet files in `examples/data/`:
- `performance_data_fit_predict.parquet` - For window function tests (includes NULL values)
- `performance_data_aggregate.parquet` - For GROUP BY aggregate tests (no NULL values)

**Configuration Parameters** (editable in the script):
- `n_groups`: 10,000 (number of groups)
- `n_obs_per_group`: 100 (observations per group)
- `n_features`: 8 (number of features x1-x8)
- `noise_std`: 2.0 (standard deviation of noise)
- `null_fraction`: 0.1 (fraction of NULL values for fit-predict dataset)

### 2. Run SQL Performance Tests

#### Fit-Predict Window Functions

Tests `anofox_statistics_ols_fit_predict` with both expanding and fixed windows:

```bash
duckdb < examples/performance_test_ols_fit_predict.sql
```

**Tests performed**:
1. Expanding window - single group (group 1)
2. Fixed window - single group (group 1)
3. Expanding window - 100 groups
4. Fixed window - 100 groups
5. Expanding window - all groups (10,000)
6. Fixed window - all groups (10,000)

**Results saved to** `examples/results/`:
- `sql_predictions_expanding_single.parquet`
- `sql_predictions_fixed_single.parquet`
- `sql_predictions_expanding_multi.parquet`
- `sql_predictions_fixed_multi.parquet`

#### Aggregate Functions with GROUP BY

Tests `anofox_statistics_ols_fit_agg` with GROUP BY:

```bash
duckdb < examples/performance_test_ols_aggregate.sql
```

**Tests performed**:
1. GROUP BY on all groups (basic output)
2. GROUP BY on all groups (full statistical output)
3. GROUP BY on subset (100 groups)

**Results saved to** `examples/results/`:
- `sql_group_models.parquet`
- `sql_group_models_full.parquet`

### 3. Run R Performance Tests

**Prerequisites**: Install required R packages:

```r
install.packages(c("arrow", "dplyr", "broom"))
```

#### Fit-Predict Tests (R)

Run equivalent fit-predict analysis using R's `lm()`:

```bash
Rscript examples/performance_test_ols_fit_predict.R
```

**Results saved to** `examples/results/`:
- `r_predictions_expanding_single.parquet`
- `r_predictions_fixed_single.parquet`
- `r_predictions_expanding_multi.parquet`
- `r_predictions_fixed_multi.parquet`

#### Aggregate Tests (R)

Run equivalent GROUP BY analysis using R's `lm()`:

```bash
Rscript examples/performance_test_ols_aggregate.R
```

**Results saved to** `examples/results/`:
- `r_group_models.parquet`
- `r_group_models_full.parquet`

## Comparing SQL vs R Results

All results are saved as parquet files, making it easy to compare SQL and R implementations:

### Example: Compare predictions

```sql
-- Load both SQL and R predictions
CREATE TABLE sql_pred AS
  SELECT * FROM 'examples/results/sql_predictions_expanding_single.parquet';

CREATE TABLE r_pred AS
  SELECT * FROM 'examples/results/r_predictions_expanding_single.parquet';

-- Compare predictions
SELECT
    s.obs_id,
    s.yhat as sql_yhat,
    r.yhat as r_yhat,
    ABS(s.yhat - r.yhat) as diff,
    CASE
        WHEN ABS(s.yhat - r.yhat) < 0.001 THEN 'MATCH'
        ELSE 'DIFFER'
    END as status
FROM sql_pred s
JOIN r_pred r ON s.obs_id = r.obs_id
ORDER BY obs_id;
```

### Example: Compare model coefficients

```sql
-- Load both SQL and R models
CREATE TABLE sql_models AS
  SELECT * FROM 'examples/results/sql_group_models.parquet';

CREATE TABLE r_models AS
  SELECT * FROM 'examples/results/r_group_models.parquet';

-- Compare coefficients for group 1
SELECT
    'Intercept' as parameter,
    s.intercept as sql_value,
    r.intercept as r_value,
    ABS(s.intercept - r.intercept) as diff
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1

UNION ALL

SELECT
    'Coefficient x1',
    s.coefficients[1],
    r.coef_x1,
    ABS(s.coefficients[1] - r.coef_x1)
FROM sql_models s
JOIN r_models r ON s.group_id = r.group_id
WHERE s.group_id = 1;
-- ... etc for x2-x8
```

## Test Data Schema

### Fit-Predict Dataset

**File**: `performance_data_fit_predict.parquet`

| Column | Type | Description |
|--------|------|-------------|
| group_id | INTEGER | Group identifier (1 to n_groups) |
| obs_id | INTEGER | Sequential observation ID within group |
| x1-x8 | DOUBLE | Feature columns (uniform random -10 to 10) |
| y | DOUBLE | Response variable (with ~10% NULL values) |
| y_true | DOUBLE | True response (no NULLs, for validation) |
| beta_0-beta_8 | DOUBLE | True coefficients for validation |

### Aggregate Dataset

**File**: `performance_data_aggregate.parquet`

| Column | Type | Description |
|--------|------|-------------|
| group_id | INTEGER | Group identifier (1 to n_groups) |
| obs_id | INTEGER | Observation ID within group |
| x1-x8 | DOUBLE | Feature columns (uniform random -10 to 10) |
| y | DOUBLE | Response variable (no NULL values) |
| beta_0-beta_8 | DOUBLE | True coefficients for validation |

## Data Generation Process

The test data simulates realistic scenarios:

1. **Heterogeneous Groups**: Each group has different true coefficients
   - Intercept β₀: random uniform(-10, 10)
   - Slopes β₁-β₈: random uniform(-5, 5)

2. **Linear Relationship with Noise**:
   ```
   y = β₀ + β₁x₁ + β₂x₂ + ... + β₈x₈ + ε
   ```
   where ε ~ N(0, noise_std²)

3. **NULL Values** (fit-predict only):
   - Approximately 10% of y values set to NULL
   - Used to test prediction functionality

## Performance Metrics

Both SQL and R scripts output timing information:

- **SQL**: Uses `.timer on` to measure execution time
- **R**: Uses `Sys.time()` to measure elapsed time

Compare performance across:
- Different window sizes (single group vs 100 groups vs all groups)
- Different modes (expanding vs fixed windows)
- SQL vs R implementations

## Validation Tests

Both test suites include validation:

1. **Coefficient Recovery**: Compare estimated coefficients to true values
2. **Prediction Accuracy**: Mean absolute error on NULL values
3. **Coverage**: Percentage of true values within prediction intervals
4. **Mode Comparison**: Fixed vs expanding window predictions

## Notes

- **Memory**: Running tests on all 10,000 groups requires significant memory
- **Time**: Full test suite can take several minutes to complete
- **Reproducibility**: Data generation uses random numbers - results will differ between runs
- **Consistency**: Use the same parquet files for SQL and R to ensure fair comparison

## Troubleshooting

**Issue**: File not found errors
- **Solution**: Ensure you run `generate_test_data.sql` first
- **Solution**: Check that `examples/data/` and `examples/results/` directories exist

**Issue**: R package errors
- **Solution**: Install required packages: `install.packages(c("arrow", "dplyr", "broom"))`

**Issue**: Out of memory
- **Solution**: Reduce `n_groups` or `n_obs_per_group` in `generate_test_data.sql`

**Issue**: Parquet write errors
- **Solution**: Ensure write permissions in `examples/data/` and `examples/results/`
