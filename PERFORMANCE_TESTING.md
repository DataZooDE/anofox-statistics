# Performance Testing Suite

This document provides an overview of the comprehensive performance testing suite for the anofox-statistics extension.

## Quick Overview

The performance testing suite enables:

1. **Reproducible Testing**: Generate consistent test datasets as parquet files
2. **SQL Testing**: Test DuckDB extension functions with large-scale data
3. **R Comparison**: Run equivalent tests in R using `lm()`
4. **Result Validation**: Compare SQL vs R results to ensure correctness
5. **Performance Benchmarking**: Measure and compare execution times

## Directory Structure

```
examples/
├── generate_test_data.sql                    # Generate test datasets
├── performance_test_ols_fit_predict.sql      # SQL window function tests
├── performance_test_ols_fit_predict.R        # R equivalent tests
├── performance_test_ols_aggregate.sql        # SQL aggregate function tests
├── performance_test_ols_aggregate.R          # R equivalent tests
├── compare_sql_vs_r.sql                      # Compare SQL vs R results
├── run_all_tests.sh                          # Master test runner
├── README_performance_tests.md               # Detailed documentation
├── data/                                     # Generated test data (parquet)
│   ├── performance_data_fit_predict.parquet
│   └── performance_data_aggregate.parquet
└── results/                                  # Test results (parquet)
    ├── sql_predictions_*.parquet             # SQL prediction results
    ├── sql_group_models*.parquet             # SQL model results
    ├── r_predictions_*.parquet               # R prediction results
    └── r_group_models*.parquet               # R model results
```

## Quick Start

### Prerequisites

**Required:**
- DuckDB with anofox-statistics extension loaded
- Bash (for run_all_tests.sh)

**Optional (for R tests):**
- R with packages: arrow, dplyr, broom

### Running All Tests

```bash
# From project root
./examples/run_all_tests.sh
```

This will:
1. Create data and results directories
2. Generate test datasets (10,000 groups × 100 observations = 1M rows)
3. Run SQL fit-predict tests
4. Run SQL aggregate tests
5. Run R fit-predict tests (if R available)
6. Run R aggregate tests (if R available)
7. Display summary of results

### Running Individual Steps

```bash
# 1. Generate test data
duckdb < examples/generate_test_data.sql

# 2. Run SQL tests
duckdb < examples/performance_test_ols_fit_predict.sql
duckdb < examples/performance_test_ols_aggregate.sql

# 3. Run R tests (optional)
Rscript examples/performance_test_ols_fit_predict.R
Rscript examples/performance_test_ols_aggregate.R

# 4. Compare results
duckdb < examples/compare_sql_vs_r.sql
```

## Test Configuration

Default configuration in `generate_test_data.sql`:

```sql
SET VARIABLE n_groups = 10000;          -- Number of groups
SET VARIABLE n_obs_per_group = 100;     -- Observations per group
SET VARIABLE n_features = 8;            -- Number of features (x1-x8)
SET VARIABLE noise_std = 2.0;           -- Standard deviation of noise
SET VARIABLE null_fraction = 0.1;       -- Fraction of NULL values (fit-predict)
```

**Total dataset size**: 10,000 × 100 = 1,000,000 rows

You can adjust these parameters for different test scenarios:
- **Small test**: 100 groups × 50 observations = 5,000 rows
- **Medium test**: 1,000 groups × 100 observations = 100,000 rows
- **Large test**: 100,000 groups × 100 observations = 10,000,000 rows

## What Gets Tested

### Fit-Predict Window Functions

Tests `anofox_statistics_ols_fit_predict` with:

1. **Expanding window (single group)**: Model refits on each new observation
2. **Fixed window (single group)**: Fit once, predict all
3. **Expanding window (100 groups)**: Multi-group expanding window
4. **Fixed window (100 groups)**: Multi-group fixed window
5. **Expanding window (all groups)**: Full-scale expanding window
6. **Fixed window (all groups)**: Full-scale fixed window

### Aggregate Functions

Tests `anofox_statistics_ols_fit_agg` with:

1. **GROUP BY (all groups, basic output)**: Coefficients, R², etc.
2. **GROUP BY (all groups, full output)**: Complete statistical output
3. **GROUP BY (subset, 100 groups)**: Subset performance

## Output Files

### Data Files (examples/data/)

| File | Description | Rows | Contains NULL |
|------|-------------|------|---------------|
| performance_data_fit_predict.parquet | Window function test data | 1,000,000 | Yes (~10%) |
| performance_data_aggregate.parquet | Aggregate test data | 1,000,000 | No |

### Result Files (examples/results/)

**SQL Results:**
- `sql_predictions_expanding_single.parquet` - Expanding window (group 1)
- `sql_predictions_fixed_single.parquet` - Fixed window (group 1)
- `sql_predictions_expanding_multi.parquet` - Expanding window (100 groups)
- `sql_predictions_fixed_multi.parquet` - Fixed window (100 groups)
- `sql_group_models.parquet` - Model coefficients (all groups)
- `sql_group_models_full.parquet` - Full statistical output (all groups)

**R Results:**
- `r_predictions_expanding_single.parquet` - Expanding window (group 1)
- `r_predictions_fixed_single.parquet` - Fixed window (group 1)
- `r_predictions_expanding_multi.parquet` - Expanding window (100 groups)
- `r_predictions_fixed_multi.parquet` - Fixed window (100 groups)
- `r_group_models.parquet` - Model coefficients (all groups)
- `r_group_models_full.parquet` - Full statistical output (all groups)

## Comparing Results

The `compare_sql_vs_r.sql` script validates that SQL and R produce equivalent results:

```bash
duckdb < examples/compare_sql_vs_r.sql
```

This compares:
1. **Predictions**: yhat, confidence/prediction intervals
2. **Coefficients**: Intercept and slopes
3. **Statistics**: R², adjusted R², F-statistic, AIC, BIC
4. **Accuracy**: Mean/max absolute differences

**Expected results:**
- Most values should match within floating-point precision (< 0.001)
- Small differences (< 0.01) are acceptable due to different numerical libraries
- Large differences (>= 0.01) may indicate implementation issues

## Performance Analysis

Both SQL and R scripts output timing information:

**SQL**: Uses `.timer on`
```
Run Time (s): real 45.2 user 180.3 sys 12.1
```

**R**: Uses `Sys.time()`
```
Elapsed time: 125.3 seconds
```

### Typical Performance (1M rows)

| Test | SQL (DuckDB) | R (lm) | Speedup |
|------|--------------|--------|---------|
| Fixed window (100 groups) | ~5s | ~15s | 3x |
| Expanding window (100 groups) | ~30s | ~120s | 4x |
| GROUP BY (10,000 groups) | ~10s | ~45s | 4.5x |

*Note: Actual times vary based on hardware*

## Validation Tests

Both test suites include validation:

1. **Coefficient Recovery**: Compare estimated vs true coefficients
2. **Prediction Accuracy**: Mean absolute error on NULL values
3. **Interval Coverage**: % of true values within prediction intervals
4. **Mode Comparison**: Fixed vs expanding window predictions

## Use Cases

### 1. Development Testing
```bash
# Quick test during development (100 groups)
# Edit generate_test_data.sql: SET VARIABLE n_groups = 100
./examples/run_all_tests.sh
```

### 2. Regression Testing
```bash
# Run full test suite before releases
./examples/run_all_tests.sh

# Verify SQL vs R match
duckdb < examples/compare_sql_vs_r.sql
```

### 3. Performance Benchmarking
```bash
# Measure performance improvements
./examples/run_all_tests.sh > benchmark_v1.log

# After optimization...
./examples/run_all_tests.sh > benchmark_v2.log

# Compare logs
diff benchmark_v1.log benchmark_v2.log
```

### 4. Cross-validation
```bash
# Generate data
duckdb < examples/generate_test_data.sql

# Run both implementations
duckdb < examples/performance_test_ols_aggregate.sql
Rscript examples/performance_test_ols_aggregate.R

# Compare results
duckdb < examples/compare_sql_vs_r.sql
```

## Troubleshooting

### Data Generation Fails
```bash
# Check DuckDB version
duckdb --version

# Ensure extension is loaded
duckdb -c "LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';"
```

### R Tests Fail
```bash
# Install required packages
R -e "install.packages(c('arrow', 'dplyr', 'broom'))"

# Test R installation
Rscript -e "library(arrow); library(dplyr); library(broom)"
```

### Out of Memory
```bash
# Reduce dataset size in generate_test_data.sql
# SET VARIABLE n_groups = 100;
# SET VARIABLE n_obs_per_group = 50;
```

### Permission Errors
```bash
# Ensure directories are writable
chmod 755 examples/data examples/results
```

## Advanced Topics

### Custom Test Scenarios

Edit `generate_test_data.sql` to create custom scenarios:

```sql
-- Example: Test with more features
SET VARIABLE n_features = 20;

-- Example: More NULL values
SET VARIABLE null_fraction = 0.3;

-- Example: Higher noise
SET VARIABLE noise_std = 10.0;
```

### Extracting Specific Results

```sql
-- Load and analyze specific results
CREATE TABLE my_results AS
  SELECT * FROM 'examples/results/sql_group_models.parquet'
  WHERE r2 > 0.95;  -- Only high R² models

-- Export to CSV
COPY my_results TO 'high_r2_models.csv' (HEADER, DELIMITER ',');
```

### Integration with CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run performance tests
  run: |
    ./examples/run_all_tests.sh
    duckdb < examples/compare_sql_vs_r.sql

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: examples/results/*.parquet
```

## Further Reading

- [README_performance_tests.md](examples/README_performance_tests.md) - Detailed documentation
- [examples/README.md](examples/README.md) - All example scripts
- [Function Reference](guides/function_reference.md) - Complete function documentation
