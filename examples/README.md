# anofox_stats Examples

This directory contains SQL and Python examples demonstrating the `anofox_stats` DuckDB extension for statistical modeling.

## Directory Structure

```
examples/
├── README.md                          # This file
├── pyproject.toml                     # Python dependencies for uv
├── model_prediction_demo.sql          # SQL demo: fit, predict, window functions
├── model_prediction_demo.py           # Python demo: equivalent to SQL demo
├── example-fit-predict-ols.sql        # Extended OLS window function examples
│
├── ols_single_series.sql              # Single series OLS: fit, inference, prediction
├── ols_multiple_series.sql            # GROUP BY regression: per-group models
├── ols_window_functions.sql           # Window functions: expanding, rolling, partitioned
├── ols_predict_agg.sql                # Predict aggregate: fit once, predict all
├── ols_inference.sql                  # Statistical inference: t-tests, p-values, CIs
├── ols_diagnostics.sql                # Diagnostics: VIF, Jarque-Bera, AIC/BIC
│
├── performance_10k_groups_R/          # DuckDB vs R comparison (10K groups)
│   ├── README.md
│   ├── generate_test_data.sql
│   ├── performance_test_ols_fit_predict.sql
│   ├── performance_test_ols_fit_predict.R
│   ├── performance_test_ols_aggregate.sql
│   ├── performance_test_ols_aggregate.R
│   ├── compare_sql_vs_r.sql
│   └── run_all_tests.sh
└── performance_1m_groups/             # Scale benchmarks (1M groups)
    ├── README.md
    ├── benchmark_ols.sql
    ├── benchmark_ridge.sql
    ├── benchmark_wls.sql
    ├── benchmark_rls.sql
    ├── benchmark_elasticnet.sql
    ├── benchmark_ols_predict_agg.sql
    └── run_all_benchmarks.sh
```

## Prerequisites

### DuckDB Extension

Build and install the extension from the project root:

```bash
make release
```

The extension binary is located at `build/release/extension/anofox_stats/anofox_stats.duckdb_extension`.

### Python Environment (Optional)

For Python examples, use `uv` to create a virtual environment:

```bash
cd examples
uv venv
uv pip install -e .
```

Or install dependencies directly:

```bash
uv pip install duckdb pandas
```

## Quick Start Demos

### SQL Demo: model_prediction_demo.sql

Demonstrates the core workflow: fitting an OLS model via aggregation, making predictions, and using window functions.

**Functions demonstrated:**
- `anofox_stats_ols_fit_agg`: Aggregate function for GROUP BY model fitting
- `anofox_stats_ols_fit_predict`: Window function for expanding/rolling predictions
- Manual prediction using stored coefficients: `intercept + coef[1]*x1 + coef[2]*x2 + ...`

**Run:**

```bash
./build/release/duckdb < examples/model_prediction_demo.sql
```

**Output:** Model coefficients, R-squared, predictions for new scenarios, and expanding window predictions.

### Python Demo: model_prediction_demo.py

Equivalent functionality in Python using the `duckdb` package.

**Run:**

```bash
cd examples
uv run model_prediction_demo.py
```

Or with system Python:

```bash
python examples/model_prediction_demo.py
```

**Note:** The extension must be loadable from the DuckDB search path or the working directory.

### Extended Demo: example-fit-predict-ols.sql

Comprehensive examples of OLS window function usage including:
- Expanding window predictions
- Fixed-size rolling window predictions
- Handling NULL values
- Extracting model diagnostics (R-squared, MSE, standard errors)

**Run:**

```bash
./build/release/duckdb < examples/example-fit-predict-ols.sql
```

## OLS Examples by Use Case

Complete runnable examples covering all OLS API functionality. Each file demonstrates a specific use case with numbered examples and clear comments.

| File | Use Case | Key Topics |
|------|----------|------------|
| `ols_single_series.sql` | Single dataset regression | Basic fit, inference, prediction, diagnostics |
| `ols_multiple_series.sql` | Per-group regression (GROUP BY) | `ols_fit_agg`, coefficient comparison, hierarchical groups |
| `ols_window_functions.sql` | Time series / rolling regression | Expanding, rolling, partitioned windows, `ols_fit_predict` |
| `ols_predict_agg.sql` | Train once, predict all | `ols_predict_agg`, `null_policy`, training vs prediction rows |
| `ols_inference.sql` | Statistical inference | t-tests, p-values, confidence intervals, F-statistic |
| `ols_diagnostics.sql` | Model diagnostics | VIF, Jarque-Bera normality, AIC/BIC, residual analysis |

### Run Examples

```bash
# Single series examples
./build/release/duckdb < examples/ols_single_series.sql

# Per-group regression
./build/release/duckdb < examples/ols_multiple_series.sql

# Window functions (expanding/rolling)
./build/release/duckdb < examples/ols_window_functions.sql

# Predict aggregate (train once, predict all)
./build/release/duckdb < examples/ols_predict_agg.sql

# Statistical inference (t-tests, p-values)
./build/release/duckdb < examples/ols_inference.sql

# Model diagnostics (VIF, normality tests)
./build/release/duckdb < examples/ols_diagnostics.sql
```

### Example Highlights

**Single Series** (`ols_single_series.sql`):
```sql
-- Basic OLS fit with array inputs
SELECT * FROM ols_fit(
    [10.0, 15.0, 20.0, 25.0, 30.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    {'intercept': true}
);
```

**Per-Group Regression** (`ols_multiple_series.sql`):
```sql
-- Fit separate model per category
SELECT category, (result).coefficients[1] AS price_effect, (result).r2
FROM (
    SELECT category, ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub;
```

**Window Functions** (`ols_window_functions.sql`):
```sql
-- Expanding window: train on all preceding rows
SELECT ols_fit_predict(y, [x], {'intercept': true})
    OVER (ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
FROM data;

-- Rolling window: train on last 10 rows
SELECT ols_fit_predict(y, [x], {'intercept': true})
    OVER (ORDER BY time ROWS BETWEEN 9 PRECEDING AND 1 PRECEDING)
FROM data;
```

**Predict Aggregate** (`ols_predict_agg.sql`):
```sql
-- Fit once per group, predict all rows (including future)
SELECT store_id, UNNEST(ols_predict_agg(sales, [marketing_spend])) AS pred
FROM data
GROUP BY store_id;
-- Returns: y, x, yhat, yhat_lower, yhat_upper, is_training
```

**Statistical Inference** (`ols_inference.sql`):
```sql
-- Full inference with significance stars
SELECT
    coefficients[1] AS estimate,
    coefficient_p_values[1] AS p_value,
    CASE WHEN coefficient_p_values[1] < 0.001 THEN '***'
         WHEN coefficient_p_values[1] < 0.01 THEN '**'
         WHEN coefficient_p_values[1] < 0.05 THEN '*' ELSE '' END AS sig
FROM ols_fit(y_arr, x_arr, {'intercept': true, 'full_output': true});
```

**Diagnostics** (`ols_diagnostics.sql`):
```sql
-- VIF for multicollinearity
SELECT vif_agg([x1, x2, x3]) AS vif_values FROM data;

-- Jarque-Bera normality test
SELECT (jarque_bera(residuals)).p_value AS normality_p FROM fitted;

-- AIC/BIC for model comparison
SELECT aic(rss, n, k), bic(rss, n, k) FROM model_stats;
```

## Performance Benchmarks

### DuckDB vs R Comparison (10K Groups)

**Location:** `examples/performance_10k_groups_R/`

Compares `anofox_stats_ols_fit_predict` window function against R's `lm()` function with equivalent semantics.

**Dataset:**
- 10,000 groups
- 100 observations per group
- 8 features (x1-x8)
- 1,000,000 total rows

**Run DuckDB tests:**

```bash
# Generate test data
./build/release/duckdb < examples/performance_10k_groups_R/generate_test_data.sql

# Run window function tests
./build/release/duckdb < examples/performance_10k_groups_R/performance_test_ols_fit_predict.sql

# Run aggregate tests
./build/release/duckdb < examples/performance_10k_groups_R/performance_test_ols_aggregate.sql
```

**Run R tests:**

```bash
# Prerequisites: install.packages(c("arrow", "dplyr", "broom"))

Rscript examples/performance_10k_groups_R/performance_test_ols_fit_predict.R
Rscript examples/performance_10k_groups_R/performance_test_ols_aggregate.R
```

**Run all tests:**

```bash
cd examples/performance_10k_groups_R
chmod +x run_all_tests.sh
./run_all_tests.sh
```

**Compare results:**

```bash
./build/release/duckdb < examples/performance_10k_groups_R/compare_sql_vs_r.sql
```

**Benchmark Results (Intel i7-6800K, 64GB RAM):**

| Test | Rows | Groups | DuckDB | R (lm) | Speedup |
|------|------|--------|--------|--------|---------|
| Expanding Window | 10,000 | 100 | 0.075s | 19.7s | 263x |
| Expanding Window | 1,000,000 | 10,000 | 2.5s | ~1,970s | ~788x |

See `examples/performance_10k_groups_R/README.md` for detailed methodology and result comparison.

### Scale Benchmarks (1M Groups)

**Location:** `examples/performance_1m_groups/`

Stress tests for all regression methods with 1 million groups (100 million rows).

**Methods tested:**
- OLS (`anofox_stats_ols_fit_predict`)
- Ridge (`anofox_stats_ridge_fit_predict`)
- WLS (`anofox_stats_wls_fit_predict`)
- RLS (`anofox_stats_rls_fit_predict`)
- Elastic Net (`anofox_stats_elasticnet_fit_predict`)

**Run individual benchmark:**

```bash
./build/release/duckdb < examples/performance_1m_groups/benchmark_ols.sql
```

**Run all benchmarks:**

```bash
cd examples/performance_1m_groups
chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```

**Benchmark Results (Intel i7-6800K, 64GB RAM):**

| Method | Groups | Rows | Time | Memory |
|--------|--------|------|------|--------|
| OLS | 1,000,000 | 100,000,000 | 178.6s | 8,275 MB |
| Ridge | 1,000,000 | 100,000,000 | 174.6s | 7,922 MB |
| WLS | 1,000,000 | 100,000,000 | 176.5s | 8,757 MB |
| RLS | 1,000,000 | 100,000,000 | 158.1s | 8,571 MB |
| Elastic Net | 1,000,000 | 100,000,000 | 166.8s | 8,146 MB |

See `examples/performance_1m_groups/README.md` for detailed methodology.

## Function Reference

### Aggregate Functions

```sql
-- Fit OLS model via GROUP BY
SELECT anofox_stats_ols_fit_agg(y, [x1, x2, ...]) AS model
FROM data
GROUP BY group_id;

-- Returns STRUCT with: intercept, coefficients[], r2, mse, n_obs
```

### Prediction from Stored Coefficients

```sql
-- Predict using the linear formula: y = intercept + b1*x1 + b2*x2 + ...
SELECT intercept + coefficients[1] * x1 + coefficients[2] * x2 AS yhat;
```

### Window Functions

```sql
-- Expanding window (train on all preceding rows)
SELECT anofox_stats_ols_fit_predict(
    y,
    [x1, x2, ...],
    {'fit_intercept': true}
) OVER (
    PARTITION BY group_id
    ORDER BY time
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
) AS pred
FROM data;

-- Fixed window (train on last N rows)
SELECT anofox_stats_ols_fit_predict(
    y,
    [x1, x2, ...],
    {'fit_intercept': true}
) OVER (
    PARTITION BY group_id
    ORDER BY time
    ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
) AS pred
FROM data;

-- Returns STRUCT with: yhat, std_error, r2, mse, n_obs
```

### Available Window Functions

| Function | Description |
|----------|-------------|
| `anofox_stats_ols_fit_predict` | Ordinary Least Squares |
| `anofox_stats_ridge_fit_predict` | Ridge regression (L2 penalty) |
| `anofox_stats_wls_fit_predict` | Weighted Least Squares |
| `anofox_stats_rls_fit_predict` | Recursive Least Squares (exponential weighting) |
| `anofox_stats_elasticnet_fit_predict` | Elastic Net (L1 + L2 penalty) |

## Troubleshooting

### Extension Not Found

Ensure the extension is built and in the search path:

```bash
# Build
make release

# Or load explicitly in DuckDB
LOAD 'build/release/extension/anofox_stats/anofox_stats.duckdb_extension';
```

### Python Import Errors

Install dependencies with uv:

```bash
cd examples
uv pip install duckdb pandas
```

### Memory Errors on Large Benchmarks

The 1M group benchmarks require ~10GB RAM. Reduce dataset size by editing the SQL files:

```sql
-- Change from 100M to 10M rows
FROM generate_series(1, 10000000) t(i)
```
