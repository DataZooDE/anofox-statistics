# Performance Benchmark: 1M Groups

Benchmarks for `fit_predict` window functions and `predict_agg` aggregate functions with 1 million groups.

## Test Configuration

- **Groups**: 1,000,000
- **Rows per group**: 100
- **Total rows**: 100,000,000
- **Features**: 3 (x1, x2, x3)
- **Window frame**: `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`

## Running the Benchmarks

### Prerequisites

Build DuckDB with the anofox_stats extension:

```bash
make
```

### Run Individual Benchmark

```bash
./build/release/duckdb < examples/performance_1m_groups/benchmark_ols.sql
```

### Run All Benchmarks

```bash
./examples/performance_1m_groups/run_all_benchmarks.sh ./build/release/duckdb
```

## Benchmark Results

### Test Machine

| Parameter | Value |
|-----------|-------|
| CPU | Intel Core i7-6800K @ 3.40GHz |
| Cores | 6 (12 threads) |
| RAM | 64 GB |
| OS | Manjaro Linux |
| Kernel | 5.15.196-2-MANJARO |

### Results (2024-12-11)

#### fit_predict Window Functions

| Method | Execution Time | Peak RSS | Parameters |
|--------|----------------|----------|------------|
| OLS | 178.6s | 8,275 MB | `fit_intercept: true` |
| Ridge | 174.6s | 7,922 MB | `alpha: 1.0` |
| WLS | 176.5s | 8,757 MB | `fit_intercept: true` |
| RLS | 158.1s | 8,571 MB | `forgetting_factor: 0.99` |
| Elastic Net | 166.8s | 8,146 MB | `alpha: 1.0, l1_ratio: 0.5` |

#### predict_agg Aggregate Functions (2024-12-12)

| Method | Execution Time | Rows | Parameters |
|--------|----------------|------|------------|
| OLS predict_agg | 208.3s | 100M (80M training, 20M prediction) | `fit_intercept: true` |

### Analysis

**Throughput**: ~560,000-630,000 rows/second across all methods.

**Memory**: Peak RSS (~8 GB) is dominated by DuckDB's window function infrastructure (partitioning, sorting, output buffering). The extension state per partition is O(p²) where p is the number of features:
- Per-partition state: ~512 bytes for 4 coefficients (3 features + intercept)
- 1M partitions theoretical: ~512 MB
- DuckDB overhead: ~7.5 GB

**Execution time** is similar across methods because:
1. All use O(p²) incremental sufficient statistics
2. Data generation (100M rows) dominates
3. Window function partitioning overhead is constant

## Modifying Benchmarks

Each SQL file can be adjusted:

- Change `1000000` to modify group count
- Change `100000000` to modify total rows
- Add features to the array `[x1, x2, x3, ...]`
- Modify model parameters in the options struct
