# API Design

**Version:** 0.6.0
**DuckDB Version:** 1.4.3+
**Backend:** Rust (anofox-regression 0.5.1, anofox-statistics 0.4.0, faer)

## Overview

The Anofox Statistics Extension provides comprehensive regression analysis and statistical testing capabilities for DuckDB. Built with Rust for performance and reliability.

## Function Types

### Scalar Functions (Array-based)

Process complete arrays of data in a single call. Best for batch operations.

```sql
SELECT anofox_stats_ols_fit(y_array, x_arrays);
```

### Aggregate Functions (Streaming)

Accumulate data row-by-row. Support `GROUP BY` and window functions via `OVER`.

```sql
SELECT anofox_stats_ols_fit_agg(y, [x1, x2]) FROM table GROUP BY category;
```

## Naming Conventions

- `anofox_stats_*` - Full namespaced functions
- Short aliases available (e.g., `ols_fit` for `anofox_stats_ols_fit`)
- `*_fit` - Model fitting functions
- `*_fit_agg` - Streaming aggregate fitting
- `*_fit_predict` - Window functions for fit and predict
- `*_fit_predict_agg` - Aggregate fit and predict
- `*_fit_predict_by` - Table macros for grouped fit-predict

## Common Options

Most functions accept an optional `options MAP` parameter for configuration:

```sql
SELECT ols_fit_agg(y, [x], {'fit_intercept': true, 'compute_inference': true});
```

See [Common Options](19-common-options.md) for full details.
