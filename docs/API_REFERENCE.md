# Anofox Statistics Extension - API Reference

**Version:** 0.6.0
**DuckDB Version:** 1.4.4+
**Backend:** Rust (anofox-regression 0.5.1, anofox-statistics 0.4.0, faer)

## Overview

The Anofox Statistics Extension provides comprehensive regression analysis capabilities for DuckDB. Built with Rust for performance and reliability, it supports multiple regression methods including linear models, generalized linear models (GLM), augmented linear models (ALM), and constrained optimization (BLS/NNLS).

## Quick Reference

### Regression Methods

| Method | Scalar | Aggregate | Documentation |
|--------|--------|-----------|---------------|
| OLS | `ols_fit` | `ols_fit_agg` | [OLS](api/regression/ols.md) |
| Ridge | `ridge_fit` | `ridge_fit_agg` | [Ridge](api/regression/ridge.md) |
| Elastic Net | `elasticnet_fit` | `elasticnet_fit_agg` | [Elastic Net](api/regression/elasticnet.md) |
| WLS | `wls_fit` | `wls_fit_agg` | [WLS](api/regression/wls.md) |
| RLS | `rls_fit` | `rls_fit_agg` | [RLS](api/regression/rls.md) |
| BLS | - | `bls_fit_agg` | [BLS/NNLS](api/regression/bls.md) |
| NNLS | - | `nnls_fit_agg` | [BLS/NNLS](api/regression/bls.md) |
| PLS | `pls_fit` | `pls_fit_agg` | [PLS](api/regression/pls.md) |
| Isotonic | `isotonic_fit` | `isotonic_fit_agg` | [Isotonic](api/regression/isotonic.md) |
| Quantile | `quantile_fit` | `quantile_fit_agg` | [Quantile](api/regression/quantile.md) |

### GLM Functions

| Method | Aggregate | Documentation |
|--------|-----------|---------------|
| Poisson | `poisson_fit_agg` | [Poisson](api/glm/poisson.md) |
| ALM | `alm_fit_agg` | [ALM](api/glm/alm.md) |

### Statistical Hypothesis Tests

| Category | Function | Documentation |
|----------|----------|---------------|
| Parametric | `t_test_agg`, `one_way_anova_agg` | [Hypothesis Tests](api/statistics/hypothesis.md) |
| Nonparametric | `mann_whitney_u_agg`, `kruskal_wallis_agg` | [Hypothesis Tests](api/statistics/hypothesis.md) |
| Normality | `shapiro_wilk_agg`, `jarque_bera_agg` | [Hypothesis Tests](api/statistics/hypothesis.md) |
| Equivalence | `tost_t_test_agg`, `tost_paired_agg` | [Hypothesis Tests](api/statistics/hypothesis.md) |

### Correlation Tests

| Function | Description | Documentation |
|----------|-------------|---------------|
| `pearson_agg` | Pearson correlation | [Correlation](api/statistics/correlation.md) |
| `spearman_agg` | Spearman rank correlation | [Correlation](api/statistics/correlation.md) |
| `kendall_agg` | Kendall tau correlation | [Correlation](api/statistics/correlation.md) |
| `distance_cor_agg` | Distance correlation | [Correlation](api/statistics/correlation.md) |
| `icc_agg` | Intraclass correlation | [Correlation](api/statistics/correlation.md) |

### Categorical Tests

| Function | Description | Documentation |
|----------|-------------|---------------|
| `chisq_test_agg` | Chi-square independence | [Categorical](api/statistics/categorical.md) |
| `fisher_exact_agg` | Fisher's exact test | [Categorical](api/statistics/categorical.md) |
| `mcnemar_agg` | McNemar's test | [Categorical](api/statistics/categorical.md) |
| `cramers_v_agg` | Cramér's V | [Categorical](api/statistics/categorical.md) |
| `cohen_kappa_agg` | Cohen's kappa | [Categorical](api/statistics/categorical.md) |

### Diagnostics & Utilities

| Function | Description | Documentation |
|----------|-------------|---------------|
| `vif`, `vif_agg` | Variance Inflation Factor | [Diagnostics](api/diagnostics/diagnostics.md) |
| `aic`, `bic` | Model selection criteria | [Diagnostics](api/diagnostics/diagnostics.md) |
| `residuals_diagnostics_agg` | Residual analysis | [Diagnostics](api/diagnostics/diagnostics.md) |
| `aid_agg`, `aid_anomaly_agg` | Demand classification | [AID](api/aid/aid.md) |

### Table Macros

| Macro | Description | Documentation |
|-------|-------------|---------------|
| `ols_fit_predict_by` | Per-group OLS predictions | [Table Macros](api/macros/table_macros.md) |
| `ridge_fit_predict_by` | Per-group Ridge predictions | [Table Macros](api/macros/table_macros.md) |
| `elasticnet_fit_predict_by` | Per-group Elastic Net | [Table Macros](api/macros/table_macros.md) |
| `wls_fit_predict_by` | Per-group WLS | [Table Macros](api/macros/table_macros.md) |
| `rls_fit_predict_by` | Per-group RLS | [Table Macros](api/macros/table_macros.md) |
| `bls_fit_predict_by` | Per-group BLS | [Table Macros](api/macros/table_macros.md) |
| `alm_fit_predict_by` | Per-group ALM | [Table Macros](api/macros/table_macros.md) |
| `poisson_fit_predict_by` | Per-group Poisson | [Table Macros](api/macros/table_macros.md) |
| `aid_anomaly_by` | Grouped anomaly detection | [Table Macros](api/macros/table_macros.md) |

> **Deprecation Notice:** The old `*_predict_agg` names (`ols_predict_agg`, etc.) are deprecated
> but still work for backwards compatibility. Use `*_fit_predict_agg` instead.

---

## Function Types

### Scalar Functions (Array-based)
Process complete arrays of data in a single call. Best for batch operations.
```sql
SELECT ols_fit(y_array, x_arrays);
```

### Aggregate Functions (Streaming)
Accumulate data row-by-row. Support `GROUP BY` and window functions via `OVER`.
```sql
SELECT ols_fit_agg(y, [x1, x2]) FROM table GROUP BY category;
```

### Table Macros
Convenience wrappers that handle GROUP BY, UNNEST, and column extraction automatically.
```sql
SELECT * FROM ols_fit_predict_by('sales', region, revenue, [ads, price]);
```

---

## Common Options

### null_policy Parameter

The `null_policy` option controls how NULL values are handled during model training.

| Value | Training Set | Predictions |
|-------|--------------|-------------|
| `'drop'` (default) | Rows where y IS NOT NULL | All rows get predictions |
| `'drop_y_zero_x'` | Rows where y IS NOT NULL AND all x != 0 | All rows get predictions |

---

## Return Types

### FitResult Structure

Standard return type for linear regression functions.

```
STRUCT(
    coefficients LIST(DOUBLE),
    intercept DOUBLE,
    r_squared DOUBLE,
    adj_r_squared DOUBLE,
    mse DOUBLE,
    rmse DOUBLE,
    mae DOUBLE,
    rss DOUBLE,
    tss DOUBLE,
    n_observations BIGINT,
    n_features INTEGER,
    -- When compute_inference=true:
    t_statistics LIST(DOUBLE),
    p_values LIST(DOUBLE),
    std_errors LIST(DOUBLE),
    conf_int_lower LIST(DOUBLE),
    conf_int_upper LIST(DOUBLE)
)
```

### Accessing Results

```sql
-- Extract specific fields
SELECT
    (result).r_squared,
    (result).coefficients[1] as beta1,
    (result).coefficients[2] as beta2
FROM (SELECT ols_fit_agg(y, [x1, x2]) as result FROM data);

-- Expand all fields
SELECT (ols_fit_agg(y, [x1, x2])).* FROM data;
```

---

## Short Aliases

Most functions have short aliases without the `anofox_stats_` prefix:

| Full Name | Alias |
|-----------|-------|
| `anofox_stats_ols_fit` | `ols_fit` |
| `anofox_stats_ridge_fit` | `ridge_fit` |
| `anofox_stats_t_test_agg` | `t_test_agg` |
| `anofox_stats_pearson_agg` | `pearson_agg` |
| ... | ... |

---

## Detailed Documentation

For comprehensive documentation on each function category:

- **Regression**: [OLS](api/regression/ols.md) | [Ridge](api/regression/ridge.md) | [Elastic Net](api/regression/elasticnet.md) | [WLS](api/regression/wls.md) | [RLS](api/regression/rls.md) | [BLS/NNLS](api/regression/bls.md) | [PLS](api/regression/pls.md) | [Isotonic](api/regression/isotonic.md) | [Quantile](api/regression/quantile.md)
- **GLM**: [Poisson](api/glm/poisson.md) | [ALM](api/glm/alm.md)
- **Statistics**: [Hypothesis Tests](api/statistics/hypothesis.md) | [Correlation](api/statistics/correlation.md) | [Categorical](api/statistics/categorical.md)
- **AID**: [Demand Classification](api/aid/aid.md)
- **Diagnostics**: [Model Diagnostics](api/diagnostics/diagnostics.md)
- **Table Macros**: [Table Macros](api/macros/table_macros.md)

---

## Error Handling

All functions return NULL on error conditions:
- Invalid input types
- Empty arrays
- Singular matrices (insufficient data variation)

Check for NULL results when using these functions:
```sql
SELECT COALESCE((ols_fit_agg(y, [x])).r_squared, 0.0) as r_squared
FROM data;
```

---

## Performance Notes

- **Aggregate functions** are generally preferred for large datasets as they process data in a streaming fashion
- **Scalar functions** may be faster for small, pre-aggregated arrays
- Use **table macros** for the simplest syntax when doing per-group predictions
- VIF computation is O(k³) where k is the number of features

---

## Version History

- **0.6.0**: Added aid_anomaly_by table macro, reorganized documentation
- **0.5.0**: Added PLS, Isotonic, Quantile regression
- **0.4.0**: Added ALM with 24 distributions
- **0.3.0**: Added comprehensive hypothesis testing
- **0.2.0**: Added BLS/NNLS, RLS
- **0.1.0**: Initial release with OLS, Ridge, Elastic Net, WLS
