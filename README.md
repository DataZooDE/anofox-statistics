# Anofox Statistics - DuckDB Extension

A statistical analysis extension for DuckDB, providing regression analysis, diagnostics, and inference capabilities directly within your database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.3-brightgreen.svg)](https://duckdb.org)

> [!WARNING]
> **BREAKING CHANGE**: Function names have changed from `anofox_statistics_*` to `anofox_stats_*` to align with the unified naming convention across Anofox extensions. Aliases without the prefix (e.g., `ols_fit`) are also available for convenience. Update your code accordingly.

> [!IMPORTANT]
> This extension is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

## Features

### Regression Methods

| Method | Function | Description |
|--------|----------|-------------|
| OLS | `ols_fit`, `ols_fit_agg` | Ordinary Least Squares |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2 regularization |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | Combined L1+L2 regularization |
| WLS | `wls_fit`, `wls_fit_agg` | Weighted Least Squares |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive Least Squares (online) |
| Poisson | `poisson_fit_agg` | GLM for count data |
| ALM | `alm_fit_agg` | 24 error distributions |
| BLS/NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded/Non-negative LS |

### Statistical Hypothesis Tests

| Category | Function | Description |
|----------|----------|-------------|
| Normality | `shapiro_wilk_agg`, `jarque_bera_agg`, `dagostino_k2_agg` | Normality tests |
| Parametric | `t_test_agg`, `one_way_anova_agg`, `yuen_agg`, `brown_forsythe_agg` | Parametric tests |
| Nonparametric | `mann_whitney_u_agg`, `kruskal_wallis_agg`, `wilcoxon_signed_rank_agg`, `brunner_munzel_agg`, `permutation_t_test_agg` | Nonparametric tests |
| Correlation | `pearson_agg`, `spearman_agg`, `kendall_agg`, `distance_cor_agg`, `icc_agg` | Correlation tests |
| Categorical | `chisq_test_agg`, `chisq_gof_agg`, `g_test_agg`, `fisher_exact_agg`, `mcnemar_agg` | Contingency table tests |
| Effect Size | `cramers_v_agg`, `phi_coefficient_agg`, `contingency_coef_agg`, `cohen_kappa_agg` | Association measures |
| Proportion | `prop_test_one_agg`, `prop_test_two_agg`, `binom_test_agg` | Proportion tests |
| Equivalence | `tost_t_test_agg`, `tost_paired_agg`, `tost_correlation_agg` | TOST equivalence tests |
| Distribution | `energy_distance_agg`, `mmd_agg` | Distribution comparison |
| Forecast | `diebold_mariano_agg`, `clark_west_agg` | Forecast evaluation |

### Diagnostics & Utilities

| Function | Description |
|----------|-------------|
| `vif`, `vif_agg` | Variance Inflation Factor |
| `aic`, `bic` | Model selection criteria |
| `residuals_diagnostics_agg` | Residual analysis |
| `aid_agg`, `aid_anomaly_agg` | Demand pattern classification |

### Key Capabilities
- **Aggregate Functions**: All methods support `GROUP BY` for per-group analysis
- **Window Functions**: Regression methods support `OVER` clause for rolling/expanding windows
- **Prediction Intervals**: Confidence and prediction intervals for forecasts
- **Full Inference**: t-statistics, p-values, confidence intervals for coefficients

## Quick Start

**Important**: All functions in this extension use **positional parameters**, not named parameters (`:=` syntax). Parameters must be provided in the order shown in the examples below.

### Naming Convention

All functions follow the `anofox_stats_*` naming convention, with convenient aliases without the prefix:

- **Primary**: `anofox_stats_ols_fit(...)`
- **Alias**: `ols_fit(...)` - Shorter and more convenient!

Both work identically - use whichever you prefer!

```sql
-- Load the extension
LOAD 'anofox_statistics';

-- Simple OLS regression (using alias - recommended)
SELECT * FROM ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [[1.1], [2.1], [2.9], [4.2], [4.8]]::DOUBLE[][],  -- x: feature matrix
    MAP{'intercept': true}                 -- options
);

-- Or use the full name if you prefer
SELECT * FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [[1.1], [2.1], [2.9], [4.2], [4.8]]::DOUBLE[][],  -- x: feature matrix
    MAP{'intercept': true}                 -- options
);

-- Multiple regression with 3 predictors
SELECT * FROM ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [[1.1, 2.0, 3.0], [2.1, 3.0, 4.0], [2.9, 4.0, 5.0],
     [4.2, 5.0, 6.0], [4.8, 6.0, 7.0]]::DOUBLE[][],  -- x: feature matrix
    MAP{'intercept': true}                 -- options
);

-- Get coefficient inference using fit with full_output
SELECT
    coefficients,
    intercept,
    coefficient_p_values,
    intercept_p_value,
    f_statistic,
    f_statistic_pvalue
FROM ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],                  -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],      -- x (matrix format)
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}  -- options
);

-- Per-group regression with aggregate functions
SELECT
    category,
    result.coefficients[1] as price_effect,
    result.intercept,
    result.r2
FROM (
    SELECT
        category,
        ols_fit_agg(
            sales,
            [price],
            MAP{'intercept': true}
        ) as result
    FROM sales_data
    GROUP BY category
) sub;
```

## Installation

### Community Extension

```sql
INSTALL anofox_statistics FROM community;
LOAD anofox_statistics;
```

## Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete function reference and specifications

Guides are available in the [`guides/`](guides/) directory:

- **[Quick Start Guide](guides/01_quick_start.md)**: Getting started
- **[Technical Guide](guides/02_technical_guide.md)**: Architecture and implementation details
- **[Business Guide](guides/03_business_guide.md)**: Real-world business use cases
- **[Advanced Use Cases](guides/04_advanced_use_cases.md)**: Complex analytical workflows

## Dependencies

- **DuckDB**: v1.4.3 or higher
- **Rust**: Stable toolchain (for building from source)
- **anofox-regression**: Rust regression library (via Cargo)
- **faer**: Linear algebra library (via Cargo, no default features)
- **ICU Extension** (Optional): Required for date/time operations in documentation examples
  ```sql
  INSTALL icu;
  LOAD icu;
  ```

## Telemetry

This extension collects anonymous usage telemetry to help improve the product. Telemetry is **enabled by default** and includes:

- Extension load events (extension name, version, platform)
- Function execution events (which functions are used)
- No personal data or query contents are collected

### Disabling Telemetry

**Environment Variable:**
```bash
export DATAZOO_DISABLE_TELEMETRY=1
```

**DuckDB SQL Setting:**
```sql
SET anofox_telemetry_enabled = false;
```

For more information, see the [posthog-telemetry](https://github.com/DataZooDE/posthog-telemetry) repository.

## License

This project is licensed under the **Business Source License 1.1** (BSL 1.1).

### Key Terms

- **Usage Grant**: Free to use, modify, and distribute for non-production purposes
- **Production Use**: Permitted after 4 years from release date, or under a commercial license
- **Change Date**: [Release Date + 4 years]
- **Change License**: Apache License 2.0

See [LICENSE](LICENSE) for full terms.

### Why BSL?

The BSL allows:
- Free use for development, testing, and research
- Open source collaboration and contributions
- Academic and educational use
- Small-scale production use

While ensuring:
- Sustainable development funding
- Protection for the project's long-term viability
- Future conversion to fully open source (Apache 2.0)

For commercial production use before the Change Date, please contact: [contact@datazoo.de]

## Contributing

Contributions are welcome. To get started:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For questions, open an issue on GitHub.

### Areas for Contribution

- Additional statistical tests (Durbin-Watson, Breusch-Pagan, etc.)
- Visualization helpers for diagnostics
- Documentation and examples
- Bug reports and fixes
- Performance optimizations
- Internationalization

## Support

- **Documentation**: [guides/](guides/)
- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-statistics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DataZooDE/anofox-statistics/discussions)
- **Email**: contact@datazoo.de

## Citation

If you use this extension in research, please cite:

```bibtex
@software{anofox_statistics,
  title = {Anofox Statistics: Statistical Analysis Extension for DuckDB},
  author = {DataZoo DE},
  year = {2025},
  url = {https://github.com/DataZooDE/anofox-statistics},
  version = {1.0.0}
}
```

## Validation

The regression algorithms in this extension are powered by the [anofox-regression](https://github.com/DataZooDE/anofox-regression) Rust crate, which is validated against R's statistical functions. The test suite compares results with R's `lm()`, `glmnet`, and other standard statistical packages to ensure numerical accuracy.

## Acknowledgments

- **DuckDB Team**: For the database and extension framework
- **anofox-regression**: Rust regression library with R-validated implementations
- **Open Source Community**: For contributions and feedback
