# Anofox Statistics - DuckDB Extension

A statistical analysis extension for DuckDB, providing regression analysis, diagnostics, and inference capabilities directly within your database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.2-brightgreen.svg)](https://duckdb.org)

> [!IMPORTANT]
> This extension is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

## Features

### 🎯 Core Regression Functions
- **OLS Regression**: Ordinary Least Squares with multiple predictors
- **Ridge Regression**: L2 regularization for multicollinearity
- **Elastic Net**: Combined L1+L2 regularization for feature selection and stability
- **Weighted Least Squares**: Handle heteroscedasticity
- **Recursive Least Squares**: Online/streaming estimation
- **Rolling/Expanding Windows**: Time-series regression

### 📊 Statistical Inference
- **Coefficient Tests**: t-statistics, p-values, confidence intervals
- **Prediction Intervals**: Confidence and prediction intervals for forecasts
- **Model-Based Prediction**: Efficient prediction using pre-fitted models (no refitting required)
- **Model Selection**: AIC, BIC, adjusted R² for model comparison

### 🔍 Diagnostics & Validation
- **Residual Diagnostics**: Outlier detection with standardized residuals
- **Residual Diagnostics Aggregate**: Group-wise residual analysis with summary/detailed modes
- **Multicollinearity**: VIF (Variance Inflation Factor) detection
- **VIF Aggregate**: Per-group multicollinearity detection with severity classification
- **Normality Tests**: Jarque-Bera test for residual normality
- **Normality Test Aggregate**: Per-group normality testing with skewness and kurtosis

### 🚀 Advanced Features
- **Aggregate Functions**: Regression per group with `GROUP BY` (OLS, WLS, Ridge, RLS, Elastic Net)
- **Window Functions**: Rolling/expanding regressions with `OVER` clause for all regression methods
  - Rolling windows: `ROWS BETWEEN N PRECEDING AND CURRENT ROW`
  - Expanding windows: `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
  - Partitioned analysis: `PARTITION BY` for group-specific time series
- **Diagnostic Aggregates**: Group-wise diagnostic analysis with `GROUP BY` only (Residuals, VIF, Normality) - does not support window functions
- **Array Operations**: Multi-variable regression with array inputs
- **Full Statistics**: Comprehensive fit statistics in structured output (coefficients, R², adj. R², intercepts, etc.)

## Quick Start

**Important**: All functions in this extension use **positional parameters**, not named parameters (`:=` syntax). Parameters must be provided in the order shown in the examples below.

```sql
-- Load the extension
LOAD 'anofox_statistics';

-- Simple OLS regression
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [[1.1], [2.1], [2.9], [4.2], [4.8]]::DOUBLE[][],  -- x: feature matrix
    MAP{'intercept': true}                 -- options
);

-- Multiple regression with 3 predictors
SELECT * FROM anofox_statistics_ols_fit(
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
FROM anofox_statistics_ols_fit(
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
        anofox_statistics_ols_fit_agg(
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

### From Source

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/DataZooDE/anofox-statistics.git
cd anofox-statistics

# Build the extension
make release

# The extension will be built to:
# build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension
```

## Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete function reference and specifications

Comprehensive guides are available in the [`guides/`](guides/) directory:

- **[Quick Start Guide](guides/01_quick_start.md)**: Get started in 5 minutes
- **[Technical Guide](guides/02_technical_guide.md)**: Architecture and implementation details
- **[Business Guide](guides/03_business_guide.md)**: Real-world business use cases
- **[Advanced Use Cases](guides/04_advanced_use_cases.md)**: Complex analytical workflows

## Dependencies

- **DuckDB**: v1.4.2 or higher
- **Eigen3**: Linear algebra library (included as header-only)
- **C++17 compiler**: GCC 7+, Clang 9+, MSVC 2019+

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
- ✅ Free use for development, testing, and research
- ✅ Open source collaboration and contributions
- ✅ Academic and educational use
- ✅ Small-scale production use

While ensuring:
- 💼 Sustainable development funding
- 🔒 Protection for the project's long-term viability
- 🎯 Future conversion to fully open source (Apache 2.0)

For commercial production use before the Change Date, please contact: [contact@datazoo.de]

## Contributing

We welcome contributions! To get started:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For questions, open an issue on GitHub.

### Development Setup

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/DataZooDE/anofox-statistics.git
cd anofox-statistics

# Build in debug mode
make debug

# Run tests
make test

# Format code
make format
```

### Areas for Contribution

- 📊 Additional statistical tests (Durbin-Watson, Breusch-Pagan, etc.)
- 🎨 Visualization helpers for diagnostics
- 📚 Documentation and examples
- 🐛 Bug reports and fixes
- ⚡ Performance optimizations
- 🌍 Internationalization

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

## Acknowledgments

- **DuckDB Team**: For the excellent database and extension framework
- **Eigen Project**: For high-performance linear algebra
- **Open Source Community**: For contributions and feedback

---

**Made with ❤️ for the data community**
