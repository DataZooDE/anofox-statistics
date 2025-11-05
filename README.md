# Anofox Statistics - DuckDB Extension

A statistical analysis extension for DuckDB, providing regression analysis, diagnostics, and inference capabilities directly within your database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.1-brightgreen.svg)](https://duckdb.org)

> [!IMPORTANT]
> Gaggle is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

## Features

### 🎯 Core Regression Functions
- **OLS Regression**: Ordinary Least Squares with multiple predictors
- **Ridge Regression**: L2 regularization for multicollinearity
- **Weighted Least Squares**: Handle heteroscedasticity
- **Recursive Least Squares**: Online/streaming estimation
- **Rolling/Expanding Windows**: Time-series regression

### 📊 Statistical Inference
- **Coefficient Tests**: t-statistics, p-values, confidence intervals
- **Prediction Intervals**: Confidence and prediction intervals for forecasts
- **Model Selection**: AIC, BIC, adjusted R² for model comparison

### 🔍 Diagnostics & Validation
- **Residual Diagnostics**: Leverage, Cook's Distance, DFFITS
- **Multicollinearity**: VIF (Variance Inflation Factor) detection
- **Normality Tests**: Jarque-Bera test for residual normality
- **Outlier Detection**: Studentized residuals and influence measures

### 🚀 Advanced Features
- **Aggregate Functions**: Regression per group with `GROUP BY`
- **Window Functions**: Rolling/expanding regressions with `OVER`
- **Array Operations**: Multi-variable regression with array inputs
- **Full Statistics**: Comprehensive fit statistics in structured output

## Quick Start

**Important**: All functions in this extension use **positional parameters**, not named parameters (`:=` syntax). Parameters must be provided in the order shown in the examples below.

```sql
-- Load the extension
LOAD 'anofox_statistics';

-- Simple OLS regression
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1: first predictor
    true                                   -- add_intercept
);

-- Multiple regression with 3 predictors
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y: response variable
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1: first predictor
    [2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],  -- x2: second predictor
    [3.0, 4.0, 5.0, 6.0, 7.0]::DOUBLE[],  -- x3: third predictor
    true                                   -- add_intercept
);

-- Coefficient inference with p-values (also uses positional parameters)
SELECT * FROM ols_inference(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],                  -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],      -- x (matrix format)
    0.95,                                                  -- confidence_level
    true                                                   -- add_intercept
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
        anofox_statistics_ols_agg(
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

Comprehensive guides are available in the [`guides/`](guides/) directory:

- **[Quick Start Guide](guides/01_quick_start.md)**: Get started in 5 minutes
- **[Technical Guide](guides/02_technical_guide.md)**: Architecture and implementation details
- **[Statistics Guide](guides/03_statistics_guide.md)**: Statistical methodology and interpretation
- **[Business Guide](guides/04_business_guide.md)**: Real-world business use cases
- **[Advanced Use Cases](guides/05_advanced_use_cases.md)**: Complex analytical workflows

## Function Reference

### Phase 1: Basic Metrics
- `ols_r2(y, x)` - R-squared
- `ols_rmse(y, x)` - Root Mean Squared Error
- `ols_mse(y, x)` - Mean Squared Error

### Phase 2: Regression Fitting
- `anofox_statistics_ols_fit(...)` - Multi-variable OLS
- `anofox_statistics_ridge_fit(...)` - Ridge regression
- `anofox_statistics_wls_fit(...)` - Weighted Least Squares

### Phase 3: Sequential/Time-Series
- `anofox_statistics_rls_fit(...)` - Recursive Least Squares
- `anofox_statistics_rolling_ols(...)` - Rolling window OLS
- `anofox_statistics_expanding_ols(...)` - Expanding window OLS

### Phase 4: Aggregates
- `anofox_statistics_ols_agg(y DOUBLE, x DOUBLE[], options MAP)` - OLS regression per group
- `anofox_statistics_wls_agg(y DOUBLE, x DOUBLE[], weights DOUBLE, options MAP)` - Weighted LS per group
- `anofox_statistics_ridge_agg(y DOUBLE, x DOUBLE[], options MAP)` - Ridge regression per group
- `anofox_statistics_rls_agg(y DOUBLE, x DOUBLE[], options MAP)` - Recursive LS per group

**Options MAP keys**:
- `intercept` (BOOLEAN): Include intercept term (default: true)
- `lambda` (DOUBLE): Ridge regularization parameter (Ridge only)
- `forgetting_factor` (DOUBLE): Exponential weighting for RLS (RLS only, default: 1.0)

### Phase 5: Inference & Diagnostics
- `ols_inference(y, x, ...)` - Coefficient inference with tests
- `ols_predict_interval(...)` - Predictions with intervals
- `information_criteria(y, x, ...)` - AIC, BIC, model selection
- `residual_diagnostics(y, x, ...)` - Outliers and influence
- `vif(x)` - Variance Inflation Factor
- `normality_test(residuals, ...)` - Jarque-Bera test

## Examples

### Marketing Mix Modeling
```sql
-- Analyze effect of marketing channels on sales
SELECT
    week,
    result.coefficients[1] as tv_roi,
    result.coefficients[2] as digital_roi,
    result.coefficients[3] as print_roi,
    result.r2
FROM (
    SELECT
        week,
        anofox_statistics_ols_agg(
            revenue,
            [tv_spend, digital_spend, print_spend],
            MAP{'intercept': true}
        ) as result
    FROM campaigns
    GROUP BY week
) sub;
```

### Financial Risk Analysis
```sql
-- Calculate beta for each stock (market sensitivity)
SELECT
    stock_id,
    result.coefficients[1] as beta,
    result.intercept as alpha,
    result.r2 as correlation
FROM (
    SELECT
        stock_id,
        anofox_statistics_ols_agg(
            stock_return,
            [market_return],
            MAP{'intercept': true}
        ) as result
    FROM daily_returns
    GROUP BY stock_id
) sub;
```

### Time-Series Forecasting
```sql
-- Rolling regression for adaptive forecasting
SELECT
    date,
    value,
    model.coefficients[1] as trend_coefficient,
    model.r2 as trend_strength
FROM (
    SELECT
        date,
        value,
        anofox_statistics_ols_agg(
            value,
            [time_index],
            MAP{'intercept': true}
        ) OVER (
            ORDER BY date
            ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
        ) as model
    FROM time_series_data
) sub;
```

### Model Diagnostics
```sql
-- Complete statistical workflow
WITH fit AS (
    SELECT * FROM ols_inference(y_data, x_data, 0.95, true)
),
diagnostics AS (
    SELECT * FROM residual_diagnostics(y_data, x_data, true, 2.5, 0.5)
),
quality AS (
    SELECT * FROM information_criteria(y_data, x_data, true)
)
SELECT
    fit.variable,
    fit.estimate,
    fit.p_value,
    fit.significant,
    quality.aic,
    quality.r_squared,
    COUNT(*) FILTER (WHERE diagnostics.is_influential) as influential_points
FROM fit, quality, diagnostics
GROUP BY fit.variable, fit.estimate, fit.p_value, fit.significant, quality.aic, quality.r_squared;
```

## Performance

The extension is implemented in C++ with Eigen for linear algebra, providing:

- **Fast computation**: Native C++ performance
- **Efficient memory usage**: Streaming algorithms where applicable
- **Parallel execution**: Automatic with DuckDB's parallel query engine
- **Large datasets**: Handles millions of rows efficiently

Benchmark (approximate):
- OLS fit with 1M observations, 10 features: ~100ms
- Aggregate regression per 1000 groups: ~500ms
- Window functions with 30-period rolling: ~200ms

## Dependencies

- **DuckDB**: v1.4.1 or higher
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

## Roadmap

### Current (v0.1.0) ✅
- ✅ **Complete regression suite**: OLS, Ridge, WLS, RLS
- ✅ **Time-series support**: Rolling and expanding window regressions
- ✅ **Rank-deficiency handling**: R-like behavior for constant features and multicollinearity
- ✅ **Statistical inference**: Full hypothesis testing, confidence intervals, prediction intervals
- ✅ **Comprehensive diagnostics**: VIF, residual diagnostics, influence measures, normality tests
- ✅ **Aggregate & window functions**: Per-group and rolling analysis
- ✅ **Multi-platform support**: Linux, macOS, Windows (x64 & ARM64)

### Planned (v0.2.0)
- ⏳ LighGBM
- ⏳ Heteroscedasticity tests (White, Breusch-Pagan)
- ⏳ Autocorrelation tests (Durbin-Watson, Ljung-Box)
- ⏳ Robust regression (M-estimators, Huber)
- ⏳ Model selection helpers (cross-validation, stepwise)

### Future 
- 🔮 Generalized Linear Models (GLM)
- 🔮 Quantile regression
- 🔮 Survival analysis (Cox proportional hazards)
- 🔮 Mixed effects models

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

## Related Projects

- [DuckDB](https://duckdb.org) - The analytical database
- [duckdb-wasm](https://github.com/duckdb/duckdb-wasm) - DuckDB in the browser
- [duckplyr](https://github.com/tidyverse/duckplyr) - dplyr backend using DuckDB

---

**Made with ❤️ for the data community**
