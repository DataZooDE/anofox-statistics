# Anofox Statistics - DuckDB Extension

A statistical analysis extension for DuckDB, providing regression analysis, diagnostics, and inference capabilities directly within your database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.1-brightgreen.svg)](https://duckdb.org)

> [!IMPORTANT]
> This extension is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

## What's New Since v0.1.0

**Breaking Changes:**

- **Function naming**: Removed `_fit` suffix from table functions (`anofox_statistics_ols_fit` → `anofox_statistics_ols`)
- **Parameter format**: Changed from individual predictor arrays to matrix format (`DOUBLE[][]`) for features
- **Options API**: Switched from positional parameters to MAP-based options for better flexibility

**New Features:**

- **Elastic Net regression**: Combined L1+L2 regularization for feature selection
- **Model-based prediction**: Efficient prediction using pre-fitted models with confidence/prediction intervals
- **Full model output**: Store complete model metadata with `full_output` option for all regression functions
- **Lateral join support**: All regression functions now support lateral joins with column references
- **Window functions**: All aggregate functions now support OVER clause for rolling/expanding analysis
- **Fit-predict operations**: Unified window functions combining model fitting and prediction in a single pass
- **Diagnostic aggregates**: Group-wise residual analysis, VIF detection, and normality testing
- **Structured output**: All aggregates return rich STRUCT types with comprehensive regression statistics

**Improvements & Bug Fixes:**

- **Validation logic**: Fixed minimum sample size requirements to ensure sufficient degrees of freedom (df_residual ≥ 1)
- **Intercept-only models**: Correctly handle models with no features (p=0) that can fit with n≥1
- **Degrees of freedom**: Corrected df_residual calculation to avoid double-counting intercept term
- **Performance**: Removed debug logging that was degrading test execution performance

**Removed Functions:**

- **Rolling/expanding table functions**: `rolling_ols` and `expanding_ols` removed in favor of window functions
- **Legacy aggregates**: `ols_coeff_agg`, `ols_fit_agg` replaced by unified `anofox_statistics_*_agg` functions

See the [Migration Guide](guides/01_quick_start.md#migration-from-v010) for upgrade instructions, or the [Complete API Changes](guides/api_changes_v0.1.0.md) document for comprehensive details.

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
- **Diagnostic Aggregates**: Group-wise diagnostic analysis (Residuals, VIF, Normality)
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

Comprehensive guides are available in the [`guides/`](guides/) directory:

- **[Quick Start Guide](guides/01_quick_start.md)**: Get started in 5 minutes
- **[Technical Guide](guides/02_technical_guide.md)**: Architecture and implementation details
- **[Statistics Guide](guides/03_statistics_guide.md)**: Statistical methodology and interpretation
- **[Business Guide](guides/04_business_guide.md)**: Real-world business use cases
- **[Advanced Use Cases](guides/05_advanced_use_cases.md)**: Complex analytical workflows

## Function Reference

### Regression Fitting Functions
- `anofox_statistics_ols_fit(y DOUBLE[], x DOUBLE[][], options MAP)` - Multi-variable OLS
- `anofox_statistics_ridge_fit(y DOUBLE[], x DOUBLE[][], options MAP)` - Ridge regression
- `anofox_statistics_wls_fit(y DOUBLE[], x DOUBLE[][], weights DOUBLE[], options MAP)` - Weighted Least Squares
- `anofox_statistics_elastic_net_fit(y DOUBLE[], x DOUBLE[][], options MAP)` - Elastic Net (L1+L2 regularization)

### Phase 3: Sequential/Time-Series
- `anofox_statistics_rls_fit(y DOUBLE[], x DOUBLE[][], options MAP)` - Recursive Least Squares

Note: Rolling and expanding window regressions are available through aggregate window functions (see Phase 4 below).

### Phase 4: Aggregates & Window Functions
All aggregate functions support both `GROUP BY` and `OVER` (window functions):

**Regression Aggregates:**
- `anofox_statistics_ols_fit_agg(y DOUBLE, x DOUBLE[], options MAP)` - OLS regression per group/window
- `anofox_statistics_wls_fit_agg(y DOUBLE, x DOUBLE[], weights DOUBLE, options MAP)` - Weighted LS per group/window
- `anofox_statistics_ridge_fit_agg(y DOUBLE, x DOUBLE[], options MAP)` - Ridge regression per group/window
- `anofox_statistics_rls_fit_agg(y DOUBLE, x DOUBLE[], options MAP)` - Recursive LS per group/window
- `anofox_statistics_elastic_net_fit_agg(y DOUBLE, x DOUBLE[], options MAP)` - Elastic Net per group/window

**Diagnostic Aggregates (GROUP BY only, no window functions):**
- `anofox_statistics_residual_diagnostics_agg(y_actual DOUBLE, y_predicted DOUBLE, options MAP)` - Residual analysis per group
- `anofox_statistics_vif_agg(x DOUBLE[])` - VIF per group
- `anofox_statistics_normality_test_agg(residual DOUBLE, options MAP)` - Jarque-Bera test per group

**Usage:**
- **GROUP BY**: `SELECT category, anofox_statistics_ols_fit_agg(...) FROM data GROUP BY category`
- **Window Functions**: `SELECT anofox_statistics_ols_fit_agg(...) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) FROM data`

**Options MAP keys**:
- `intercept` (BOOLEAN): Include intercept term (default: true)
- `lambda` (DOUBLE): Ridge regularization parameter (Ridge only)
- `forgetting_factor` (DOUBLE): Exponential weighting for RLS (RLS only, default: 1.0)

### Inference & Diagnostics
- Statistical inference is integrated into fit functions with `full_output=true` option
- `anofox_statistics_predict_ols(y_train, x_train, x_new, options MAP)` - Predictions with intervals
- `anofox_statistics_model_predict(...)` - Efficient prediction using pre-fitted models (with confidence/prediction intervals)
- `anofox_statistics_information_criteria(y, x, options MAP)` - AIC, BIC, model selection
- `anofox_statistics_residual_diagnostics(y_actual, y_predicted, outlier_threshold)` - Outlier detection
- `anofox_statistics_vif(x DOUBLE[][])` - Variance Inflation Factor
- `anofox_statistics_normality_test(residuals DOUBLE[], alpha DOUBLE)` - Jarque-Bera test

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
        anofox_statistics_ols_fit_agg(
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
        anofox_statistics_ols_fit_agg(
            stock_return,
            [market_return],
            MAP{'intercept': true}
        ) as result
    FROM daily_returns
    GROUP BY stock_id
) sub;
```

### Elastic Net for Feature Selection

```sql
-- Elastic Net with both L1 and L2 regularization
SELECT *
FROM anofox_statistics_elastic_net_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]::DOUBLE[],  -- y
    [[1.0, 2.0, 1.5], [2.0, 4.0, 3.0], [3.0, 6.0, 4.5],
     [4.0, 8.0, 6.0], [5.0, 10.0, 7.5], [6.0, 12.0, 9.0],
     [7.0, 14.0, 10.5], [8.0, 16.0, 12.0]]::DOUBLE[][],  -- x
    MAP{
        'alpha': 0.5,        -- Mix of L1 and L2 (0=Ridge, 1=Lasso)
        'lambda': 0.1,       -- Regularization strength
        'intercept': true,   -- Include intercept
        'max_iterations': 1000,
        'tolerance': 1e-6
    }
);
-- Returns: coefficients[], intercept, n_nonzero, n_iterations, converged, r_squared, etc.
```

### Time-Series Forecasting with Window Functions

All five regression aggregates (OLS, WLS, Ridge, RLS, Elastic Net) support SQL window functions with the `OVER` clause, enabling rolling and expanding window regression:

```sql
-- Rolling OLS (30-period window)
SELECT
    date,
    value,
    model.coefficients[1] as trend_coefficient,
    model.intercept,
    model.r2 as trend_strength
FROM (
    SELECT
        date,
        value,
        anofox_statistics_ols_fit_agg(
            value,
            [time_index],
            MAP{'intercept': true}
        ) OVER (
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as model
    FROM time_series_data
) sub;

-- Expanding window (cumulative regression)
SELECT
    date,
    anofox_statistics_ols_fit_agg(
        sales,
        [price, advertising],
        MAP{'intercept': true}
    ) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_model
FROM sales_history;

-- Partitioned rolling window (per-category)
SELECT
    category,
    date,
    anofox_statistics_wls_fit_agg(
        outcome,
        [predictor1, predictor2],
        weight,
        MAP{'intercept': true}
    ) OVER (
        PARTITION BY category
        ORDER BY date
        ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
    ) as category_model
FROM panel_data;

-- Ridge regression with rolling window (addresses multicollinearity)
SELECT
    date,
    anofox_statistics_ridge_fit_agg(
        returns,
        [market_factor, size_factor, value_factor],
        MAP{'intercept': true, 'lambda': 1.0}
    ) OVER (
        ORDER BY date
        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW  -- 1 year rolling
    ) as factor_model
FROM daily_returns;

-- RLS for adaptive online learning
SELECT
    timestamp,
    anofox_statistics_rls_fit_agg(
        sensor_reading,
        [temperature, humidity, pressure],
        MAP{'intercept': true, 'forgetting_factor': 0.99}
    ) OVER (
        ORDER BY timestamp
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as adaptive_model
FROM sensor_data;
```

**Window Function Features:**
- ✅ **All regression methods**: OLS, WLS, Ridge, RLS work with `OVER`
- ✅ **Rolling windows**: `ROWS BETWEEN N PRECEDING AND CURRENT ROW`
- ✅ **Expanding windows**: `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
- ✅ **Partitioned analysis**: `PARTITION BY` for group-specific models
- ✅ **Full statistics**: Each window returns complete regression results (coefficients, R², intercept, etc.)
- ✅ **Efficient computation**: Frame-based processing optimized for performance

### Model Diagnostics
```sql
-- Complete statistical workflow
WITH fit AS (
    SELECT * FROM anofox_statistics_ols_fit(
        y_data, x_data,
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
    )
),
diagnostics AS (
    SELECT * FROM anofox_statistics_residual_diagnostics(
        y_actual, y_predicted, 2.5
    )
),
quality AS (
    SELECT aic, bic FROM fit  -- AIC/BIC included in fit output with full_output=true
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

### Efficient Model-Based Prediction

Store a fitted model once, then make predictions on new data without refitting:

```sql
-- 1. Fit model with full_output to store all metadata
CREATE TABLE sales_model AS
SELECT * FROM anofox_statistics_ols_fit(
    sales_array,
    [[price], [advertising], [seasonality]]::DOUBLE[][],
    MAP{'intercept': true, 'full_output': true}
);

-- 2. Make predictions on new data with confidence intervals
SELECT p.*
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept,
    m.coefficients,
    m.mse,
    m.x_train_means,
    m.coefficient_std_errors,
    m.intercept_std_error,
    m.df_residual,
    [[29.99, 5000.0, 0.8], [34.99, 6000.0, 0.9]]::DOUBLE[][],  -- new observations
    0.95,           -- confidence level
    'confidence'    -- interval type: 'confidence', 'prediction', or 'none'
) p;

-- Returns:
-- observation_id | predicted | ci_lower | ci_upper | se
-- 1              | 125000.0  | 120000.0 | 130000.0 | 2500.0
-- 2              | 135000.0  | 129500.0 | 140500.0 | 2750.0

-- 3. Prediction intervals (wider than confidence intervals)
SELECT
    observation_id,
    round(predicted, 2) as forecast,
    round(ci_lower, 2) as lower_bound,
    round(ci_upper, 2) as upper_bound
FROM sales_model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[29.99, 5000.0, 0.8]]::DOUBLE[][],
    0.95,
    'prediction'  -- Prediction intervals account for individual observation uncertainty
) p;

-- 4. Batch predictions without intervals (fastest)
SELECT
    customer_id,
    p.predicted as expected_sales
FROM sales_model m,
     customers c,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[c.price, c.ad_budget, c.seasonality]]::DOUBLE[][],
    0.95,
    'none'  -- Skip interval computation for speed
) p;
```

**Benefits of Model-Based Prediction:**
- ✅ **Performance**: No model refitting - just matrix multiplication
- ✅ **Flexibility**: Predict on any new observations
- ✅ **Intervals**: Confidence intervals (mean) or prediction intervals (individual)
- ✅ **Batch-friendly**: Efficient for scoring large datasets
- ✅ **Works with all regression types**: OLS, Ridge, WLS, Elastic Net, RLS

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
