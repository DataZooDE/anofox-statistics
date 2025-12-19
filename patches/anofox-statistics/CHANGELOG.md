# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-12-17

### Added

- **Equivalence Testing Module**: New `equivalence` module providing TOST (Two One-Sided Tests) procedures for establishing practical equivalence

- **TOST for T-Tests**:
  - `tost_t_test_one_sample()`: One-sample equivalence test
  - `tost_t_test_two_sample()`: Two-sample equivalence test (Welch/Student)
  - `tost_t_test_paired()`: Paired samples equivalence test

- **TOST for Correlations**:
  - `tost_correlation()`: Test if correlation is practically equivalent to zero (or other value)
  - Supports Pearson and Spearman methods via `CorrelationTostMethod` enum
  - Uses Fisher's z-transformation for inference

- **TOST for Proportions**:
  - `tost_prop_one()`: Single proportion equivalence test
  - `tost_prop_two()`: Two proportions equivalence test

- **Non-parametric TOST**:
  - `tost_wilcoxon_paired()`: Paired Wilcoxon signed-rank equivalence test
  - `tost_wilcoxon_two_sample()`: Two-sample Wilcoxon rank-sum equivalence test
  - Uses Hodges-Lehmann estimator for location

- **Robust and Resampling TOST**:
  - `tost_bootstrap()`: Bootstrap-based equivalence test with percentile CI
  - `tost_yuen()`: Robust trimmed means equivalence test

- **New Types**:
  - `TostResult`: Comprehensive result struct with estimates, CIs, bounds, and test statistics
  - `OneSidedTestResult`: Result for each one-sided test within TOST
  - `EquivalenceBounds`: Enum supporting Raw, Symmetric, and Cohen's d bounds
  - `CorrelationTostMethod`: Enum for Pearson/Spearman correlation methods

- **Documentation**:
  - Full API documentation for all 10 TOST functions
  - R validation reference (TOSTER package)
  - Usage examples in README and API reference

### R Equivalents

All TOST functions are designed to match the TOSTER package in R:
- `TOSTER::TOSTone()`, `TOSTtwo()`, `TOSTpaired()` for t-tests
- `TOSTER::TOSTr()` for correlations
- `TOSTER::TOSTtwo.prop()` for proportions
- `TOSTER::wilcox_TOST()` for Wilcoxon tests
- `TOSTER::boot_t_TOST()` for bootstrap

## [0.3.0] - 2025-12-12

### Added

- **Full R Output Parity**: All statistical test result structs now include all fields returned by R's `htest` objects (except `method` and `data.name`)
  - `DMResult`: Added `horizon`, `loss_function`, `varestimator`, `alternative` fields
  - `TTestResult`: Added `null_value` field
  - `MannWhitneyResult`: Added `null_value` field
  - `WilcoxonResult`: Added `null_value` field
  - `YuenResult`: Added `conf_int` field with `YuenConfInt` struct

- **Full R Input Parameter Parity**: All statistical tests now support the same parameters as their R equivalents
  - `mann_whitney_u()`: Added `alternative`, `continuity_correction`, `exact`, `conf_level`, `mu` parameters
  - `wilcoxon_signed_rank()`: Added `alternative`, `continuity_correction`, `exact`, `conf_level`, `mu` parameters
  - `diebold_mariano()`: Added `alternative`, `varestimator` parameters
  - `yuen_test()`: Added `alternative`, `conf_level` parameters
  - `t_test()`: Added `mu`, `conf_level` parameters
  - `brunner_munzel()`: Added `alternative`, `alpha` parameters

- **Exact P-Values**: Mann-Whitney U and Wilcoxon signed-rank tests now support exact p-value computation for small samples without ties

- **Confidence Intervals**: Added Hodges-Lehmann confidence intervals for Mann-Whitney U and Wilcoxon signed-rank tests

- **Numerically Stable Algorithms**: Added `stable_mean()` and `stable_variance()` using Welford's online algorithm

- **CI/CD**: Added automatic cargo publish to crates.io on GitHub release

- **Documentation**:
  - Comprehensive API reference in `doc/API_REFERENCE.md`
  - Scientific references for all statistical tests
  - Runnable examples for all test categories
  - R validation documentation in `R/VALIDATION.md`

### Changed

- Refactored complex modules to improve code maintainability
- Consolidated documentation structure

## [0.2.0] - 2025-12-09

### Added

- **Complete R Validation Coverage**: All 31 statistical tests now have comprehensive R validation
- **New Test Modules**:
  - `tdd_modern.rs` - Tests for Energy Distance and MMD
  - `tdd_resampling.rs` - Tests for Permutation T-Test and Bootstrap methods

### Validated Against R

The following tests now have R reference validation:

#### Distributional Tests
- D'Agostino's K-squared test (validated against `moments::agostino.test()`)

#### Nonparametric Tests
- Brunner-Munzel test (validated against `lawstat::brunner.munzel.test()`)

#### Forecast Evaluation
- Clark-West test for nested model comparison
- Superior Predictive Ability (SPA) test
- Model Confidence Set (MCS) with Range and Semi-Quadratic statistics

#### Modern Distribution Tests
- Energy Distance test (validated against `energy::eqdist.etest()`)
- Maximum Mean Discrepancy (MMD) test

#### Resampling Methods
- Permutation T-test (validated against `coin::independence_test()`)
- Stationary Bootstrap
- Circular Block Bootstrap

### Test Coverage

- Total tests: 194 (all passing)
- R validation coverage: 100% of statistical tests

## [0.1.0] - 2025-01-XX

### Added

- Initial release
- **Parametric Tests**: T-tests (Welch, Student, Paired), Yuen's test, Brown-Forsythe
- **Nonparametric Tests**: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis
- **Distributional Tests**: Shapiro-Wilk normality test
- **Forecast Evaluation**: Diebold-Mariano test
- **Math Primitives**: Mean, variance, std_dev, median, trimmed_mean, skewness, kurtosis
- R validation framework with CSV-based reference data
