# Test: OLS Regression Validation

## Purpose

Validates the OLS (Ordinary Least Squares) regression functionality of the anofox_statistics extension against R's `lm()` function, which is the gold standard for statistical computation.

## Test Data

### Input Data

- **Location:** `test/data/ols_tests/input/`
- **Files:**
  - `simple_linear.csv` - Simple linear regression (n=100, 1 predictor)
  - `multiple_regression.csv` - Multiple regression (n=200, 3 predictors)
  - `no_intercept.csv` - Regression without intercept (n=150)
  - `rank_deficient.csv` - Rank-deficient matrix with constant feature (n=100)
  - `perfect_collinearity.csv` - Perfect collinearity between predictors (n=80)

- **Format:** CSV with headers
- **Description:** Synthetic data generated with controlled properties to test specific statistical scenarios

### Expected Output

- **Location:** `test/data/ols_tests/expected/`
- **Files:** JSON files with corresponding names containing R computation results
- **Format:** JSON with high precision (15 digits)
- **Contents:**
  - `coefficients`: Regression coefficients from R's `lm()`
  - `r_squared`: R² value
  - `adj_r_squared`: Adjusted R²
  - `sigma`: Residual standard error
  - `residuals`: Model residuals
  - `fitted_values`: Fitted values
  - `df_residual`: Residual degrees of freedom

### Computation Details

Expected results were computed using R's `lm()` function:

```r
model <- lm(y ~ x1 + x2 + ..., data = input_data)
summary(model)
```

R automatically handles:
- Rank-deficient matrices (drops linearly dependent columns)
- Perfect collinearity (drops collinear predictors)
- Intercept term (can be included or excluded)

## Validation

The SQL test (`test/sql/validate_ols_simple.sql`):
1. Loads input data from CSV
2. Applies extension aggregate function `ols_fit_agg(y, x)`
3. Compares R² with expected output
4. Asserts match within tolerance: **0.01** (1% relative error)

**Note:** The validation uses aggregate functions (`ols_fit_agg`) rather than table functions because they work directly with table data without requiring subqueries.

## Regenerating Data

To regenerate this test data:

```bash
cd validation
Rscript generators/generate_ols_tests.R
```

Or regenerate all test data:

```bash
make generate-test-data
```

## Test Cases

### 1. Simple Linear Regression
- **Purpose:** Basic functionality test with single predictor
- **Model:** y = β₀ + β₁x + ε
- **Expected R²:** ~0.99 (high fit)

### 2. Multiple Regression
- **Purpose:** Test with multiple independent predictors
- **Model:** y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ε
- **Expected R²:** ~0.97
- **Predictors:** 3 uncorrelated variables

### 3. No Intercept
- **Purpose:** Regression through the origin
- **Model:** y = β₁x + ε (no β₀ term)
- **Expected R²:** ~0.999

### 4. Rank Deficient
- **Purpose:** Test handling of constant features
- **Issue:** x₂ is constant (no variation)
- **Expected:** R drops constant column, fits with remaining predictors

### 5. Perfect Collinearity
- **Purpose:** Test handling of linearly dependent predictors
- **Issue:** x₂ = 2×x₁ + 3 (perfect linear relationship)
- **Expected:** R drops collinear column automatically

## Tolerance Levels

- **Coefficients:** 1e-10 (strict - expect exact match)
- **R²:** 1e-10 (strict)
- **P-values:** 1e-8 (relaxed - can vary slightly due to numerical precision)
- **Aggregate tests:** 0.01 (1% relative error - practical tolerance)

## References

- R documentation: `?lm`, `?summary.lm`
- Statistical reference: Ordinary Least Squares regression
- Extension functions: `ols_fit_agg`, `ols_coeff_agg`, `anofox_statistics_ols_fit`
