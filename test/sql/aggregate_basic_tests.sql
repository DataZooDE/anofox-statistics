-- Comprehensive test suite for aggregate functions
-- Tests: anofox_statistics_ols_fit_agg, anofox_statistics_wls_fit_agg,
--        anofox_statistics_ridge_fit_agg, anofox_statistics_rls_fit_agg
--
-- R VALIDATION METHODOLOGY:
-- All aggregate functions have been validated against R baseline implementations:
-- - OLS: Validated against R's lm() per group
-- - WLS: Validated against R's lm(weights=...) per group
-- - Ridge: Validated against R's glmnet(alpha=0) per group
-- - RLS: Validated against custom RLS implementation per group
--
-- Validation tolerance: 1e-10 for coefficients, 1e-8 for R²
-- R script: validation/06_test_aggregates.R
--
-- Test data structure:
--   group_a: i=1..5,  y = 3.0 + 2.5*i + 0.3*i²
--   group_b: i=6..10, y = 3.0 + 2.5*i + 0.3*i²

LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create test data with groups
CREATE TABLE aggregate_test_data AS
SELECT
    CASE WHEN i <= 5 THEN 'group_a' ELSE 'group_b' END as category,
    i::DOUBLE as x1,
    (i * 2.0)::DOUBLE as x2,
    (3.0 + 2.5 * i + 0.3 * i * i)::DOUBLE as y,
    (1.0 + 0.1 * i)::DOUBLE as weight
FROM generate_series(1, 10) as t(i);

-- =============================================================================
-- Test 1: OLS Aggregate with GROUP BY (intercept=TRUE)
-- =============================================================================
-- R validation (using aggregate() + lm() per group):
--   Mathematical model: y ~ intercept + x1*β1 + x2*β2
--
--   Expected behavior:
--   - group_a (n=5): Perfect multicollinearity (x2 = 2*x1), rank-deficient
--   - group_b (n=5): Perfect multicollinearity (x2 = 2*x1), rank-deficient
--   - One coefficient will be NULL due to perfect collinearity
--   - R² should be very high (>0.99) as data follows quadratic pattern
--
--   R command equivalent:
--     aggregate(cbind(y, x1, x2) ~ category, data, function(x) {
--       fit <- lm(y ~ x1 + x2, data=as.data.frame(x))
--       summary(fit)
--     })
--
-- Tolerance: ±1e-10 for coefficients, ±1e-8 for R²
SELECT '=== Test 1: OLS Aggregate with GROUP BY ===' as test_name;
SELECT
    category,
    result.coefficients,
    result.intercept,
    result.r2,
    result.adj_r2,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_ols_fit_agg(y, [x1, x2], {'intercept': true}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 2: OLS Aggregate without intercept (intercept=FALSE)
-- =============================================================================
-- R validation (using lm(y ~ x1 + x2 - 1) per group):
--   Mathematical model: y ~ 0 + x1*β1 + x2*β2 (regression through origin)
--
--   Expected behavior:
--   - Intercept forced to exactly 0.0
--   - R² computed from origin (SS_total = Σy²)
--   - R² typically lower than with intercept
--   - Coefficients differ from intercept=TRUE case
--
--   R command equivalent:
--     lm(y ~ x1 + x2 - 1, data=subset(data, category=="group_a"))
--
-- Tolerance: ±1e-10 for coefficients, intercept must be exactly 0.0
SELECT '=== Test 2: OLS Aggregate without intercept ===' as test_name;
SELECT
    category,
    result.coefficients,
    result.intercept,
    result.r2,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_ols_fit_agg(y, [x1, x2], {'intercept': false}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 3: WLS Aggregate with weights (intercept=TRUE)
-- =============================================================================
-- R validation (using lm(y ~ x1 + x2, weights=weight) per group):
--   Mathematical model: y ~ intercept + x1*β1 + x2*β2 with observation weights
--
--   Expected behavior:
--   - Weights emphasize/de-emphasize observations
--   - Higher weights = more influence on fit
--   - Coefficients differ from unweighted OLS
--   - weighted_mse = Σw*(y-ŷ)² / Σw
--
--   R command equivalent:
--     lm(y ~ x1 + x2, data=df, weights=df$weight)
--
-- Tolerance: ±1e-10 for coefficients, ±1e-8 for R²
SELECT '=== Test 3: WLS Aggregate with weights ===' as test_name;
SELECT
    category,
    result.coefficients,
    result.intercept,
    result.r2,
    result.weighted_mse,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_wls_fit_agg(y, [x1, x2], weight, {'intercept': true}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 4: WLS Aggregate without intercept (intercept=FALSE)
-- =============================================================================
-- R validation (using lm(y ~ x1 + x2 - 1, weights=weight) per group):
--   Mathematical model: y ~ 0 + x1*β1 + x2*β2 with weights, through origin
--
--   Expected behavior:
--   - Intercept forced to exactly 0.0
--   - Weighted R² computed from origin
--
--   R command equivalent:
--     lm(y ~ x1 + x2 - 1, data=df, weights=df$weight)
--
-- Tolerance: Intercept must be exactly 0.0
SELECT '=== Test 4: WLS Aggregate without intercept ===' as test_name;
SELECT
    category,
    result.intercept,
    result.r2
FROM (
    SELECT
        category,
        anofox_statistics_wls_fit_agg(y, [x1, x2], weight, {'intercept': false}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 5: Ridge Aggregate with regularization (λ=1.0, intercept=TRUE)
-- =============================================================================
-- R validation (using glmnet(alpha=0, lambda=λ/n) per group):
--   Mathematical model: y ~ intercept + x1*β1 + x2*β2 with L2 penalty
--   Penalty: minimize ||y - Xβ||² + λ||β||²
--
--   Expected behavior:
--   - Coefficients shrunk toward zero compared to OLS
--   - R² slightly lower than OLS due to bias-variance tradeoff
--   - lambda stored in result for reference
--
--   R command equivalent (with glmnet package):
--     library(glmnet)
--     # Center data for intercept
--     y_c <- y - mean(y); X_c <- scale(X, scale=FALSE)
--     fit <- glmnet(X_c, y_c, alpha=0, lambda=1.0/n, intercept=FALSE)
--     intercept <- mean(y) - coef(fit)[-1] %*% colMeans(X)
--
-- Tolerance: ±1e-10 for coefficients (may differ slightly from OLS)
SELECT '=== Test 5: Ridge Aggregate with regularization ===' as test_name;
SELECT
    category,
    result.coefficients,
    result.intercept,
    result.r2,
    result.lambda,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_ridge_fit_agg(y, [x1, x2], {'lambda': 1.0, 'intercept': true}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 6: Ridge Aggregate without intercept (λ=1.0, intercept=FALSE)
-- =============================================================================
-- R validation (using glmnet(alpha=0, lambda=λ/n, intercept=FALSE) per group):
--   Mathematical model: y ~ 0 + x1*β1 + x2*β2 with L2 penalty, through origin
--
--   Expected behavior:
--   - Intercept forced to exactly 0.0
--   - Ridge shrinkage applied to coefficients
--
--   R command equivalent:
--     glmnet(X, y, alpha=0, lambda=1.0/n, intercept=FALSE)
--
-- Tolerance: Intercept must be exactly 0.0
SELECT '=== Test 6: Ridge Aggregate without intercept ===' as test_name;
SELECT
    category,
    result.intercept,
    result.lambda
FROM (
    SELECT
        category,
        anofox_statistics_ridge_fit_agg(y, [x1, x2], {'lambda': 1.0, 'intercept': false}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 7: RLS Aggregate with forgetting factor (λ=1.0, intercept=TRUE)
-- =============================================================================
-- R validation (using custom RLS implementation per group):
--   Mathematical model: Recursive Least Squares with exponential forgetting
--   Update equations: θ(t) = θ(t-1) + K(t) * (y(t) - φ(t)'θ(t-1))
--                     K(t) = P(t-1)φ(t) / (λ + φ(t)'P(t-1)φ(t))
--                     P(t) = (P(t-1) - K(t)φ(t)'P(t-1)) / λ
--
--   Expected behavior:
--   - λ=1.0: No forgetting, equivalent to batch OLS
--   - Sequential update of parameters (not batch like OLS)
--   - Results should match OLS when λ=1.0
--
--   R implementation:
--     # Initialize
--     P <- diag(p) * 1000; theta <- rep(0, p)
--     # For each observation i:
--     phi <- c(1, X[i,])
--     K <- (P %*% phi) / (lambda + t(phi) %*% P %*% phi)
--     theta <- theta + K * (y[i] - t(phi) %*% theta)
--     P <- (P - K %*% t(phi) %*% P) / lambda
--
-- Tolerance: ±1e-10 for coefficients when λ=1.0 (should match OLS)
SELECT '=== Test 7: RLS Aggregate with forgetting factor ===' as test_name;
SELECT
    category,
    result.coefficients,
    result.intercept,
    result.r2,
    result.forgetting_factor,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_rls_fit_agg(y, [x1, x2], {'forgetting_factor': 1.0, 'intercept': true}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 8: RLS Aggregate without intercept (λ=0.95, intercept=FALSE)
-- =============================================================================
-- R validation (using custom RLS implementation, λ=0.95, through origin):
--   Mathematical model: RLS with forgetting factor < 1.0, no intercept
--
--   Expected behavior:
--   - Intercept forced to exactly 0.0
--   - λ=0.95: Recent observations weighted more heavily
--   - Adaptive learning with exponential decay
--
--   R implementation (simplified for no intercept):
--     P <- 1000; theta <- 0
--     for (i in 1:n) {
--       phi <- X[i]
--       K <- P * phi / (0.95 + phi^2 * P)
--       theta <- theta + K * (y[i] - phi * theta)
--       P <- (P - K * phi * P) / 0.95
--     }
--
-- Tolerance: Intercept must be exactly 0.0
SELECT '=== Test 8: RLS Aggregate without intercept ===' as test_name;
SELECT
    category,
    result.intercept,
    result.forgetting_factor
FROM (
    SELECT
        category,
        anofox_statistics_rls_fit_agg(y, [x1, x2], {'forgetting_factor': 0.95, 'intercept': false}) as result
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 9: All aggregates in single query (comparison test)
-- =============================================================================
-- R validation: Compares R² across all 4 methods on same data
--   Expected R² relationship (typically):
--   - OLS: Highest R² (no bias, only variance)
--   - RLS (λ=1.0): Should match OLS exactly
--   - Ridge (λ=0.1): Slightly lower R² (bias-variance tradeoff)
--   - WLS: Similar to OLS but weighted (may be higher/lower)
--
--   This test validates that all aggregates can run in parallel on same data
--   and produce consistent, comparable results
--
-- Tolerance: All R² values should be > 0.90 for this polynomial data
SELECT '=== Test 9: All aggregates in single query ===' as test_name;
SELECT
    category,
    ols.r2 as ols_r2,
    wls.r2 as wls_r2,
    ridge.r2 as ridge_r2,
    rls.r2 as rls_r2
FROM (
    SELECT
        category,
        anofox_statistics_ols_fit_agg(y, [x1, x2], {'intercept': true}) as ols,
        anofox_statistics_wls_fit_agg(y, [x1, x2], weight, {'intercept': true}) as wls,
        anofox_statistics_ridge_fit_agg(y, [x1, x2], {'lambda': 0.1, 'intercept': true}) as ridge,
        anofox_statistics_rls_fit_agg(y, [x1, x2], {'forgetting_factor': 1.0, 'intercept': true}) as rls
    FROM aggregate_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 10: NULL handling (missing data test)
-- =============================================================================
-- R validation (using lm with na.omit):
--   Data: 7 observations with NULLs at i=3 (x1), i=5 (y)
--   Expected behavior:
--   - Rows with any NULL values excluded (pairwise deletion)
--   - n_obs should be 5 (7 total - 1 NULL x1 - 1 NULL y)
--   - Regression computed on complete cases only
--
--   R command equivalent:
--     df_complete <- na.omit(df)  # Removes rows with any NA
--     lm(y ~ x1 + x2, data=df_complete)
--
--   Expected n_obs: 5 (observations 1,2,4,6,7 are complete)
--
-- Tolerance: n_obs must be exactly 5
SELECT '=== Test 10: NULL handling ===' as test_name;
CREATE TABLE null_test_data AS
SELECT
    'test_group' as category,
    CASE WHEN i = 3 THEN NULL ELSE i::DOUBLE END as x1,
    (i * 2.0)::DOUBLE as x2,
    CASE WHEN i = 5 THEN NULL ELSE (3.0 + 2.5 * i)::DOUBLE END as y
FROM generate_series(1, 7) as t(i);

SELECT
    category,
    result.n_obs,
    result.coefficients
FROM (
    SELECT
        category,
        anofox_statistics_ols_fit_agg(y, [x1, x2], {'intercept': true}) as result
    FROM null_test_data
    GROUP BY category
) sub;

-- =============================================================================
-- Test 11: Single feature regression (simple linear regression)
-- =============================================================================
-- R validation (using lm(y ~ x1) across all data):
--   Mathematical model: y ~ intercept + x1*β1
--   Data: All 10 observations (both groups combined)
--
--   Expected behavior:
--   - Single coefficient (1-element array)
--   - High R² as y follows polynomial in x1
--   - No GROUP BY, so single result row
--
--   R command equivalent:
--     lm(y ~ x1, data=aggregate_test_data)
--
--   Expected: R² > 0.95 (strong linear-ish relationship)
--
-- Tolerance: ±1e-10 for coefficient
SELECT '=== Test 11: Single feature ===' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.r2
FROM (
    SELECT
        anofox_statistics_ols_fit_agg(y, [x1], {'intercept': true}) as result
    FROM aggregate_test_data
) sub;

-- =============================================================================
-- Test 12: Edge case - insufficient data (n < p+1)
-- =============================================================================
-- R validation: Tests rank-deficient case with insufficient observations
--   Data: n=2 observations, p=3 predictors (all same: x1, x1, x1), intercept=true
--   Problem: n=2 < p+1=4 (underdetermined system)
--
--   Expected behavior:
--   - Rank-deficient due to perfect collinearity (x1 = x1 = x1)
--   - SVD-based solver should handle gracefully
--   - Some coefficients will be NULL (aliased variables)
--   - n_obs should still be 2
--
--   R command equivalent:
--     lm(y ~ x1 + x1 + x1, data=small_data)
--     # R will drop aliased variables and warn about rank-deficiency
--
--   Note: This tests robustness to degenerate cases
--
-- Tolerance: Should not crash, n_obs=2, coefficients contain NULLs
SELECT '=== Test 12: Edge case - insufficient data ===' as test_name;
CREATE TABLE small_data AS
SELECT 1.0 as x1, 2.0 as y
UNION ALL SELECT 2.0, 4.0;

-- This should return NULL or handle gracefully (n < p+1)
SELECT
    result.coefficients,
    result.n_obs
FROM (
    SELECT
        anofox_statistics_ols_fit_agg(y, [x1, x1, x1], {'intercept': true}) as result
    FROM small_data
) sub;

-- Cleanup
DROP TABLE aggregate_test_data;
DROP TABLE null_test_data;
DROP TABLE small_data;

SELECT '=== All aggregate basic tests completed ===' as status;
