LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample diagnostics data
CREATE TEMP TABLE diagnostics AS
SELECT
    i as obs_id,
    (10 + i * 0.5)::DOUBLE as predicted,
    (random() * 2 - 1)::DOUBLE as residual,
    (random() * 2 - 1)::DOUBLE as studentized_residual,
    ABS(random() * 2 - 1)::DOUBLE as sqrt_abs_residual,
    (random() * 0.5)::DOUBLE as leverage,
    (random() * 0.3)::DOUBLE as cooks_distance
FROM generate_series(1, 50) t(i);

-- 1. Residuals vs Fitted
SELECT
    predicted,
    residual
FROM diagnostics;
-- Look for: Random scatter (good), patterns (bad)

-- 2. Q-Q Plot (normal quantiles)
SELECT
    obs_id,
    studentized_residual,
    PERCENT_RANK() OVER (ORDER BY studentized_residual) as percentile
FROM diagnostics;
-- Look for: Points on diagonal line

-- 3. Scale-Location (homoscedasticity)
SELECT
    predicted,
    SQRT(ABS(studentized_residual)) as sqrt_abs_std_resid
FROM diagnostics;
-- Look for: Horizontal band (good), funnel (heteroscedastic)

-- 4. Leverage vs Residuals
SELECT
    leverage,
    studentized_residual,
    cooks_distance
FROM diagnostics;
-- Look for: High leverage + high residual = influential
