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
    SQRT(ABS(std_residual)) as sqrt_abs_std_resid
FROM diagnostics;
-- Look for: Horizontal band (good), funnel (heteroscedastic)

-- 4. Leverage vs Residuals
SELECT
    leverage,
    studentized_residual,
    cooks_distance
FROM diagnostics;
-- Look for: High leverage + high residual = influential
