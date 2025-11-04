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
