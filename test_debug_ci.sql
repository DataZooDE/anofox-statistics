-- Test confidence interval calculation
LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create the model
CREATE TEMP TABLE model AS
SELECT * FROM anofox_statistics_ols(
    [2.1, 4.2, 5.9, 8.1, 10.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    {'intercept': true, 'full_output': true}
);

-- Check the model metadata
SELECT
    'Model Metadata' as section,
    round(intercept::DOUBLE, 6) as intercept,
    round(coefficients[1]::DOUBLE, 6) as slope,
    round(mse::DOUBLE, 10) as mse,
    round(rmse::DOUBLE, 10) as rmse,
    rank::BIGINT as rank,
    df_residual::BIGINT as df_residual,
    n_obs::BIGINT as n_obs,
    round(intercept_std_error::DOUBLE, 10) as intercept_se,
    round(coefficient_std_errors[1]::DOUBLE, 10) as coef_se
FROM model;

-- Make predictions with confidence intervals
SELECT
    'Predictions' as section,
    observation_id,
    round(predicted::DOUBLE, 6) as pred,
    round(ci_lower::DOUBLE, 6) as ci_low,
    round(ci_upper::DOUBLE, 6) as ci_high,
    round(se::DOUBLE, 10) as se
FROM model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [6.0]::DOUBLE[],
    0.95,
    'confidence'
) p;
