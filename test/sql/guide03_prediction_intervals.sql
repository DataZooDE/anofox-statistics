LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Use the predict function for prediction intervals
SELECT
    predicted,
    ci_lower,
    ci_upper,
    ci_upper - ci_lower as interval_width
FROM anofox_statistics_predict_ols(
    [50.0, 55.0, 60.0, 65.0, 70.0]::DOUBLE[],           -- y_train: historical_sales
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],    -- x_train: historical_features
    [[6.0], [7.0], [8.0]]::DOUBLE[][],                  -- x_new: future_features
    0.95,                                                 -- confidence_level
    'prediction',                                         -- interval_type
    true                                                  -- intercept
);
