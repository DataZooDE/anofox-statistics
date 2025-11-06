-- Recursive Least Squares (positional parameters, literal arrays)
SELECT * FROM anofox_statistics_rls(
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]::DOUBLE[],  -- y: streaming_values
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],        -- x1: streaming_features
    0.99::DOUBLE,                                      -- forgetting_factor (explicit cast required)
    true::BOOLEAN                                      -- add_intercept
);
