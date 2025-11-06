-- Variance proportional to x (positional parameters, literal arrays)
SELECT * FROM anofox_statistics_wls(
    [50.0, 100.0, 150.0, 200.0, 250.0]::DOUBLE[],  -- y: sales
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],      -- x1: size
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],      -- weights: proportional to size
    true                                             -- add_intercept
);
