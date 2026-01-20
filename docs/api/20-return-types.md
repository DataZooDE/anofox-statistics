# Return Types

Standard return structures used by regression and statistical functions.

## FitResult Structure

Returned by all regression `*_fit` and `*_fit_agg` functions.

```
STRUCT(
    coefficients LIST(DOUBLE),      -- Model coefficients
    intercept DOUBLE,               -- Intercept term (if fit_intercept=true)
    r_squared DOUBLE,               -- Coefficient of determination
    adj_r_squared DOUBLE,           -- Adjusted R²
    mse DOUBLE,                     -- Mean squared error
    rmse DOUBLE,                    -- Root mean squared error
    mae DOUBLE,                     -- Mean absolute error
    n BIGINT,                       -- Number of observations
    p BIGINT,                       -- Number of features

    -- Inference fields (if compute_inference=true)
    std_errors LIST(DOUBLE),        -- Standard errors
    t_statistics LIST(DOUBLE),      -- t-statistics
    p_values LIST(DOUBLE),          -- p-values
    ci_lower LIST(DOUBLE),          -- CI lower bounds
    ci_upper LIST(DOUBLE),          -- CI upper bounds
    f_statistic DOUBLE,             -- F-statistic
    f_p_value DOUBLE                -- F-test p-value
)
```

## TestResult Structure

Returned by hypothesis testing functions.

```
STRUCT(
    statistic DOUBLE,    -- Test statistic
    p_value DOUBLE,      -- p-value
    df DOUBLE,           -- Degrees of freedom (if applicable)
    effect_size DOUBLE,  -- Effect size measure
    ci_lower DOUBLE,     -- CI lower bound
    ci_upper DOUBLE,     -- CI upper bound
    n BIGINT,            -- Sample size
    method VARCHAR       -- Test method name
)
```

## CorrelationResult Structure

Returned by correlation functions.

```
STRUCT(
    coefficient DOUBLE,  -- Correlation coefficient
    p_value DOUBLE,      -- p-value
    ci_lower DOUBLE,     -- CI lower bound
    ci_upper DOUBLE,     -- CI upper bound
    n BIGINT,            -- Sample size
    method VARCHAR       -- Correlation method
)
```
