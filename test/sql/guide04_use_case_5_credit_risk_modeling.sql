LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample loan data
CREATE OR REPLACE TABLE loans AS
SELECT
    i as loan_id,
    CURRENT_DATE - (RANDOM() * 365 * 3)::INT as origination_date,
    CASE WHEN RANDOM() < 0.15 THEN 1 ELSE 0 END as default_flag,  -- 15% default rate
    (650 + RANDOM() * 150)::DOUBLE as credit_score,  -- 650-800
    (0.15 + RANDOM() * 0.35)::DOUBLE as debt_to_income,  -- 15%-50%
    (0.60 + RANDOM() * 0.35)::DOUBLE as loan_to_value,  -- 60%-95%
    (1 + RANDOM() * 19)::DOUBLE as employment_years  -- 1-20 years
FROM range(1, 101) t(i);

-- Build default prediction model using aggregate functions
WITH risk_factors AS (
    SELECT
        'Credit Score' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
    UNION ALL
    SELECT
        'Debt-to-Income' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
    UNION ALL
    SELECT
        'Loan-to-Value' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
    UNION ALL
    SELECT
        'Employment Years' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
)
SELECT *
FROM risk_factors
ORDER BY ABS(coefficient) DESC;
