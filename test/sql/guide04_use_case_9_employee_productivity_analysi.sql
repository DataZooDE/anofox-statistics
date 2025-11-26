LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample employee productivity data
CREATE OR REPLACE TABLE employee_productivity AS
SELECT
    i as employee_id,
    department,
    training_hours,
    experience_years,
    team_size,
    output_per_hour
FROM (
    SELECT
        i,
        CASE (i % 4)
            WHEN 0 THEN 'Manufacturing'
            WHEN 1 THEN 'Customer Service'
            WHEN 2 THEN 'IT'
            ELSE 'Sales'
        END as department,
        (10 + RANDOM() * 30)::DOUBLE as training_hours,  -- 10-40 hours training
        (1 + RANDOM() * 14)::DOUBLE as experience_years,  -- 1-15 years
        (3 + RANDOM() * 7)::INT::DOUBLE as team_size,  -- 3-10 people
        output as output_per_hour
    FROM (
        SELECT
            i,
            (50 + training * 2.5 + experience * 1.8 + team * 0.5 + RANDOM() * 10)::DOUBLE as output,
            training,
            experience,
            team
        FROM (
            SELECT
                i,
                (10 + RANDOM() * 30) as training,
                (1 + RANDOM() * 14) as experience,
                (3 + RANDOM() * 7) as team
            FROM range(1, 201) t(i)
        )
    )
);

-- Analyze productivity drivers by department - focus on training impact
SELECT
    department,
    ROUND((anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1], 2) as training_impact,
    ROUND((anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).r_squared, 3) as model_fit,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1] > 5.0 THEN 'High Training ROI'
        WHEN (anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1] > 2.0 THEN 'Medium Training ROI'
        ELSE 'Low Training ROI'
    END as training_effectiveness,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1] > 3.0 THEN 'Increase Training Budget'
        ELSE 'Maintain Current Level'
    END as budget_recommendation,
    COUNT(*) as sample_size
FROM employee_productivity
GROUP BY department
ORDER BY training_impact DESC;
