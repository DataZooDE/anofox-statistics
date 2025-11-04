-- Model metadata table
CREATE TABLE model_registry (
    model_id VARCHAR PRIMARY KEY,
    model_name VARCHAR,
    model_type VARCHAR,
    training_data_query TEXT,
    dependent_variable VARCHAR,
    independent_variables VARCHAR[],
    training_date TIMESTAMP,
    training_observations BIGINT,
    r_squared DOUBLE,
    coefficients DOUBLE[],
    business_owner VARCHAR,
    use_case TEXT,
    refresh_frequency VARCHAR,
    validation_checks TEXT
);
