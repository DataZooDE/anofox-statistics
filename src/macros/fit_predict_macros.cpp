#include "anofox_statistics_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb_compat.hpp"

namespace duckdb {

// Structure for defining table macros
struct FitPredictTableMacro {
	const char *name;
	const char *parameters[8]; // Positional parameters (nullptr terminated)
	struct {
		const char *name;
		const char *default_value;
	} named_params[8]; // Named parameters with defaults
	const char *macro; // SQL definition
};

// clang-format off
static const FitPredictTableMacro fit_predict_table_macros[] = {
    // The fit-predict macros hold rows out with the aggregate's split argument, not by NULLing y:
    // under null_policy 'zero' a NULL y is fitted as 0, so it cannot mark a held-out row (#8).
    // The augment macros (no split argument) pass NaN instead, which 'zero' does not fill.
    // ols_fit_predict_by: OLS fit and predict per group (long format - one row per observation)
    // C++ API: ols_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: fit_intercept, confidence_level, null_policy
    // Returns: all source columns (incl. y_col) + yhat, yhat_lower, yhat_upper, is_training
    // Note: Output column preserves the original column name passed by the user
    {"ols_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        ols_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // huber_fit_predict_by: Huber M-estimator fit and predict per group (long format)
    // C++ API: huber_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: epsilon, alpha, max_iterations, tolerance, fit_intercept, confidence_level, null_policy
    // Returns: all source columns (incl. y_col) + yhat, yhat_lower, yhat_upper, is_training
    // Note: Output column preserves the original column name passed by the user
    {"huber_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        huber_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // ransac_fit_predict_by: RANSAC robust fit and predict per group (long format)
    // C++ API: ransac_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: residual_threshold, max_trials, min_samples, stop_probability, stop_n_inliers,
    //          random_state, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"ransac_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        ransac_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // theil_sen_fit_predict_by: Theil-Sen robust fit and predict per group (long format)
    // C++ API: theil_sen_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: max_subpopulation, n_subsamples, max_iterations, tolerance, random_state,
    //          fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"theil_sen_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        theil_sen_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // ridge_fit_predict_by: Ridge fit and predict per group (long format)
    // C++ API: ridge_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: alpha, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"ridge_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        ridge_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // elasticnet_fit_predict_by: ElasticNet fit and predict per group (long format)
    // C++ API: elasticnet_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: alpha, l1_ratio, max_iterations, tolerance, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"elasticnet_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        elasticnet_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // wls_fit_predict_by: WLS fit and predict per group (long format)
    // C++ API: wls_fit_predict_by(table_name, group_col, y_col, x_cols, weight_col, options)
    // Options: fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"wls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", "weight_col", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        wls_fit_predict_agg(y_col, x_cols, weight_col, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // ols_augment_by: per-row diagnostics of an OLS fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: ols_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, null_policy, solver
    // Returns: all source columns (incl. the group column) + model_id, model_name, row_id, fitted, residual, resid_type, std_residual,
    //          stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, loo_residual, loo_fitted
    //          (leave-one-out, #220: residual / (1 - leverage) and y - loo_residual; NULL where leverage = 1 or off the fit)
    {"ols_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'ols' AS model_id,
    'ols' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).loo_residual AS loo_residual,
    (_aug[_rn]).loo_fitted AS loo_fitted
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        ols_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col ELSE 'NaN'::DOUBLE END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // ridge_augment_by: per-row leverage and exact leave-one-out of a Ridge fit per group (contract `obs` schema, #235)
    // C++ API: ridge_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: alpha / lambda, lambda_scaling, fit_intercept, null_policy, solver
    // Returns: as ols_augment_by; leverage is the ridge hat diagonal (intercept unpenalised), loo_residual / loo_fitted the exact
    //          leave-one-out at the same penalty; std_residual, stud_residual, cooks_d and dffits are NULL
    {"ridge_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'ridge' AS model_id,
    'ridge' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).loo_residual AS loo_residual,
    (_aug[_rn]).loo_fitted AS loo_fitted
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        ridge_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col ELSE 'NaN'::DOUBLE END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // wls_augment_by: per-row diagnostics of a WLS fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: wls_augment_by(table_name, group_col, y_col, x_cols, weight_col, options, split, order_by)
    // Options: fit_intercept, null_policy, solver
    // Returns: as ols_augment_by; weight is the prior weight from weight_col; loo_residual is on the response scale
    {"wls_augment_by", {"source", "group_col", "y_col", "x_cols", "weight_col", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'wls' AS model_id,
    'wls' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).loo_residual AS loo_residual,
    (_aug[_rn]).loo_fitted AS loo_fitted
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        wls_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col ELSE 'NaN'::DOUBLE END, x_cols, weight_col, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // poisson_augment_by: per-row diagnostics of a Poisson GLM fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: poisson_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, max_iterations, tolerance, offset, poisson_link (glm_lambda > 0 / prior: error)
    // Returns: as ols_augment_by without loo_*; fitted on the response scale, residual = deviance residual, weight = 1,
    //          plus approx_loo_fitted / approx_loo_residual (one-step approximate leave-one-out, #235)
    {"poisson_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'poisson' AS model_id,
    'poisson' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).approx_loo_fitted AS approx_loo_fitted,
    (_aug[_rn]).approx_loo_residual AS approx_loo_residual
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        poisson_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // binomial_augment_by: per-row diagnostics of a Binomial GLM fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: binomial_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, max_iterations, tolerance, offset, binomial_link (glm_lambda > 0 / prior: error)
    // Returns: as ols_augment_by without loo_*; fitted on the response scale, residual = deviance residual, weight = 1,
    //          plus approx_loo_fitted / approx_loo_residual (one-step approximate leave-one-out, #235)
    {"binomial_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'binomial' AS model_id,
    'binomial' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).approx_loo_fitted AS approx_loo_fitted,
    (_aug[_rn]).approx_loo_residual AS approx_loo_residual
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        binomial_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // logistic_augment_by: per-row diagnostics of a logistic regression fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: logistic_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, max_iterations, tolerance, offset (glm_lambda > 0 / prior: error)
    // Returns: as ols_augment_by without loo_*; fitted on the response scale, residual = deviance residual, weight = 1,
    //          plus approx_loo_fitted / approx_loo_residual (one-step approximate leave-one-out, #235)
    {"logistic_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'logistic' AS model_id,
    'logistic' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).approx_loo_fitted AS approx_loo_fitted,
    (_aug[_rn]).approx_loo_residual AS approx_loo_residual
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        logistic_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // negbinom_augment_by: per-row diagnostics of a Negative Binomial GLM fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: negbinom_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, max_iterations, tolerance, offset, theta (glm_lambda > 0 / prior: error)
    // Returns: as ols_augment_by without loo_*; fitted on the response scale, residual = deviance residual, weight = 1,
    //          plus approx_loo_fitted / approx_loo_residual (one-step approximate leave-one-out, #235)
    {"negbinom_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'negbinom' AS model_id,
    'negbinom' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).approx_loo_fitted AS approx_loo_fitted,
    (_aug[_rn]).approx_loo_residual AS approx_loo_residual
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        negbinom_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // gamma_augment_by: per-row diagnostics of a Gamma GLM (log link) fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: gamma_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, max_iterations, tolerance, offset (glm_lambda > 0 / prior: error)
    // Returns: as ols_augment_by without loo_*; fitted on the response scale, residual = deviance residual, weight = 1,
    //          plus approx_loo_fitted / approx_loo_residual (one-step approximate leave-one-out, #235)
    {"gamma_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'gamma' AS model_id,
    'gamma' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).approx_loo_fitted AS approx_loo_fitted,
    (_aug[_rn]).approx_loo_residual AS approx_loo_residual
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        gamma_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // tweedie_augment_by: per-row diagnostics of a Tweedie GLM (log link) fit per group (broom augment, contract `obs` schema, #151)
    // C++ API: tweedie_augment_by(table_name, group_col, y_col, x_cols, options, split, order_by)
    // Options: fit_intercept, max_iterations, tolerance, offset, power (glm_lambda > 0 / prior: error)
    // Returns: as ols_augment_by without loo_*; fitted on the response scale, residual = deviance residual, weight = 1,
    //          plus approx_loo_fitted / approx_loo_residual (one-step approximate leave-one-out, #235)
    {"tweedie_augment_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_aug, _rn, _anofox_rid),
    'tweedie' AS model_id,
    'tweedie' AS model_name,
    _anofox_rid AS row_id,
    (_aug[_rn]).fitted AS fitted,
    (_aug[_rn]).residual AS residual,
    (_aug[_rn]).resid_type AS resid_type,
    (_aug[_rn]).std_residual AS std_residual,
    (_aug[_rn]).stud_residual AS stud_residual,
    (_aug[_rn]).leverage AS leverage,
    (_aug[_rn]).cooks_d AS cooks_d,
    (_aug[_rn]).dffits AS dffits,
    (_aug[_rn]).weight AS weight,
    (_aug[_rn]).is_training AS is_training,
    (_aug[_rn]).n_params AS n_params,
    (_aug[_rn]).approx_loo_fitted AS approx_loo_fitted,
    (_aug[_rn]).approx_loo_residual AS approx_loo_residual
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        tweedie_augment_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, row_id
)"},

    // rls_fit_predict_by: RLS fit and predict per group (long format)
    // C++ API: rls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: forgetting_factor, initial_p_diagonal, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"rls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        rls_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // bls_fit_predict_by: BLS (Bounded Least Squares) fit and predict per group (long format)
    // C++ API: bls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: lower_bound, upper_bound, intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"bls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        bls_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // alm_fit_predict_by: ALM (Augmented Linear Model) fit and predict per group (long format)
    // C++ API: alm_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: distribution, intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"alm_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        alm_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // glmm_fit_by: mixed-effects GLM with a random intercept (issue #107)
    // C++ API: glmm_fit_by(source, group_col, y_col, x_cols, options)
    // Unlike the *_fit_predict_by macros this fits ONE model across all groups and
    // returns the per-group random effects (BLUPs), which is the point of partial
    // pooling: every group's estimate borrows strength from the others.
    // Returns: one row per group -- group, ranef, ranef_se, n, plus the shared
    // fixed effects and variance components.
    {"glmm_fit_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    r.group AS group,
    r.intercept AS ranef,
    r.se AS ranef_se,
    r.n AS n,
    sub._fit.intercept AS fixed_intercept,
    sub._fit.coefficients AS fixed_coefficients,
    sub._fit.var_group AS var_group,
    sub._fit.var_residual AS var_residual,
    sub._fit.icc AS icc
FROM (
    SELECT glmm_fit_agg(y_col, x_cols, group_col, options) AS _fit
    FROM query_table(source::VARCHAR)
) sub, LATERAL UNNEST(sub._fit.ranef) AS u(r)
ORDER BY r.group
)"},

    // ols_anova_by: nested-model F test per group (issue #221), R's
    // anova(lm(y ~ x_reduced), lm(y ~ x_full)) on the same rows.
    // C++ API: ols_anova_by(source, group_col, y_col, x_reduced := [...], x_full := [...], options)
    // x_reduced defaults to [] (the intercept-only model: the overall F test).
    // Returns: one row per group -- the group column, id (the group as text),
    // model_id / model_name 'ols' (#187), the contract `test` fields method,
    // statistic, p_value, df1, df2, then rss_reduced, rss_full,
    // adj_r_squared_reduced, adj_r_squared_full, delta_adj_r_squared and n_observations.
    {"ols_anova_by", {"source", "group_col", "y_col", nullptr}, {{"x_reduced", "[]"}, {"x_full", "NULL"}, {"options", "NULL"}},
R"(
SELECT
    group_col,
    group_col::VARCHAR AS id,
    'ols' AS model_id,
    'ols' AS model_name,
    (_t).method AS method,
    (_t).statistic AS statistic,
    (_t).p_value AS p_value,
    (_t).df1 AS df1,
    (_t).df2 AS df2,
    (_t).rss_reduced AS rss_reduced,
    (_t).rss_full AS rss_full,
    (_t).adj_r_squared_reduced AS adj_r_squared_reduced,
    (_t).adj_r_squared_full AS adj_r_squared_full,
    (_t).delta_adj_r_squared AS delta_adj_r_squared,
    (_t).n_observations AS n_observations
FROM (
    SELECT group_col, ols_anova_agg(y_col, x_reduced, x_full, options) AS _t
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
) sub
ORDER BY group_col
)"},

    // ols_anova_table_by: per-term ANOVA table per group (issue #231), R's
    // anova(lm(y ~ x)) (type I) or car::Anova(fit, type = 2) (type II).
    // C++ API: ols_anova_table_by(source, group_col, y_col, x_cols, options)
    // options: type ('I' | 'II'), terms (one label per x column), fit_intercept.
    // Returns: one row per group and term -- the group column, id (the group as
    // text), model_id / model_name 'ols' (#187), term, the contract `test` fields
    // method, statistic, p_value, df1, df2, then sum_sq, mean_sq,
    // residual_sum_sq, residual_mean_sq and n_observations.
    {"ols_anova_table_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    group_col,
    group_col::VARCHAR AS id,
    'ols' AS model_id,
    'ols' AS model_name,
    (_t).term AS term,
    (_t).method AS method,
    (_t).statistic AS statistic,
    (_t).p_value AS p_value,
    (_t).df1 AS df1,
    (_t).df2 AS df2,
    (_t).sum_sq AS sum_sq,
    (_t).mean_sq AS mean_sq,
    (_t).residual_sum_sq AS residual_sum_sq,
    (_t).residual_mean_sq AS residual_mean_sq,
    (_t).n_observations AS n_observations
FROM (
    SELECT group_col, unnest(range(len(_l))) AS _i, unnest(_l) AS _t
    FROM (
        SELECT group_col, ols_anova_table_agg(y_col, x_cols, options) AS _l
        FROM query_table(source::VARCHAR)
        GROUP BY group_col
    )
) sub
ORDER BY group_col, _i
)"},

    // eb_shrink_by: empirical-Bayes shrinkage of per-group estimates (issue #107)
    // C++ API: eb_shrink_by(source, estimate_col, se_col, options)
    // Unlike the *_fit_predict_by macros this does not fit a model -- it consumes
    // estimates that already exist (typically one row per group from a GROUP BY
    // fit) and shrinks each toward the precision-weighted mean.
    // Returns: all source columns + shrunken, shrunken_se, weight, mu, tau_squared
    {"eb_shrink_by", {"source", "estimate_col", "se_col", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_res, _rn),
    (_res.shrunken[_rn]).shrunken AS shrunken,
    (_res.shrunken[_rn]).shrunken_se AS shrunken_se,
    (_res.shrunken[_rn]).weight AS weight,
    _res.mu AS mu,
    _res.tau_squared AS tau_squared
FROM (
    SELECT *,
        ROW_NUMBER() OVER () AS _rn,
        eb_shrink_agg(estimate_col, se_col, options) OVER () AS _res
    FROM query_table(source::VARCHAR)
) sub
)"},

    // p_adjust_by: multiple-testing correction per group (issue #219)
    // C++ API: p_adjust_by(source, group_col, p_col, method := 'holm', n := NULL)
    // Adjusts the p-values of each group with p_adjust_agg (R p.adjust) and aligns
    // them back to their rows through a row id: one aggregate per group, so the
    // cost is linear (a windowed list per row would be quadratic).
    // Returns: all source columns + p_adjusted, in source order
    {"p_adjust_by", {"source", "group_col", "p_col", nullptr}, {{"method", "'holm'"}, {"n", "NULL"}},
R"(
WITH _anofox_src AS MATERIALIZED (
    SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR)
),
_anofox_adj AS (
    SELECT
        UNNEST(list(_anofox_rid ORDER BY _anofox_rid)) AS _anofox_rid,
        UNNEST(p_adjust_agg(p_col, method, n ORDER BY _anofox_rid)) AS p_adjusted
    FROM _anofox_src
    GROUP BY group_col
)
SELECT _anofox_src.* EXCLUDE (_anofox_rid), _anofox_adj.p_adjusted
FROM _anofox_src JOIN _anofox_adj USING (_anofox_rid)
ORDER BY _anofox_src._anofox_rid
)"},

    // mediation_by: causal mediation analysis per group (issue #223)
    // C++ API: mediation_by(source, group_col, treatment, mediator, outcome, covariates, options)
    // Fits m ~ x + covariates and y ~ x + m + covariates per group with mediation_agg
    // and returns the contract `terms` columns, one row per effect (acme, ade, total,
    // prop_mediated), plus method, n_obs and n_sims. id is the group, model_id 'mediation' (#187).
    {"mediation_by", {"source", "group_col", "treatment", "mediator", "outcome", nullptr}, {{"covariates", "NULL"}, {"options", "NULL"}},
R"(
SELECT
    group_col, id, model_id, model_name,
    (_t).term AS term,
    (_t).estimate AS estimate,
    (_t).std_error AS std_error,
    (_t).statistic AS statistic,
    (_t).p_value AS p_value,
    (_t).conf_low AS conf_low,
    (_t).conf_high AS conf_high,
    (_t).conf_level AS conf_level,
    NULL::VARCHAR AS index_name,
    NULL::DOUBLE AS index_value,
    (_t).method AS method,
    (_t).n_obs AS n_obs,
    (_t).n_sims AS n_sims
FROM (
    SELECT group_col, group_col::VARCHAR AS id, 'mediation' AS model_id, 'mediation' AS model_name, unnest(_m) AS _t
    FROM (
        SELECT group_col,
            mediation_agg(outcome::DOUBLE, mediator::DOUBLE, treatment::DOUBLE,
                          coalesce(covariates, []::DOUBLE[])::DOUBLE[], options) AS _m
        FROM query_table(source::VARCHAR)
        GROUP BY group_col
    )
)
ORDER BY group_col, list_position(['acme', 'ade', 'total', 'prop_mediated'], term)
)"},

    // poisson_fit_predict_by: Poisson GLM fit and predict per group (long format)
    // C++ API: poisson_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: link, intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"poisson_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        poisson_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // binomial_fit_predict_by: Binomial GLM (success rates in [0,1]; link via options) fit per group, predictions on the response scale.
    // C++ API: binomial_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits binomial_fit_agg on each group's training rows (y not NULL and split omitted or 'train' / 'training' in any case) and
    // applies predict(model, x) to every row. Returns all source columns + yhat, yhat_lower,
    // yhat_upper (NULL: no interval for GLMs), is_training.
    {"binomial_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}},
R"(
WITH _anofox_src AS (
    SELECT * FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, binomial_fit_agg(y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND SPLIT_IS_TRAIN
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND SPLIT_IS_TRAIN) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // logistic_fit_predict_by: Logistic regression (binary y; predictions are probabilities) fit per group, predictions on the response scale.
    // C++ API: logistic_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits logistic_fit_agg on each group's training rows (y not NULL and split omitted or 'train' / 'training' in any case) and
    // applies predict(model, x) to every row. Returns all source columns + yhat, yhat_lower,
    // yhat_upper (NULL: no interval for GLMs), is_training.
    {"logistic_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}},
R"(
WITH _anofox_src AS (
    SELECT * FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, logistic_fit_agg(y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND SPLIT_IS_TRAIN
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND SPLIT_IS_TRAIN) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // negbinom_fit_predict_by: Negative Binomial GLM (overdispersed counts, log link) fit per group, predictions on the response scale.
    // C++ API: negbinom_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits negbinom_fit_agg on each group's training rows (y not NULL and split omitted or 'train' / 'training' in any case) and
    // applies predict(model, x) to every row. Returns all source columns + yhat, yhat_lower,
    // yhat_upper (NULL: no interval for GLMs), is_training.
    {"negbinom_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}},
R"(
WITH _anofox_src AS (
    SELECT * FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, negbinom_fit_agg(y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND SPLIT_IS_TRAIN
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND SPLIT_IS_TRAIN) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // gamma_fit_predict_by: Gamma GLM (strictly positive y, log link) fit per group, predictions on the response scale.
    // C++ API: gamma_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits gamma_fit_agg on each group's training rows (y not NULL and split omitted or 'train' / 'training' in any case) and
    // applies predict(model, x) to every row. Returns all source columns + yhat, yhat_lower,
    // yhat_upper (NULL: no interval for GLMs), is_training.
    {"gamma_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}},
R"(
WITH _anofox_src AS (
    SELECT * FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, gamma_fit_agg(y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND SPLIT_IS_TRAIN
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND SPLIT_IS_TRAIN) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // tweedie_fit_predict_by: Tweedie GLM (non-negative skewed y, log link) fit per group, predictions on the response scale.
    // C++ API: tweedie_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits tweedie_fit_agg on each group's training rows (y not NULL and split omitted or 'train' / 'training' in any case) and
    // applies predict(model, x) to every row. Returns all source columns + yhat, yhat_lower,
    // yhat_upper (NULL: no interval for GLMs), is_training.
    {"tweedie_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}},
R"(
WITH _anofox_src AS (
    SELECT * FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, tweedie_fit_agg(y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND SPLIT_IS_TRAIN
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND SPLIT_IS_TRAIN) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // pls_fit_predict_by: PLS (Partial Least Squares) fit and predict per group (long format)
    // C++ API: pls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: n_components, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"pls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        pls_fit_predict_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // isotonic_fit_predict_by: Isotonic regression fit and predict per group (long format)
    // C++ API: isotonic_fit_predict_by(table_name, group_col, y_col, x_col, options)
    // Note: Isotonic takes a single x column, not a list
    // Options: increasing, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"isotonic_fit_predict_by", {"source", "group_col", "y_col", "x_col", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        isotonic_fit_predict_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_col, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // bayesian_ridge_fit_predict_by: Bayesian Ridge fit and predict per group (long format, #61)
    // Options: fit_intercept, max_iterations, tolerance, alpha_1, alpha_2, lambda_1, lambda_2,
    //          confidence_level (posterior predictive interval)
    {"bayesian_ridge_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        bayesian_ridge_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // ard_fit_predict_by: ARD fit and predict per group (long format, #61)
    // Options: fit_intercept, max_iterations, tolerance, alpha_1, alpha_2, lambda_1, lambda_2, threshold_lambda,
    //          confidence_level (posterior predictive interval)
    {"ard_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        ard_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // passive_aggressive_fit_predict_by: Passive-Aggressive fit and predict per group (long format, #63)
    // Options: c, epsilon, variant, fit_intercept, max_iterations, tolerance. Rows are processed in order_by order (sequential model).
    {"passive_aggressive_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        passive_aggressive_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // lm_dynamic_fit_predict_by: lmDynamic (time-varying coefficients) fit and predict per group (long format, #63)
    // Options: ic, span, fit_intercept. Rows are processed in order_by order (sequential model).
    {"lm_dynamic_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        lm_dynamic_fit_predict_agg(y_col, x_cols, CASE WHEN SPLIT_IS_TRAIN THEN 'train' ELSE 'test' END, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // quantile_fit_predict_by: Quantile regression fit and predict per group (long format)
    // C++ API: quantile_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: tau, fit_intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"quantile_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}, {"split", "NULL"}, {"order_by", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn, _anofox_rid),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _rn,
        quantile_fit_predict_agg(CASE WHEN SPLIT_IS_TRAIN THEN y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // aid_by: AID (Automatic Identification of Demand) classification per group (wide format - one row per group)
    // C++ API: aid_by(table_name, group_col, y_col, options)
    // Options: intermittent_threshold, outlier_method
    // Returns: <group_col>, demand_type, is_intermittent, distribution, mean, variance, zero_proportion,
    //          n_observations, has_stockouts, is_new_product, is_obsolete_product, stockout_count,
    //          new_product_count, obsolete_product_count, high_outlier_count, low_outlier_count
    // Note: Output column preserves the original column name passed by the user
    // order_by (#271): AID reads the series in order (leading / trailing zeros mark new and
    // obsolete products), so pass the time column for a reproducible result. coalesce(): a bare
    // ORDER BY order_by would expand to ORDER BY NULL, which aggregates reject.
    {"aid_by", {"source", "group_col", "y_col", nullptr}, {{"options", "NULL"}, {"order_by", "NULL"}},
R"(
WITH agg AS (
    SELECT
        group_col,
        aid_agg(y_col, options ORDER BY coalesce(order_by)) AS result
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
)
SELECT
    group_col,
    (result).demand_type AS demand_type,
    (result).is_intermittent AS is_intermittent,
    (result).distribution AS distribution,
    (result).mean AS mean,
    (result).variance AS variance,
    (result).zero_proportion AS zero_proportion,
    (result).n_observations AS n_observations,
    (result).has_stockouts AS has_stockouts,
    (result).is_new_product AS is_new_product,
    (result).is_obsolete_product AS is_obsolete_product,
    (result).stockout_count AS stockout_count,
    (result).new_product_count AS new_product_count,
    (result).obsolete_product_count AS obsolete_product_count,
    (result).high_outlier_count AS high_outlier_count,
    (result).low_outlier_count AS low_outlier_count
FROM agg
ORDER BY group_col
)"},

    // aid_anomaly_by: AID anomaly detection per group (long format - one row per observation)
    // C++ API: aid_anomaly_by(table_name, group_col, order_col, y_col, options)
    // Options: intermittent_threshold, outlier_method
    // Returns: <group_col>, <order_col>, stockout, new_product, obsolete_product, high_outlier, low_outlier
    // Note: Output columns preserve the original column names passed by the user
    {"aid_anomaly_by", {"source", "group_col", "order_col", "y_col", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    group_col,
    order_col,
    (anomaly_flags[row_num]).stockout AS stockout,
    (anomaly_flags[row_num]).new_product AS new_product,
    (anomaly_flags[row_num]).obsolete_product AS obsolete_product,
    (anomaly_flags[row_num]).high_outlier AS high_outlier,
    (anomaly_flags[row_num]).low_outlier AS low_outlier
FROM (
    SELECT
        group_col,
        order_col,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_col) AS row_num,
        aid_anomaly_agg(y_col, options ORDER BY order_col) OVER (PARTITION BY group_col) AS anomaly_flags
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col, order_col
)"},

    // Sentinel
    {nullptr, {nullptr}, {{nullptr, nullptr}}, nullptr}
};
// clang-format on


// Metadata (description, example, category) shown in duckdb_functions() for each table macro.
struct FitPredictTableMacroDoc {
	const char *name;
	const char *description;
	const char *example;
	const char *category;
};

// clang-format off
static const FitPredictTableMacroDoc fit_predict_table_macro_docs[] = {
    {"ols_fit_predict_by", "Table macro: fits an OLS regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "ols_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"huber_fit_predict_by", "Table macro: fits a Huber M-estimator regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "huber_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"ransac_fit_predict_by", "Table macro: fits a RANSAC robust regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "ransac_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"theil_sen_fit_predict_by", "Table macro: fits a Theil-Sen regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "theil_sen_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"ridge_fit_predict_by", "Table macro: fits a Ridge regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "ridge_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"elasticnet_fit_predict_by", "Table macro: fits an ElasticNet regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "elasticnet_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"wls_fit_predict_by", "Table macro: fits a weighted least squares (WLS) regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "wls_fit_predict_by('my_table', group_col, y, [x1, x2], weight)", "regression"},
    {"ols_augment_by", "Table macro: fits an OLS regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, loo_residual and loo_fitted (exact leave-one-out: residual / (1 - leverage) and y minus it, NULL where leverage = 1). Non-training rows get the prediction as fitted and NULL diagnostics.", "ols_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"wls_augment_by", "Table macro: fits a weighted least squares (WLS) regression per group of source (training rows: y_col NOT NULL, a positive weight and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, loo_residual and loo_fitted (exact leave-one-out on the response scale: residual / (1 - leverage) and y minus it, NULL where leverage = 1). Non-training rows get the prediction as fitted and NULL diagnostics.", "wls_augment_by('my_table', group_col, y, [x1, x2], w)", "regression"},
    {"ridge_augment_by", "Table macro: fits a Ridge regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the contract obs columns: model_id and model_name, row_id, fitted, residual, resid_type, leverage (diagonal of the ridge hat matrix X(X'X + lambda I)^-1 X', intercept unpenalised), weight, is_training, n_params, loo_residual and loo_fitted (exact leave-one-out at the same penalty: residual / (1 - leverage) and y minus it). std_residual, stud_residual, cooks_d and dffits are NULL. Non-training rows get the prediction as fitted and NULL diagnostics.", "ridge_augment_by('my_table', group_col, y, [x1, x2], {'alpha': 1.0})", "regression"},
    {"poisson_augment_by", "Table macro: fits a Poisson GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted (response scale), residual (deviance), resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, approx_loo_fitted and approx_loo_residual (one-step approximate leave-one-out prediction and residual, response scale: not the refit). Non-training rows get the prediction as fitted and NULL diagnostics. Penalized fits (glm_lambda > 0 or a prior) are rejected.", "poisson_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"binomial_augment_by", "Table macro: fits a Binomial GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted (response scale), residual (deviance), resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, approx_loo_fitted and approx_loo_residual (one-step approximate leave-one-out prediction and residual, response scale: not the refit). Non-training rows get the prediction as fitted and NULL diagnostics. Penalized fits (glm_lambda > 0 or a prior) are rejected.", "binomial_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"logistic_augment_by", "Table macro: fits a logistic regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted (response scale), residual (deviance), resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, approx_loo_fitted and approx_loo_residual (one-step approximate leave-one-out prediction and residual, response scale: not the refit). Non-training rows get the prediction as fitted and NULL diagnostics. Penalized fits (glm_lambda > 0 or a prior) are rejected.", "logistic_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"negbinom_augment_by", "Table macro: fits a Negative Binomial GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted (response scale), residual (deviance), resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, approx_loo_fitted and approx_loo_residual (one-step approximate leave-one-out prediction and residual, response scale: not the refit). Non-training rows get the prediction as fitted and NULL diagnostics. Penalized fits (glm_lambda > 0 or a prior) are rejected.", "negbinom_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"gamma_augment_by", "Table macro: fits a Gamma GLM (log link) per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted (response scale), residual (deviance), resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, approx_loo_fitted and approx_loo_residual (one-step approximate leave-one-out prediction and residual, response scale: not the refit). Non-training rows get the prediction as fitted and NULL diagnostics. Penalized fits (glm_lambda > 0 or a prior) are rejected.", "gamma_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"tweedie_augment_by", "Table macro: fits a Tweedie GLM (log link) per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with the per-observation diagnostics of the fit (broom augment, contract obs schema): model_id and model_name (the method; the group column is among the source columns), row_id, fitted (response scale), residual (deviance), resid_type, std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training, n_params, approx_loo_fitted and approx_loo_residual (one-step approximate leave-one-out prediction and residual, response scale: not the refit). Non-training rows get the prediction as fitted and NULL diagnostics. Penalized fits (glm_lambda > 0 or a prior) are rejected.", "tweedie_augment_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"rls_fit_predict_by", "Table macro: fits a Recursive Least Squares (RLS) regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "rls_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"bls_fit_predict_by", "Table macro: fits a bounded/non-negative least squares (BLS) regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "bls_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"alm_fit_predict_by", "Table macro: fits an Augmented Linear Model (ALM) per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "alm_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"poisson_fit_predict_by", "Table macro: fits a Poisson GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "poisson_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"binomial_fit_predict_by", "Table macro: fits a binomial GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "binomial_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"logistic_fit_predict_by", "Table macro: fits a logistic regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "logistic_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"negbinom_fit_predict_by", "Table macro: fits a negative binomial GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "negbinom_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"gamma_fit_predict_by", "Table macro: fits a Gamma GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "gamma_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"tweedie_fit_predict_by", "Table macro: fits a Tweedie GLM per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "tweedie_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"pls_fit_predict_by", "Table macro: fits a partial least squares (PLS) regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "pls_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"isotonic_fit_predict_by", "Table macro: fits an isotonic regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "isotonic_fit_predict_by('my_table', group_col, y, x)", "regression"},
    {"quantile_fit_predict_by", "Table macro: fits a quantile regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "quantile_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"glmm_fit_by", "Table macro: fits one generalized linear mixed model (random intercept per group_col) over the whole table and returns one row per group with its random effect and the fixed effects and variance components.", "glmm_fit_by('my_table', group_col, y, [x1, x2])", "mixed-models"},
    {"ols_anova_by", "Table macro: nested-model F test per group of source (R's anova(lm(y ~ x_reduced), lm(y ~ x_full)) on the same rows); returns one row per group with method 'nested F', statistic, p_value, df1, df2, both RSS values and the adjusted R^2 of both models. x_reduced := [] (the default) is the intercept-only model.", "ols_anova_by('sales', region, y, x_reduced := [price], x_full := [price, promo, temp])", "regression"},
    {"ols_anova_table_by", "Table macro: per-term ANOVA table per group of source (R's anova(lm(y ~ x)), type I sequential sums of squares in the column order, or car::Anova(fit, type = 2) with options {'type': 'II'}); returns one row per group and term with term, method, statistic (F), p_value, df1, df2 (residual df), sum_sq, mean_sq and the residual sum of squares. options terms: one label per x column, equal labels form one term (e.g. factor dummies).", "ols_anova_table_by('sales', region, y, [price, promo, temp], options := {'type': 'II'})", "regression"},
    {"eb_shrink_by", "Table macro: empirical-Bayes shrinkage of the estimates in estimate_col (standard errors in se_col); returns every source row with shrunken, shrunken_se, weight, mu and tau_squared appended.", "eb_shrink_by('my_table', estimate, se)", "shrinkage"},
    {"p_adjust_by", "Table macro: adjusts the p-values in p_col for multiple testing within each group of group_col (R p.adjust; method holm (default), hochberg, hommel, bonferroni, BH/fdr, BY or none; n: number of hypotheses per group, default the number of non-NULL p-values); returns every source row, in source order, with p_adjusted appended. NULL p-values stay NULL and are not counted.", "p_adjust_by('results', endpoint, p_value, method := 'BH')", "multiple-testing"},
    {"mediation_by", "Table macro: causal mediation analysis per group of source (linear models, as R mediation::mediate): fits mediator ~ treatment + covariates and outcome ~ treatment + mediator + covariates and returns one row per group and effect (acme, ade, total, prop_mediated) in the contract terms schema plus method, n_obs and n_sims. Inference by nonparametric bootstrap (default, n_bootstrap 1000, seed 42), Sobel or quasi-Bayesian Monte Carlo (options method, ci_type, n_bootstrap, seed, confidence_level, treat_value, control_value).", "mediation_by('trial', site, treat, m, y, covariates := [age, baseline])", "mediation"},
    {"aid_by", "Table macro: Automatic Identification of Demand (AID); classifies the demand pattern of each group of source and returns one row per group. Pass order_by := <time column> (by name): AID reads each series in order.", "aid_by('sales', product_id, demand, order_by := week)", "demand-analysis"},
    {"aid_anomaly_by", "Table macro: AID anomaly flags per group of source, ordered by order_col; returns one row per input row with stockout, new_product, obsolete_product, high_outlier and low_outlier flags.", "aid_anomaly_by('sales', product_id, date, demand)", "demand-analysis"},
    {"bayesian_ridge_fit_predict_by", "Table macro: fits a Bayesian Ridge regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat, yhat_lower / yhat_upper (posterior predictive interval at confidence_level, default 0.95) and is_training appended.", "bayesian_ridge_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"ard_fit_predict_by", "Table macro: fits an ARD (automatic relevance determination) regression per group of source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat, yhat_lower / yhat_upper (posterior predictive interval at confidence_level, default 0.95) and is_training appended.", "ard_fit_predict_by('my_table', group_col, y, [x1, x2, x3])", "regression"},
    {"passive_aggressive_fit_predict_by", "Table macro: fits a Passive-Aggressive online regression per group of source, updating row by row in order_by order (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case), and returns every source row with yhat from the final coefficients, yhat_lower / yhat_upper (NULL: no interval) and is_training appended.", "passive_aggressive_fit_predict_by('my_table', group_col, y, [x1, x2], order_by := t)", "regression"},
    {"lm_dynamic_fit_predict_by", "Table macro: fits a dynamic linear model with time-varying coefficients (greybox lmDynamic) per group of source in order_by order (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every source row with yhat (the row's own time-varying fit; rows outside the fit use the time-averaged coefficients), yhat_lower / yhat_upper (NULL) and is_training appended.", "lm_dynamic_fit_predict_by('my_table', group_col, y, [x1, x2], order_by := t)", "regression"},
    {nullptr, nullptr, nullptr, nullptr}
};
// clang-format on

// SPLIT_IS_TRAIN in a macro body: whether a row is a training row by its split, as the
// *_fit_predict_agg split_col (#205): 'train' / 'training' in any case. With split omitted
// (the NULL literal default, typed "NULL") every row is a training row; a split column's
// per-row NULL is a prediction row. Never NULL, so it can be returned as is_training.
static const char *const SPLIT_IS_TRAIN_SQL =
    "coalesce(lower(split::VARCHAR) IN ('train', 'training'), replace(typeof(split), '\"', '') = 'NULL')";

// Helper function to create a table macro from the definition
static unique_ptr<CreateMacroInfo> CreateFitPredictTableMacro(const FitPredictTableMacro &macro_def) {
	// Parse the SQL
	auto parser = MakeParser();
	parser.ParseQuery(StringUtil::Replace(macro_def.macro, "SPLIT_IS_TRAIN", SPLIT_IS_TRAIN_SQL));
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("Expected a single select statement in CreateFitPredictTableMacro");
	}
	auto node = std::move(parser.statements[0]->Cast<SelectStatement>().node);

	// Create the macro function
	auto function = make_uniq<TableMacroFunction>(std::move(node));

	// Add positional parameters
	for (idx_t i = 0; macro_def.parameters[i] != nullptr; i++) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(AsDuckName(macro_def.parameters[i])));
	}

	// Add named parameters with defaults
	for (idx_t i = 0; macro_def.named_params[i].name != nullptr; i++) {
		const auto &param = macro_def.named_params[i];
		function->parameters.push_back(make_uniq<ColumnRefExpression>(AsDuckName(param.name)));

		// Parse the default value
		auto expr_list = parser.ParseExpressionList(param.default_value);
		if (!expr_list.empty()) {
			function->default_parameters.insert(make_pair(AsDuckName(param.name), std::move(expr_list[0])));
		}
	}

	// Create the macro info
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	SetCreateInfoName(*info, DEFAULT_SCHEMA, macro_def.name);
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(function));

	for (idx_t i = 0; fit_predict_table_macro_docs[i].name != nullptr; i++) {
		const auto &doc = fit_predict_table_macro_docs[i];
		if (macro_def.name != string(doc.name)) {
			continue;
		}
		FunctionDescription description;
		description.description = doc.description;
		description.examples = {doc.example};
		description.categories = {doc.category, "table-macro"};
		for (idx_t p = 0; macro_def.parameters[p] != nullptr; p++) {
			description.parameter_names.push_back(macro_def.parameters[p]);
		}
		info->descriptions.push_back(std::move(description));
		break;
	}

	return info;
}

void RegisterFitPredictTableMacros(ExtensionLoader &loader) {
	for (idx_t i = 0; fit_predict_table_macros[i].name != nullptr; i++) {
		auto info = CreateFitPredictTableMacro(fit_predict_table_macros[i]);
		loader.RegisterFunction(*info);
	}
}

} // namespace duckdb
