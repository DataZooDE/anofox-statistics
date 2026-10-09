#include "anofox_statistics_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/function/table_macro_function.hpp"

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
        ols_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        huber_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        ransac_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        theil_sen_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        ridge_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        elasticnet_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        wls_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, weight_col, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
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
        rls_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        bls_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        alm_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        poisson_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col
)"},

    // binomial_fit_predict_by: Binomial GLM (success rates in [0,1]; link via options) fit per group, predictions on the response scale.
    // C++ API: binomial_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits binomial_fit_agg on each group's training rows (y not NULL and split NULL or 'train') and
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
    WHERE y_col IS NOT NULL AND (split IS NULL OR split = 'train')
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND (split IS NULL OR split = 'train')) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // logistic_fit_predict_by: Logistic regression (binary y; predictions are probabilities) fit per group, predictions on the response scale.
    // C++ API: logistic_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits logistic_fit_agg on each group's training rows (y not NULL and split NULL or 'train') and
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
    WHERE y_col IS NOT NULL AND (split IS NULL OR split = 'train')
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND (split IS NULL OR split = 'train')) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // negbinom_fit_predict_by: Negative Binomial GLM (overdispersed counts, log link) fit per group, predictions on the response scale.
    // C++ API: negbinom_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits negbinom_fit_agg on each group's training rows (y not NULL and split NULL or 'train') and
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
    WHERE y_col IS NOT NULL AND (split IS NULL OR split = 'train')
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND (split IS NULL OR split = 'train')) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // gamma_fit_predict_by: Gamma GLM (strictly positive y, log link) fit per group, predictions on the response scale.
    // C++ API: gamma_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits gamma_fit_agg on each group's training rows (y not NULL and split NULL or 'train') and
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
    WHERE y_col IS NOT NULL AND (split IS NULL OR split = 'train')
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND (split IS NULL OR split = 'train')) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)"},
    // tweedie_fit_predict_by: Tweedie GLM (non-negative skewed y, log link) fit per group, predictions on the response scale.
    // C++ API: tweedie_fit_predict_by(source, group_col, y_col, x_cols, options := NULL, split := NULL)
    // Fits tweedie_fit_agg on each group's training rows (y not NULL and split NULL or 'train') and
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
    WHERE y_col IS NOT NULL AND (split IS NULL OR split = 'train')
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model),
    predict(_anofox_model, x_cols) AS yhat,
    NULL::DOUBLE AS yhat_lower,
    NULL::DOUBLE AS yhat_upper,
    (y_col IS NOT NULL AND (split IS NULL OR split = 'train')) AS is_training
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
        pls_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        isotonic_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_col, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
        quantile_fit_predict_agg(CASE WHEN split IS NOT NULL AND split != 'train' THEN NULL ELSE y_col END, x_cols, options) OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _pred
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
    {"aid_by", {"source", "group_col", "y_col", nullptr}, {{"options", "NULL"}},
R"(
WITH agg AS (
    SELECT
        group_col,
        aid_agg(y_col, options) AS result
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
    {"ols_fit_predict_by", "Table macro: fits an OLS regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "ols_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"huber_fit_predict_by", "Table macro: fits a Huber M-estimator regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "huber_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"ransac_fit_predict_by", "Table macro: fits a RANSAC robust regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "ransac_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"theil_sen_fit_predict_by", "Table macro: fits a Theil-Sen regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "theil_sen_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"ridge_fit_predict_by", "Table macro: fits a Ridge regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "ridge_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"elasticnet_fit_predict_by", "Table macro: fits an ElasticNet regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "elasticnet_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"wls_fit_predict_by", "Table macro: fits a weighted least squares (WLS) regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "wls_fit_predict_by('my_table', group_col, y, [x1, x2], weight)", "regression"},
    {"rls_fit_predict_by", "Table macro: fits a Recursive Least Squares (RLS) regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "rls_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"bls_fit_predict_by", "Table macro: fits a bounded/non-negative least squares (BLS) regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "bls_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"alm_fit_predict_by", "Table macro: fits an Augmented Linear Model (ALM) per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "alm_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"poisson_fit_predict_by", "Table macro: fits a Poisson GLM per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "poisson_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"binomial_fit_predict_by", "Table macro: fits a binomial GLM per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "binomial_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"logistic_fit_predict_by", "Table macro: fits a logistic regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "logistic_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"negbinom_fit_predict_by", "Table macro: fits a negative binomial GLM per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "negbinom_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"gamma_fit_predict_by", "Table macro: fits a Gamma GLM per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "gamma_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"tweedie_fit_predict_by", "Table macro: fits a Tweedie GLM per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "tweedie_fit_predict_by('my_table', group_col, y, [x1, x2])", "glm"},
    {"pls_fit_predict_by", "Table macro: fits a partial least squares (PLS) regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "pls_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"isotonic_fit_predict_by", "Table macro: fits an isotonic regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "isotonic_fit_predict_by('my_table', group_col, y, x)", "regression"},
    {"quantile_fit_predict_by", "Table macro: fits a quantile regression per group of source (training rows: y_col NOT NULL and split NULL or 'train') and returns every source row with yhat (plus yhat_lower / yhat_upper prediction-interval bounds where the model provides them) and is_training appended.", "quantile_fit_predict_by('my_table', group_col, y, [x1, x2])", "regression"},
    {"glmm_fit_by", "Table macro: fits one generalized linear mixed model (random intercept per group_col) over the whole table and returns one row per group with its random effect and the fixed effects and variance components.", "glmm_fit_by('my_table', group_col, y, [x1, x2])", "mixed-models"},
    {"eb_shrink_by", "Table macro: empirical-Bayes shrinkage of the estimates in estimate_col (standard errors in se_col); returns every source row with shrunken, shrunken_se, weight, mu and tau_squared appended.", "eb_shrink_by('my_table', estimate, se)", "shrinkage"},
    {"aid_by", "Table macro: Automatic Identification of Demand (AID); classifies the demand pattern of each group of source and returns one row per group.", "aid_by('sales', product_id, demand)", "demand-analysis"},
    {"aid_anomaly_by", "Table macro: AID anomaly flags per group of source, ordered by order_col; returns one row per input row with stockout, new_product, obsolete_product, high_outlier and low_outlier flags.", "aid_anomaly_by('sales', product_id, date, demand)", "demand-analysis"},
    {nullptr, nullptr, nullptr, nullptr}
};
// clang-format on

// Helper function to create a table macro from the definition
static unique_ptr<CreateMacroInfo> CreateFitPredictTableMacro(const FitPredictTableMacro &macro_def) {
	// Parse the SQL
	Parser parser;
	parser.ParseQuery(macro_def.macro);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("Expected a single select statement in CreateFitPredictTableMacro");
	}
	auto node = std::move(parser.statements[0]->Cast<SelectStatement>().node);

	// Create the macro function
	auto function = make_uniq<TableMacroFunction>(std::move(node));

	// Add positional parameters
	for (idx_t i = 0; macro_def.parameters[i] != nullptr; i++) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(macro_def.parameters[i]));
	}

	// Add named parameters with defaults
	for (idx_t i = 0; macro_def.named_params[i].name != nullptr; i++) {
		const auto &param = macro_def.named_params[i];
		function->parameters.push_back(make_uniq<ColumnRefExpression>(param.name));

		// Parse the default value
		auto expr_list = Parser::ParseExpressionList(param.default_value);
		if (!expr_list.empty()) {
			function->default_parameters.insert(make_pair(string(param.name), std::move(expr_list[0])));
		}
	}

	// Create the macro info
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = macro_def.name;
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
