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
    const char *parameters[8];          // Positional parameters (nullptr terminated)
    struct {
        const char *name;
        const char *default_value;
    } named_params[8];                  // Named parameters with defaults
    const char *macro;                  // SQL definition
};

// clang-format off
static const FitPredictTableMacro fit_predict_table_macros[] = {
    // ols_fit_predict_by: OLS fit and predict per group (long format - one row per observation)
    // C++ API: ols_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: fit_intercept, confidence_level, null_policy
    // Returns: all source columns (incl. y_col) + yhat, yhat_lower, yhat_upper, is_training
    // Note: Output column preserves the original column name passed by the user
    {"ols_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        ols_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // ridge_fit_predict_by: Ridge fit and predict per group (long format)
    // C++ API: ridge_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: alpha, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"ridge_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        ridge_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // elasticnet_fit_predict_by: ElasticNet fit and predict per group (long format)
    // C++ API: elasticnet_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: alpha, l1_ratio, max_iterations, tolerance, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"elasticnet_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        elasticnet_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // wls_fit_predict_by: WLS fit and predict per group (long format)
    // C++ API: wls_fit_predict_by(table_name, group_col, y_col, x_cols, weight_col, options)
    // Options: fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"wls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", "weight_col", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        wls_fit_predict_agg(y_col, x_cols, weight_col, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // rls_fit_predict_by: RLS fit and predict per group (long format)
    // C++ API: rls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: forgetting_factor, initial_p_diagonal, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"rls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        rls_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // bls_fit_predict_by: BLS (Bounded Least Squares) fit and predict per group (long format)
    // C++ API: bls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: lower_bound, upper_bound, intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"bls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        bls_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // alm_fit_predict_by: ALM (Augmented Linear Model) fit and predict per group (long format)
    // C++ API: alm_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: distribution, intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"alm_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        alm_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // poisson_fit_predict_by: Poisson GLM fit and predict per group (long format)
    // C++ API: poisson_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: link, intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"poisson_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).yhat_lower AS yhat_lower,
    (_pred[_rn]).yhat_upper AS yhat_upper,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        poisson_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // pls_fit_predict_by: PLS (Partial Least Squares) fit and predict per group (long format)
    // C++ API: pls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: n_components, fit_intercept, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"pls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        pls_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // isotonic_fit_predict_by: Isotonic regression fit and predict per group (long format)
    // C++ API: isotonic_fit_predict_by(table_name, group_col, y_col, x_col, options)
    // Note: Isotonic takes a single x column, not a list
    // Options: increasing, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"isotonic_fit_predict_by", {"source", "group_col", "y_col", "x_col", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        isotonic_fit_predict_agg(y_col, x_col, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
) sub
ORDER BY group_col
)"},

    // quantile_fit_predict_by: Quantile regression fit and predict per group (long format)
    // C++ API: quantile_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: tau, fit_intercept, max_iterations, tolerance, confidence_level, null_policy
    // Note: Output column preserves the original column name passed by the user
    {"quantile_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
SELECT
    * EXCLUDE (_pred, _rn),
    (_pred[_rn]).yhat AS yhat,
    (_pred[_rn]).is_training AS is_training
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_col) AS _rn,
        quantile_fit_predict_agg(y_col, x_cols, options) OVER (PARTITION BY group_col) AS _pred
    FROM query_table(source::VARCHAR)
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

    return info;
}

void RegisterFitPredictTableMacros(ExtensionLoader &loader) {
    for (idx_t i = 0; fit_predict_table_macros[i].name != nullptr; i++) {
        auto info = CreateFitPredictTableMacro(fit_predict_table_macros[i]);
        loader.RegisterFunction(*info);
    }
}

} // namespace duckdb
