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
    // Returns: group_id, y, x, yhat, yhat_lower, yhat_upper, is_training
    {"ols_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        ols_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // ridge_fit_predict_by: Ridge fit and predict per group (long format)
    // C++ API: ridge_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: alpha, fit_intercept, confidence_level, null_policy
    {"ridge_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        ridge_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // elasticnet_fit_predict_by: ElasticNet fit and predict per group (long format)
    // C++ API: elasticnet_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: alpha, l1_ratio, max_iterations, tolerance, fit_intercept, confidence_level, null_policy
    {"elasticnet_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        elasticnet_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // wls_fit_predict_by: WLS fit and predict per group (long format)
    // C++ API: wls_fit_predict_by(table_name, group_col, y_col, x_cols, weight_col, options)
    // Options: fit_intercept, confidence_level, null_policy
    {"wls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", "weight_col", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        wls_fit_predict_agg(y_col, x_cols, weight_col, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // rls_fit_predict_by: RLS fit and predict per group (long format)
    // C++ API: rls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: forgetting_factor, initial_p_diagonal, fit_intercept, confidence_level, null_policy
    {"rls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        rls_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // bls_fit_predict_by: BLS (Bounded Least Squares) fit and predict per group (long format)
    // C++ API: bls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: lower_bound, upper_bound, intercept, max_iterations, tolerance, confidence_level, null_policy
    {"bls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        bls_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // alm_fit_predict_by: ALM (Augmented Linear Model) fit and predict per group (long format)
    // C++ API: alm_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: distribution, intercept, max_iterations, tolerance, confidence_level, null_policy
    {"alm_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        alm_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // poisson_fit_predict_by: Poisson GLM fit and predict per group (long format)
    // C++ API: poisson_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: link, intercept, max_iterations, tolerance, confidence_level, null_policy
    {"poisson_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        poisson_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // pls_fit_predict_by: PLS (Partial Least Squares) fit and predict per group (long format)
    // C++ API: pls_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: n_components, fit_intercept, confidence_level, null_policy
    {"pls_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        pls_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // isotonic_fit_predict_by: Isotonic regression fit and predict per group (long format)
    // C++ API: isotonic_fit_predict_by(table_name, group_col, y_col, x_col, options)
    // Note: Isotonic takes a single x column, not a list
    // Options: increasing, confidence_level, null_policy
    {"isotonic_fit_predict_by", {"source", "group_col", "y_col", "x_col", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        isotonic_fit_predict_agg(y_col, x_col, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // quantile_fit_predict_by: Quantile regression fit and predict per group (long format)
    // C++ API: quantile_fit_predict_by(table_name, group_col, y_col, x_cols, options)
    // Options: tau, fit_intercept, max_iterations, tolerance, confidence_level, null_policy
    {"quantile_fit_predict_by", {"source", "group_col", "y_col", "x_cols", nullptr}, {{"options", "NULL"}},
R"(
WITH predictions AS (
    SELECT
        group_col AS group_id,
        quantile_fit_predict_agg(y_col, x_cols, options) AS pred
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(pred) AS p
    FROM predictions
)
SELECT
    group_id,
    (p).y AS y,
    (p).x AS x,
    (p).yhat AS yhat,
    (p).yhat_lower AS yhat_lower,
    (p).yhat_upper AS yhat_upper,
    (p).is_training AS is_training
FROM unnested
ORDER BY group_id
)"},

    // aid_by: AID (Automatic Identification of Demand) classification per group (wide format - one row per group)
    // C++ API: aid_by(table_name, group_col, y_col, options)
    // Options: intermittent_threshold, outlier_method
    // Returns: group_id, demand_type, is_intermittent, distribution, mean, variance, zero_proportion,
    //          n_observations, has_stockouts, is_new_product, is_obsolete_product, stockout_count,
    //          new_product_count, obsolete_product_count, high_outlier_count, low_outlier_count
    {"aid_by", {"source", "group_col", "y_col", nullptr}, {{"options", "NULL"}},
R"(
WITH agg AS (
    SELECT
        group_col AS group_id,
        aid_agg(y_col, options) AS result
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
)
SELECT
    group_id,
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
ORDER BY group_id
)"},

    // aid_anomaly_by: AID anomaly detection per group (long format - one row per observation)
    // C++ API: aid_anomaly_by(table_name, group_col, order_col, y_col, options)
    // Options: intermittent_threshold, outlier_method
    // Returns: group_id, order_value, stockout, new_product, obsolete_product, high_outlier, low_outlier
    {"aid_anomaly_by", {"source", "group_col", "order_col", "y_col", nullptr}, {{"options", "NULL"}},
R"(
WITH original_data AS (
    SELECT
        group_col AS group_id,
        order_col AS order_value,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_col) AS rn
    FROM query_table(source::VARCHAR)
),
anomalies AS (
    SELECT
        group_col AS group_id,
        aid_anomaly_agg(y_col, options ORDER BY order_col) AS flags
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
),
unnested AS (
    SELECT
        group_id,
        UNNEST(flags) AS f,
        generate_subscripts(flags, 1) AS rn
    FROM anomalies
)
SELECT
    o.group_id,
    o.order_value,
    (u.f).stockout AS stockout,
    (u.f).new_product AS new_product,
    (u.f).obsolete_product AS obsolete_product,
    (u.f).high_outlier AS high_outlier,
    (u.f).low_outlier AS low_outlier
FROM original_data o
JOIN unnested u ON o.group_id = u.group_id AND o.rn = u.rn
ORDER BY o.group_id, o.order_value
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
