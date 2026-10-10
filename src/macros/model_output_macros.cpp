// <model>_tidy_by and <model>_glance_by: long-form model outputs per group (#152),
// the integration contract's `terms` and `summary` schemas.
//
//   SELECT * FROM ols_tidy_by('sales', store, revenue, [price, promo], names := ['price', 'promo']);
//   SELECT * FROM ols_glance_by('sales', store, revenue, [price, promo]);
//
// Both fit one model per group with <model>_fit_agg, so they accept the same
// options. tidy_by computes inference by default where the model has it.

#include "anofox_statistics_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

namespace duckdb {

namespace {

struct ModelOutputModel {
	//! Prefix of <prefix>_fit_agg and of the generated macros.
	const char *prefix;
	//! Human-readable model name for the descriptions.
	const char *label;
	//! Extra column <prefix>_fit_agg takes after x (WLS weights, AFT event), or nullptr.
	const char *extra_col;
	//! Whether the model has coefficient inference (accepts compute_inference);
	//! <prefix>_tidy_by then computes it by default.
	bool inference;
};

// clang-format off
const ModelOutputModel MODEL_OUTPUT_MODELS[] = {
    {"ols", "an OLS regression", nullptr, true},
    {"wls", "a weighted least squares regression", "weight_col", true},
    {"ridge", "a Ridge regression", nullptr, true},
    {"elasticnet", "an Elastic Net regression", nullptr, false},
    {"huber", "a Huber M-estimator regression", nullptr, true},
    {"ransac", "a RANSAC regression", nullptr, true},
    {"theil_sen", "a Theil-Sen regression", nullptr, true},
    {"rls", "a recursive least squares regression", nullptr, false},
    {"lars", "a LARS / Lasso-LARS regression", nullptr, false},
    {"bls", "a bounded least squares regression", nullptr, false},
    {"nnls", "a non-negative least squares regression", nullptr, false},
    {"pls", "a partial least squares regression", nullptr, false},
    {"quantile", "a quantile regression", nullptr, false},
    {"poisson", "a Poisson GLM", nullptr, true},
    {"binomial", "a binomial GLM", nullptr, true},
    {"logistic", "a logistic regression", nullptr, true},
    {"negbinom", "a negative binomial GLM", nullptr, true},
    {"gamma", "a Gamma GLM", nullptr, true},
    {"tweedie", "a Tweedie GLM", nullptr, true},
    {"alm", "an augmented linear model (ALM)", nullptr, true},
    {"aft", "an accelerated failure time (AFT) survival model", "event_col", true},
};
// clang-format on

// The glance metrics emitted by <model>_glance_by, in order.
const char *const GLANCE_METRICS[] = {"n_observations", "n_features",     "r_squared",        "adj_r_squared",
                                      "residual_std_error", "f_statistic", "f_pvalue",         "log_likelihood",
                                      "aic",            "bic",            "deviance",         "null_deviance",
                                      "pseudo_r_squared", "dispersion",   "iterations"};

string FitCall(const ModelOutputModel &m) {
	return string(m.prefix) + "_fit_agg(y_col, x_cols" + (m.extra_col ? string(", ") + m.extra_col : string()) +
	       ", options)";
}

string TidySql(const ModelOutputModel &m) {
	return "SELECT group_col, model_id, (_t).term AS term, (_t).estimate AS estimate, (_t).std_error AS std_error, "
	       "(_t).statistic AS statistic, (_t).p_value AS p_value, (_t).conf_low AS conf_low, "
	       "(_t).conf_high AS conf_high, (_t).conf_level AS conf_level, (_t).index_name AS index_name, "
	       "(_t).index_value AS index_value "
	       "FROM (SELECT group_col, group_col::VARCHAR AS model_id, unnest(tidy(_m, names)) AS _t "
	       "FROM (SELECT group_col, " +
	       FitCall(m) +
	       " AS _m FROM query_table(source::VARCHAR) GROUP BY group_col)) "
	       "ORDER BY group_col";
}

string GlanceSql(const ModelOutputModel &m) {
	string metrics;
	for (auto metric : GLANCE_METRICS) {
		if (!metrics.empty()) {
			metrics += ", ";
		}
		metrics += "{'metric': '" + string(metric) + "', 'value': (_g)." + metric + "::DOUBLE}";
	}
	return "SELECT group_col, model_id, model_type, (_mv).metric AS metric, (_mv).value AS value "
	       "FROM (SELECT group_col, group_col::VARCHAR AS model_id, (_g).model_type AS model_type, unnest([" +
	       metrics +
	       "]) AS _mv FROM (SELECT group_col, glance(" + FitCall(m) +
	       ") AS _g FROM query_table(source::VARCHAR) GROUP BY group_col)) "
	       "WHERE (_mv).value IS NOT NULL ORDER BY group_col, metric";
}

unique_ptr<CreateMacroInfo> CreateTableMacro(const string &name, const vector<string> &parameters,
                                             const vector<pair<string, string>> &named_parameters, const string &sql,
                                             const string &description, const string &example) {
	Parser parser;
	parser.ParseQuery(sql);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("Expected a single select statement for table macro %s", name);
	}
	auto function = make_uniq<TableMacroFunction>(std::move(parser.statements[0]->Cast<SelectStatement>().node));
	for (auto &p : parameters) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(p));
	}
	for (auto &np : named_parameters) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(np.first));
		auto defaults = Parser::ParseExpressionList(np.second);
		function->default_parameters.insert(make_pair(np.first, std::move(defaults[0])));
	}

	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = name;
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(function));

	FunctionDescription desc;
	desc.description = description;
	desc.examples = {example};
	desc.categories = {"regression", "table-macro"};
	desc.parameter_names = parameters;
	info->descriptions.push_back(std::move(desc));
	return info;
}

} // namespace

void RegisterModelOutputMacros(ExtensionLoader &loader) {
	for (auto &m : MODEL_OUTPUT_MODELS) {
		vector<string> params = {"source", "group_col", "y_col", "x_cols"};
		if (m.extra_col) {
			params.push_back(m.extra_col);
		}
		string args = string("'my_table', group_col, y, [x1, x2]") +
		              (!m.extra_col ? "" : string(m.extra_col) == "weight_col" ? ", w" : ", event");
		string prefix = m.prefix;

		auto tidy = CreateTableMacro(
		    prefix + "_tidy_by", params,
		    {{"options", m.inference ? "{'compute_inference': true}" : "NULL"}, {"names", "NULL"}}, TidySql(m),
		    "Table macro: fits " + string(m.label) +
		        " per group of source and returns one row per term (intercept first): the group column, model_id, "
		        "term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, "
		        "index_value. " +
		        (m.inference ? "Inference is computed by default. " : "The model has no coefficient inference. ") +
		        "names := [...] labels the terms.",
		    prefix + "_tidy_by(" + args + ", names := ['x1', 'x2'])");
		loader.RegisterFunction(*tidy);

		auto glance = CreateTableMacro(
		    prefix + "_glance_by", params, {{"options", "NULL"}}, GlanceSql(m),
		    "Table macro: fits " + string(m.label) +
		        " per group of source and returns its fit statistics in long form, one row per available metric: "
		        "the group column, model_id, model_type, metric, value.",
		    prefix + "_glance_by(" + args + ")");
		loader.RegisterFunction(*glance);
	}
}

} // namespace duckdb
