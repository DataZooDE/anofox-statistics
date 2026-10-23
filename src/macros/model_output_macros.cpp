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
#include "duckdb_compat.hpp"

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
	//! Sequential models (#63): the fit depends on the row order, so the macros
	//! take an order_col after group_col and fit with ORDER BY order_col.
	bool ordered = false;
	//! The coefficients vary over time (lmDynamic): <prefix>_tidy_by returns one
	//! set of terms per observation (DynamicTidySql) instead of tidy().
	bool dynamic_terms = false;
	//! The intervals are Bayesian credible intervals (#222), not confidence intervals.
	bool credible = false;
	//! Extra option defaults of <prefix>_tidy_by, merged under the user's options
	//! (Theil-Sen #222: a smaller bootstrap than the aggregate's default).
	const char *tidy_defaults = nullptr;
	//! The fit has the opt-in leave-one-out statistics (#220): <prefix>_glance_by
	//! turns compute_loo on, so press / press_rmse / loo_r_squared are reported.
	bool loo = false;
};

// clang-format off
const ModelOutputModel MODEL_OUTPUT_MODELS[] = {
    {"ols", "an OLS regression", nullptr, true, false, false, true},
    {"wls", "a weighted least squares regression", "weight_col", true, false, false, true},
    {"ridge", "a Ridge regression", nullptr, true},
    {"elasticnet", "an Elastic Net regression", nullptr, false},
    {"huber", "a Huber M-estimator regression", nullptr, true},
    {"ransac", "a RANSAC regression", nullptr, true},
    {"theil_sen", "a Theil-Sen regression", nullptr, true, false, false, false,
     "'ci_method': 'exact_or_none', 'n_boot': '199'"},
    {"rls", "a recursive least squares regression", nullptr, false},
    {"lars", "a LARS / Lasso-LARS regression", nullptr, false},
    {"bayesian_ridge", "a Bayesian Ridge regression", nullptr, false, false, false, true},
    {"ard", "an ARD (automatic relevance determination) regression", nullptr, false, false, false, true},
    {"bls", "a bounded least squares regression", nullptr, false},
    {"nnls", "a non-negative least squares regression", nullptr, false},
    {"pls", "a partial least squares regression", nullptr, false},
    {"quantile", "a quantile regression", nullptr, true},
    {"poisson", "a Poisson GLM", nullptr, true},
    {"binomial", "a binomial GLM", nullptr, true},
    {"logistic", "a logistic regression", nullptr, true},
    {"negbinom", "a negative binomial GLM", nullptr, true},
    {"gamma", "a Gamma GLM", nullptr, true},
    {"tweedie", "a Tweedie GLM", nullptr, true},
    {"alm", "an augmented linear model (ALM)", nullptr, true},
    {"aft", "an accelerated failure time (AFT) survival model", "event_col", true},
    {"passive_aggressive", "a Passive-Aggressive online regression", nullptr, false, true},
    {"lm_dynamic", "a dynamic linear model with time-varying coefficients (greybox lmDynamic)", nullptr, false, true, true},
};
// clang-format on

// The glance metrics emitted by <model>_glance_by, in order.
const char *const GLANCE_METRICS[] = {"n_observations", "n_features",     "r_squared",        "adj_r_squared",
                                      "residual_std_error", "f_statistic", "f_pvalue",         "log_likelihood",
                                      "aic",            "bic",            "deviance",         "null_deviance",
                                      "pseudo_r_squared", "dispersion",   "iterations",
                                      "press",          "press_rmse",     "loo_r_squared"};

string FitCall(const ModelOutputModel &m) {
	return string(m.prefix) + "_fit_agg(y_col, x_cols" + (m.extra_col ? string(", ") + m.extra_col : string()) +
	       ", options" + (m.ordered ? " ORDER BY order_col" : "") + ")";
}

//! The fit call of <model>_tidy_by: inference on by default (compute_inference := true merged under the
//! user's options, which win), so passing other options does not drop the standard errors.
string TidyFitCall(const ModelOutputModel &m) {
	if (!m.inference) {
		return FitCall(m);
	}
	return string(m.prefix) + "_fit_agg(y_col, x_cols" + (m.extra_col ? string(", ") + m.extra_col : string()) +
	       ", map_concat(MAP {'compute_inference': 'true'" +
	       (m.tidy_defaults ? ", " + string(m.tidy_defaults) : string()) +
	       "}, coalesce(options::MAP(VARCHAR, VARCHAR), "
	       "MAP {}::MAP(VARCHAR, VARCHAR))))";
}

//! model_id / model_name are the method and id the group (#183): the group column is kept as well.
string ModelIds(const string &method) {
	return "group_col::VARCHAR AS id, '" + method + "' AS model_id, '" + method + "' AS model_name";
}

string TidySql(const ModelOutputModel &m) {
	return "SELECT group_col, id, model_id, model_name, (_t).term AS term, (_t).estimate AS estimate, "
	       "(_t).std_error AS std_error, "
	       "(_t).statistic AS statistic, (_t).p_value AS p_value, (_t).conf_low AS conf_low, "
	       "(_t).conf_high AS conf_high, (_t).conf_level AS conf_level, (_t).index_name AS index_name, "
	       "(_t).index_value AS index_value "
	       "FROM (SELECT group_col, " +
	       ModelIds(m.prefix) + ", unnest(tidy(_m, names)) AS _t FROM (SELECT group_col, " + TidyFitCall(m) +
	       " AS _m FROM query_table(source::VARCHAR) GROUP BY group_col)) "
	       "ORDER BY group_col";
}

// Rolling coefficients (#177): <model>_fit_agg over a trailing window of
// window_size rows per group, as `terms` rows with index_name 'window_end'.
// index_value is the 1-based position of the window's last row in its group
// (DOUBLE in the contract); window_end carries the order value itself.
const char *const ROLLING_MODELS[] = {"ols", "wls", "ridge", "elasticnet", "huber", "poisson"};

// lmDynamic (#63): the time-varying coefficients of every observation as `terms`
// rows with index_name 'window_end', the shape of <model>_rolling_tidy_by, so
// the same plot draws them. window_end is the observation's order_col value and
// index_value its 1-based position among the fitted rows of its group.
string DynamicTidySql(const ModelOutputModel &m) {
	const string frame = "OVER (PARTITION BY group_col ORDER BY order_col, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING "
	                     "AND UNBOUNDED FOLLOWING)";
	return "SELECT group_col, id, model_id, model_name, window_end, "
	       "CASE WHEN _j = 0 THEN '(Intercept)' ELSE COALESCE((names::VARCHAR[])[_j], 'x' || _j) END AS term, "
	       "CASE WHEN _j = 0 THEN (_p).intercept ELSE (_p).coefficients[_j] END AS estimate, "
	       "NULL::DOUBLE AS std_error, NULL::DOUBLE AS statistic, NULL::DOUBLE AS p_value, NULL::DOUBLE AS conf_low, "
	       "NULL::DOUBLE AS conf_high, NULL::DOUBLE AS conf_level, 'window_end' AS index_name, "
	       "_pos::DOUBLE AS index_value "
	       "FROM (SELECT group_col, " +
	       ModelIds(m.prefix) +
	       ", window_end, _p, row_number() OVER (PARTITION BY group_col ORDER BY _rn) AS _pos, "
	       "unnest(range(0, len((_p).coefficients) + 1)) AS _j "
	       "FROM (SELECT group_col, window_end, _rn, _pred[_rn] AS _p "
	       "FROM (SELECT group_col, order_col AS window_end, row_number() " +
	       frame + " AS _rn, " + m.prefix + "_fit_predict_agg(y_col, x_cols, options) " + frame +
	       " AS _pred FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR)))) "
	       "WHERE (_p).is_training) "
	       "ORDER BY group_col, window_end, _j";
}

string RollingTidySql(const ModelOutputModel &m) {
	return "SELECT group_col, id, model_id, model_name, window_end, (_t).term AS term, (_t).estimate AS estimate, "
	       "(_t).std_error AS std_error, (_t).statistic AS statistic, (_t).p_value AS p_value, "
	       "(_t).conf_low AS conf_low, (_t).conf_high AS conf_high, (_t).conf_level AS conf_level, "
	       "'window_end' AS index_name, _pos::DOUBLE AS index_value, _n AS n_window "
	       "FROM (SELECT group_col, " +
	       ModelIds(m.prefix) + ", window_end, _pos, _n, unnest(tidy(_m, names)) AS _t "
	       "FROM (SELECT group_col, order_col AS window_end, "
	       "row_number() OVER (PARTITION BY group_col ORDER BY order_col) AS _pos, count(*) OVER _w AS _n, " +
	       TidyFitCall(m) +
	       " OVER _w AS _m FROM query_table(source::VARCHAR) "
	       "WINDOW _w AS (PARTITION BY group_col ORDER BY order_col ROWS BETWEEN window_size - 1 PRECEDING AND "
	       "CURRENT ROW)) "
	       "WHERE _n >= coalesce(min_obs, window_size)) "
	       "ORDER BY group_col, window_end";
}

//! The fit call of <model>_glance_by: compute_loo := true merged under the user's options for the models
//! with leave-one-out statistics (#220).
string GlanceFitCall(const ModelOutputModel &m) {
	if (!m.loo) {
		return FitCall(m);
	}
	return string(m.prefix) + "_fit_agg(y_col, x_cols" + (m.extra_col ? string(", ") + m.extra_col : string()) +
	       ", map_concat(MAP {'compute_loo': 'true'}, coalesce(options::MAP(VARCHAR, VARCHAR), "
	       "MAP {}::MAP(VARCHAR, VARCHAR))))";
}

string GlanceSql(const ModelOutputModel &m) {
	string metrics;
	for (auto metric : GLANCE_METRICS) {
		if (!metrics.empty()) {
			metrics += ", ";
		}
		metrics += "{'metric': '" + string(metric) + "', 'value': (_g)." + metric + "::DOUBLE}";
	}
	return "SELECT group_col, id, model_id, model_name, model_type, (_mv).metric AS metric, (_mv).value AS value "
	       "FROM (SELECT group_col, " +
	       ModelIds(m.prefix) + ", (_g).model_type AS model_type, unnest([" +
	       metrics +
	       "]) AS _mv FROM (SELECT group_col, glance(" + GlanceFitCall(m) +
	       ") AS _g FROM query_table(source::VARCHAR) GROUP BY group_col)) "
	       "WHERE (_mv).value IS NOT NULL ORDER BY group_col, metric";
}

unique_ptr<CreateMacroInfo> CreateTableMacro(const string &name, const vector<string> &parameters,
                                             const vector<pair<string, string>> &named_parameters, const string &sql,
                                             const string &description, const string &example,
                                             const vector<string> &categories = {"regression", "table-macro"}) {
	auto parser = MakeParser();
	parser.ParseQuery(sql);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("Expected a single select statement for table macro %s", name);
	}
	auto function = make_uniq<TableMacroFunction>(std::move(parser.statements[0]->Cast<SelectStatement>().node));
	for (auto &p : parameters) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(AsDuckName(p)));
	}
	for (auto &np : named_parameters) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(AsDuckName(np.first)));
		auto defaults = parser.ParseExpressionList(np.second);
		function->default_parameters.insert(make_pair(AsDuckName(np.first), std::move(defaults[0])));
	}

	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	SetCreateInfoName(*info, DEFAULT_SCHEMA, name);
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(function));

	FunctionDescription desc;
	desc.description = description;
	desc.examples = {example};
	desc.categories = categories;
	desc.parameter_names = parameters;
	info->descriptions.push_back(std::move(desc));
	return info;
}

// Pairwise post-hoc tests (#159): <test>_by emits the `test` schema, one row
// per pair of factor levels within each group of source.
struct PosthocMacro {
	const char *test;
	const char *label;
};

const PosthocMacro POSTHOC_MACROS[] = {
    {"pairwise_t_test", "pairwise t-tests with p-value adjustment (R pairwise.t.test)"},
    {"tukey_hsd", "Tukey's HSD (R TukeyHSD(aov(...)))"},
    {"dunn_test", "Dunn's rank test with p-value adjustment (FSA::dunnTest)"},
};

string PosthocSql(const PosthocMacro &m) {
	return "SELECT group_col, model_id, (_r).group2 || '-' || (_r).group1 AS test_id, (_r).method AS method, "
	       "(_r).statistic AS statistic, (_r).p_value AS p_value, (_r).group1 AS group1, (_r).group2 AS group2, "
	       "(_r).p_adj AS p_adj, (_r).df1 AS df1, NULL::DOUBLE AS df2, (_r).estimate AS estimate, "
	       "(_r).conf_low AS conf_low, (_r).conf_high AS conf_high, (_r).conf_level AS conf_level, "
	       "NULL::DOUBLE AS effect_size, NULL::VARCHAR AS effect_name, (_r).alternative AS alternative, "
	       "(_r).p_adjust_method AS p_adjust_method "
	       "FROM (SELECT group_col, model_id, unnest(range(len(_l))) AS _i, unnest(_l) AS _r "
	       "FROM (SELECT group_col, group_col::VARCHAR AS model_id, " +
	       string(m.test) +
	       "_agg(value_col::DOUBLE, factor_col, options) AS _l FROM query_table(source::VARCHAR) GROUP BY group_col)) "
	       "ORDER BY group_col, _i";
}

// Coefficient paths, CV curves, nested CV and the quantile process per group
// (#154, #157, #204): the `terms` schema with index_name 'lambda' / 'tau' and the
// `curve` schema with curve_type 'lambda_cv' / 'nested_cv'.
struct PathMacro {
	const char *agg;
	const char *label;
	bool curve;
	//! The model, reported as model_id / model_name.
	const char *method;
	//! Nested CV (the `curve` schema with the nested fields).
	bool nested = false;
};

const PathMacro PATH_MACROS[] = {
    {"ridge_path", "the ridge coefficient path (glmnet alpha = 0)", false, "ridge"},
    {"elasticnet_path", "the elastic-net coefficient path (glmnet, l1_ratio = alpha)", false, "elasticnet"},
    {"lars_path", "the LARS / Lasso-LARS coefficient path", false, "lars"},
    {"quantile_process", "the regression-quantile process (one quantile regression per tau, with inference)", false, "quantile"},
    {"ridge_cv", "the k-fold cross-validation curve of the ridge path (cv.glmnet)", true, "ridge"},
    {"elasticnet_cv", "the k-fold cross-validation curve of the elastic-net path (cv.glmnet)", true, "elasticnet"},
    {"ridge_nested_cv", "the nested cross-validation of the ridge path", true, "ridge", true},
    {"elasticnet_nested_cv", "the nested cross-validation of the elastic-net path", true, "elasticnet", true},
};

string PathSql(const PathMacro &m) {
	// The CV aggregates take the cv_key (NULL by default: no key).
	string args = m.curve ? "y_col, x_cols, cv_key, options" : "y_col, x_cols, options";
	string inner = "FROM (SELECT group_col, id, model_id, model_name, unnest(range(len(_l))) AS _i, unnest(_l) AS _t "
	               "FROM (SELECT group_col, " +
	               ModelIds(m.method) + ", " + string(m.agg) + "_agg(" + args +
	               ") AS _l FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";
	if (m.nested) {
		return "SELECT group_col, id, model_id, model_name, (_t).curve_type AS curve_type, (_t).series AS series, "
		       "(_t).x AS x, (_t).y AS y, (_t).y_low AS y_low, (_t).y_high AS y_high, (_t).threshold AS threshold, "
		       "NULL::DOUBLE AS weight, NULL::BIGINT AS n_censor, (_t).fold AS fold, (_t).lambda AS lambda, "
		       "(_t).test_error AS test_error, (_t).test_se AS test_se, (_t).inner_cv_mean AS inner_cv_mean, "
		       "(_t).inner_cv_se AS inner_cv_se, (_t).n_train AS n_train, (_t).n_test AS n_test, "
		       "(_t).n_inner_folds AS n_inner_folds, (_t).n_folds AS n_folds, (_t).metric AS metric " +
		       inner;
	}
	if (m.curve) {
		return "SELECT group_col, id, model_id, model_name, (_t).curve_type AS curve_type, NULL::VARCHAR AS series, (_t).x AS x, "
		       "(_t).y AS y, (_t).y_low AS y_low, (_t).y_high AS y_high, (_t).threshold AS threshold, "
		       "NULL::DOUBLE AS weight, NULL::BIGINT AS n_censor, (_t).lambda AS lambda, (_t).cv_mean AS cv_mean, "
		       "(_t).cv_se AS cv_se, (_t).n_nonzero AS n_nonzero, (_t).is_min AS is_min, (_t).is_1se AS is_1se, "
		       "(_t).n_folds AS n_folds " +
		       inner;
	}
	// names := [...] relabels the default x1, x2, ... terms.
	return "SELECT group_col, id, model_id, model_name, CASE WHEN names IS NULL OR (_t).term = '(Intercept)' THEN (_t).term "
	       "ELSE COALESCE((names::VARCHAR[])[TRY_CAST(substr((_t).term, 2) AS BIGINT)], (_t).term) END AS term, "
	       "(_t).estimate AS estimate, (_t).std_error AS std_error, (_t).statistic AS statistic, "
	       "(_t).p_value AS p_value, (_t).conf_low AS conf_low, (_t).conf_high AS conf_high, "
	       "(_t).conf_level AS conf_level, (_t).index_name AS index_name, (_t).index_value AS index_value " +
	       inner;
}

// <model>_partial_residuals_by (#158): one row per input row and fitted term.
struct PartialMacro {
	const char *prefix;
	const char *label;
	bool weighted;
};

const PartialMacro PARTIAL_MACROS[] = {
    {"ols", "an OLS regression", false},          {"wls", "a weighted least squares regression", true},
    {"poisson", "a Poisson GLM", false},          {"binomial", "a binomial GLM", false},
    {"logistic", "a logistic regression", false}, {"negbinom", "a negative binomial GLM", false},
    {"gamma", "a Gamma GLM", false},              {"tweedie", "a Tweedie GLM", false},
};

string PartialSql(const PartialMacro &m) {
	const string frame = "OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED "
	                     "PRECEDING AND UNBOUNDED FOLLOWING)";
	return "SELECT group_col, id, model_id, model_name, row_id, (_t).term_index AS term_index, "
	       "COALESCE((names::VARCHAR[])[(_t).term_index], 'x' || (_t).term_index) AS term, (_t).x AS x, "
	       "(_t).partial_residual AS partial_residual, (_t).component AS component, "
	       "(_t).x_residual AS x_residual, (_t).y_residual AS y_residual "
	       "FROM (SELECT group_col, " +
	       ModelIds(m.prefix) + ", _anofox_rid AS row_id, unnest(_pr[_rn]) AS _t "
	       "FROM (SELECT group_col, _anofox_rid, ROW_NUMBER() " +
	       frame + " AS _rn, " + m.prefix + "_partial_residuals_agg(y_col, x_cols, " +
	       (m.weighted ? "weight_col, " : "") + "options) " + frame +
	       " AS _pr FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR)))) "
	       "ORDER BY group_col, term_index, row_id";
}

// Curve producers (#155, #156): one row per curve point in the integration
// contract's `curve` schema: model_id, series, curve_type, x, y, y_low, y_high,
// threshold, weight, n_censor (NULL where they do not apply).
string AcfCurveSql(const char *agg, const char *curve_type, const char *field) {
	return "SELECT group_col, group_col::VARCHAR AS model_id, NULL::VARCHAR AS series, '" + string(curve_type) +
	       "' AS curve_type, (_r).lag::DOUBLE AS x, (_r)." + field +
	       " AS y, (_r).conf_low AS y_low, (_r).conf_high AS y_high, NULL::DOUBLE AS threshold, "
	       "NULL::DOUBLE AS weight, NULL::BIGINT AS n_censor "
	       "FROM (SELECT group_col, unnest(" +
	       agg +
	       "(value_col::DOUBLE, max_lag ORDER BY order_col)) AS _r FROM query_table(source::VARCHAR) "
	       "GROUP BY group_col) ORDER BY group_col, x";
}

const char *const KAPLAN_MEIER_BY_SQL =
    "SELECT strata_col, strata_col::VARCHAR AS model_id, strata_col::VARCHAR AS series, 'km' AS curve_type, "
    "(_r).time AS x, (_r).surv AS y, (_r).conf_low AS y_low, (_r).conf_high AS y_high, NULL::DOUBLE AS threshold, "
    "(_r).n_risk::DOUBLE AS weight, (_r).n_censor AS n_censor, (_r).n_risk AS n_risk, (_r).n_event AS n_event, "
    "(_r).std_err AS std_err "
    "FROM (SELECT strata_col, unnest(kaplan_meier_agg(time_col::DOUBLE, event_col, conf_level, conf_type)) AS _r "
    "FROM query_table(source::VARCHAR) GROUP BY strata_col) ORDER BY strata_col, x";

// Classifier evaluation curves (#153): <curve>_curve_by emits the `curve`
// schema, one row per curve point within each group of source, in curve order.
struct ClassifierCurveMacro {
	const char *curve;
	//! Aggregate call producing the LIST of points.
	const char *agg_call;
	//! Named parameters with defaults.
	vector<pair<string, string>> named;
	//! Contract columns x, y, y_low, y_high, threshold, weight from point _p.
	const char *columns;
	const char *label;
	const char *example;
};

string ClassifierCurveSql(const ClassifierCurveMacro &m) {
	return string("SELECT group_col, model_id, NULL::VARCHAR AS series, '") + m.curve +
	       "'::VARCHAR AS curve_type, " + m.columns +
	       ", NULL::BIGINT AS n_censor "
	       "FROM (SELECT group_col, model_id, unnest(range(len(_l))) AS _i, unnest(_l) AS _p "
	       "FROM (SELECT group_col, group_col::VARCHAR AS model_id, " +
	       m.agg_call + " AS _l FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";
}

} // namespace

// lowess_by (#63): the LOWESS curve of y on x per group in the contract's
// `curve` schema, curve_type 'lowess'. model_id / model_name are the method and
// id the group, as the other model-fitting macros.
const char *const LOWESS_BY_SQL =
    "SELECT group_col, group_col::VARCHAR AS id, 'lowess' AS model_id, 'lowess' AS model_name, "
    "'lowess' AS curve_type, NULL::VARCHAR AS series, (_p).x AS x, (_p).yhat AS y, NULL::DOUBLE AS y_low, "
    "NULL::DOUBLE AS y_high, NULL::DOUBLE AS threshold, NULL::DOUBLE AS weight, NULL::BIGINT AS n_censor "
    "FROM (SELECT group_col, unnest(range(len(_l))) AS _i, unnest(_l) AS _p "
    "FROM (SELECT group_col, lowess_agg(y_col::DOUBLE, x_col::DOUBLE, frac::DOUBLE, iterations::INTEGER) AS _l "
    "FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";

void RegisterCurveMacros(ExtensionLoader &loader) {
	struct AcfMacro {
		const char *name;
		const char *agg;
		const char *curve_type;
		const char *label;
	};
	const AcfMacro acf_macros[] = {{"acf_by", "acf_agg", "acf", "sample autocorrelation function (R acf)"},
	                               {"pacf_by", "pacf_agg", "pacf", "sample partial autocorrelation function (R pacf)"}};
	for (auto &m : acf_macros) {
		auto macro = CreateTableMacro(
		    m.name, {"source", "group_col", "value_col", "order_col"}, {{"max_lag", "NULL"}},
		    AcfCurveSql(m.agg, m.curve_type, m.curve_type),
		    "Table macro: the " + string(m.label) + " of value_col ordered by order_col within each group of source, "
		    "one row per lag (the integration contract's `curve` schema, curve_type '" + m.curve_type +
		        "'): the group column, model_id, series, curve_type, x (lag), y, y_low / y_high (95% white-noise "
		        "band), threshold, weight, n_censor. max_lag defaults to R's floor(10 log10 n).",
		    string(m.name) + "('residuals', store, resid, t, max_lag := 12)", {"time-series", "diagnostics", "table-macro"});
		loader.RegisterFunction(*macro);
	}

	auto km = CreateTableMacro(
	    "kaplan_meier_by", {"source", "strata_col", "time_col", "event_col"},
	    {{"conf_level", "0.95"}, {"conf_type", "'log'"}}, KAPLAN_MEIER_BY_SQL,
	    "Table macro: one Kaplan-Meier curve per stratum of source (R survfit(Surv(time, event) ~ strata)), one row "
	    "per distinct time (the integration contract's `curve` schema, curve_type 'km', series = stratum): the "
	    "stratum column, model_id, series, curve_type, x (time), y (survival), y_low / y_high (pointwise interval), "
	    "threshold, weight (number at risk), n_censor, then n_risk, n_event and std_err.",
	    "kaplan_meier_by('patients', treatment, duration, event, conf_type := 'log-log')",
	    {"survival", "table-macro"});
	loader.RegisterFunction(*km);

	auto lowess = CreateTableMacro(
	    "lowess_by", {"source", "group_col", "x_col", "y_col"}, {{"frac", "2 / 3"}, {"iterations", "3"}},
	    LOWESS_BY_SQL,
	    "Table macro: the LOWESS smoother of y_col on x_col per group of source (R stats::lowess(x, y, f = frac, "
	    "iter = iterations), see lowess_agg), one row per point in x order (the integration contract's `curve` "
	    "schema, curve_type 'lowess'): the group column, id (the group), model_id and model_name ('lowess'), "
	    "curve_type, series, x, y (the smoothed value), y_low, y_high, threshold, weight, n_censor.",
	    "lowess_by('measurements', sensor, temperature, load, frac := 0.3)", {"regression", "smoothing", "table-macro"});
	loader.RegisterFunction(*lowess);
}

void RegisterClassifierCurveMacros(ExtensionLoader &loader) {
	const ClassifierCurveMacro macros[] = {
	    {"roc", "roc_agg(y_true_col, score_col, drop_intermediate)", {{"drop_intermediate", "true"}},
	     "(_p).fpr AS x, (_p).tpr AS y, NULL::DOUBLE AS y_low, NULL::DOUBLE AS y_high, (_p).threshold AS threshold, "
	     "NULL::DOUBLE AS weight",
	     "the ROC curve (x = false positive rate, y = true positive rate; scikit-learn roc_curve, see roc_agg)",
	     "roc_curve_by('scored', model, label, score)"},
	    {"pr", "pr_agg(y_true_col, score_col)", {},
	     "(_p).recall AS x, (_p).precision AS y, NULL::DOUBLE AS y_low, NULL::DOUBLE AS y_high, "
	     "(_p).threshold AS threshold, NULL::DOUBLE AS weight",
	     "the precision-recall curve (x = recall, y = precision; scikit-learn precision_recall_curve, see pr_agg)",
	     "pr_curve_by('scored', model, label, score)"},
	    {"calibration", "calibration_agg(y_true_col, prob_col, n_bins, conf_level)",
	     {{"n_bins", "10"}, {"conf_level", "0.95"}},
	     "(_p).mean_pred AS x, (_p).frac_pos AS y, (_p).conf_low AS y_low, (_p).conf_high AS y_high, "
	     "NULL::DOUBLE AS threshold, (_p).n::DOUBLE AS weight",
	     "the calibration curve (x = mean predicted probability, y = fraction of positives, y_low/y_high = Wilson "
	     "interval, weight = bin count; see calibration_agg)",
	     "calibration_curve_by('scored', model, label, prob, n_bins := 10)"},
	};
	for (auto &m : macros) {
		string name = string(m.curve) + "_curve_by";
		string value_col = string(m.curve) == "calibration" ? "prob_col" : "score_col";
		auto macro = CreateTableMacro(
		    name, {"source", "group_col", "y_true_col", value_col}, m.named, ClassifierCurveSql(m),
		    "Table macro: computes " + string(m.label) +
		        " per group of source and returns one row per point in curve order (the integration contract's "
		        "`curve` schema): the group column, model_id, series, curve_type ('" +
		        m.curve +
		        "'), x, y, y_low, y_high, threshold, weight, n_censor. y_true_col is BOOLEAN or 0/1.",
		    m.example, {"classification", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

void RegisterPosthocMacros(ExtensionLoader &loader) {
	for (auto &m : POSTHOC_MACROS) {
		string name = string(m.test) + "_by";
		auto macro = CreateTableMacro(
		    name, {"source", "group_col", "value_col", "factor_col"}, {{"options", "NULL"}}, PosthocSql(m),
		    "Table macro: runs " + string(m.label) +
		        " between the levels of factor_col within each group of source and returns one row per pair "
		        "(the integration contract's `test` schema): the group column, model_id, test_id ('<group2>-<group1>'), "
		        "method, statistic, p_value, group1, group2, p_adj, df1, df2, estimate, conf_low, conf_high, "
		        "conf_level, effect_size, effect_name, alternative, p_adjust_method. options are passed to " +
		        m.test + "_agg.",
		    name + "('my_table', experiment, response, treatment)", {"hypothesis-testing", "post-hoc", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

// GLMM contract macros (#218): one mixed model per group_col (the outer group),
// with re_group the random-effects grouping factor. model_id = model_name =
// 'glmm' and id = the outer group, as the other model-fitting macros (#187).
//
// The fit with inference on, its option-derived labels, and the label of every
// random-effect component: '(Intercept)', then each random slope named like its
// fixed effect (names := [...] labels the design columns; the 'random' / 'offset'
// / 'groups' indices count x columns, so the offset and grouping columns are
// skipped when mapping one to the other).
const char *const GLMM_FITS_SQL =
    "WITH _fits AS (SELECT group_col, group_col::VARCHAR AS id, 'glmm' AS model_id, 'glmm' AS model_name, "
    "glmm_fit_agg(y_col, x_cols, re_group, map_concat(MAP {'compute_inference': 'true'}, "
    "coalesce(options::MAP(VARCHAR, VARCHAR), MAP {}::MAP(VARCHAR, VARCHAR)))) AS _m "
    "FROM query_table(source::VARCHAR) GROUP BY group_col), "
    "_cfg AS (SELECT coalesce(TRY_CAST(coalesce(map_extract(_o, 'random')[1], map_extract(_o, 'random_slopes')[1]) "
    "AS INTEGER[]), []::INTEGER[]) AS _rs, "
    "coalesce(TRY_CAST(coalesce(map_extract(_o, 'groups')[1], map_extract(_o, 'crossed')[1]) AS INTEGER[]), "
    "[]::INTEGER[]) AS _gc, TRY_CAST(map_extract(_o, 'offset')[1] AS INTEGER) AS _off "
    "FROM (SELECT coalesce(options::MAP(VARCHAR, VARCHAR), MAP {}::MAP(VARCHAR, VARCHAR)) AS _o)), "
    "_f AS (SELECT _fits.*, _gc, coalesce((re_names::VARCHAR[])[1], 'group') AS _re_label, "
    "list_concat(['(Intercept)'], list_transform(_rs, lambda _k: coalesce((names::VARCHAR[])[_k - "
    "len(list_filter(list_concat(_gc, [_off]), lambda _d: _d < _k))], 'x' || (_k - "
    "len(list_filter(list_concat(_gc, [_off]), lambda _d: _d < _k)))))) AS _re_terms "
    "FROM _fits, _cfg WHERE _m IS NOT NULL) ";

// The `terms` columns, then effect / re_group / re_level / df (broom.mixed::tidy).
const char *const GLMM_TERMS_COLUMNS =
    "group_col, id, model_id, model_name, term, estimate, std_error, statistic, p_value, conf_low, conf_high, "
    "conf_level, index_name, index_value, effect, _re_grp AS re_group, re_level, df";

static string GlmmTidySql() {
	const string no_inference = "NULL::DOUBLE AS std_error, NULL::DOUBLE AS statistic, NULL::DOUBLE AS p_value, "
	                            "NULL::DOUBLE AS conf_low, NULL::DOUBLE AS conf_high, NULL::DOUBLE AS conf_level, ";
	const string no_index = "NULL::VARCHAR AS index_name, NULL::DOUBLE AS index_value, ";
	// Fixed effects: tidy() of the fit, with the Satterthwaite df alongside.
	const string fixed =
	    "SELECT group_col, id, model_id, model_name, (_t).term AS term, (_t).estimate AS estimate, "
	    "(_t).std_error AS std_error, (_t).statistic AS statistic, (_t).p_value AS p_value, (_t).conf_low AS conf_low, "
	    "(_t).conf_high AS conf_high, (_t).conf_level AS conf_level, (_t).index_name AS index_name, "
	    "(_t).index_value AS index_value, 'fixed' AS effect, NULL::VARCHAR AS _re_grp, NULL::VARCHAR AS re_level, "
	    "_df AS df, 1 AS _sec, _i AS _pos "
	    "FROM (SELECT *, unnest(tidy(_m, names)) AS _t, unnest(range(len(tidy(_m, names)))) AS _i, "
	    "unnest(list_concat(CASE WHEN (_m).intercept IS NULL THEN []::DOUBLE[] ELSE [(_m).intercept_df] END, "
	    "coalesce((_m).df, []::DOUBLE[]))) AS _df FROM _f)";
	// Variance components of re_group: the SD of every random effect and their
	// correlations, from the covariance Sigma (a crossed fit without random slopes
	// has no Sigma; its first factor's variance is factors[1]).
	const string ran_pars =
	    "SELECT group_col, id, model_id, model_name, CASE WHEN _a = _b THEN 'sd__' || _re_terms[_a + 1] "
	    "ELSE 'cor__' || _re_terms[_a + 1] || '.' || _re_terms[_b + 1] END AS term, "
	    "CASE WHEN _a = _b THEN sqrt(_cov[_a * _q + _a + 1]) "
	    "ELSE _cov[_a * _q + _b + 1] / sqrt(_cov[_a * _q + _a + 1] * _cov[_b * _q + _b + 1]) END AS estimate, " +
	    no_inference + no_index +
	    "'ran_pars' AS effect, _re_label AS _re_grp, NULL::VARCHAR AS re_level, NULL::DOUBLE AS df, 2 AS _sec, "
	    "CASE WHEN _a = _b THEN _a ELSE _q + _a * _q + _b END AS _pos "
	    "FROM (SELECT *, unnest(range(_a, _q)) AS _b FROM (SELECT *, unnest(range(_q)) AS _a FROM (SELECT *, "
	    "CASE WHEN (_m).random_dim > 0 THEN (_m).random_cov ELSE [((_m).factors[1]).var] END AS _cov, "
	    "greatest((_m).random_dim, 1) AS _q FROM _f)))";
	// The random-intercept SD of every further crossed factor ('groups'), labelled by
	// re_names[2..] or its x column.
	const string crossed =
	    "SELECT group_col, id, model_id, model_name, 'sd__(Intercept)' AS term, "
	    "sqrt(((_m).factors[_fi]).var) AS estimate, " +
	    no_inference + no_index +
	    "'ran_pars' AS effect, coalesce((re_names::VARCHAR[])[_fi], 'x' || _gc[_fi - 1]) AS _re_grp, "
	    "NULL::VARCHAR AS re_level, NULL::DOUBLE AS df, 2 AS _sec, 1000000 + _fi AS _pos "
	    "FROM (SELECT *, unnest(range(2, len((_m).factors) + 1)) AS _fi FROM _f)";
	// The residual SD, for the families with a free dispersion.
	const string residual =
	    "SELECT group_col, id, model_id, model_name, 'sd__Observation' AS term, sqrt((_m).var_residual) AS estimate, " +
	    no_inference + no_index +
	    "'ran_pars' AS effect, 'Residual' AS _re_grp, NULL::VARCHAR AS re_level, NULL::DOUBLE AS df, 2 AS _sec, "
	    "2000000 AS _pos FROM _f WHERE (_m).family NOT IN ('poisson', 'binomial', 'negative_binomial')";
	// The BLUP of every level of re_group and random effect, with its conditional
	// standard error and normal interval.
	const string ran_vals =
	    "SELECT group_col, id, model_id, model_name, _re_terms[_a + 1] AS term, (_r).effects[_a + 1] AS estimate, "
	    "(_r).effects_se[_a + 1] AS std_error, NULL::DOUBLE AS statistic, NULL::DOUBLE AS p_value, "
	    "(_r).effects_conf_low[_a + 1] AS conf_low, (_r).effects_conf_high[_a + 1] AS conf_high, "
	    "(_m).conf_level AS conf_level, " +
	    no_index +
	    "'ran_vals' AS effect, _re_label AS _re_grp, (_r).group AS re_level, NULL::DOUBLE AS df, 3 AS _sec, "
	    "_a AS _pos FROM (SELECT *, unnest(range(len((_r).effects))) AS _a FROM (SELECT *, unnest((_m).ranef) AS _r "
	    "FROM _f))";
	return string(GLMM_FITS_SQL) + "SELECT " + GLMM_TERMS_COLUMNS + " FROM (" + fixed + " UNION ALL " + ran_pars +
	       " UNION ALL " + crossed + " UNION ALL " + residual + " UNION ALL " + ran_vals +
	       ") ORDER BY group_col, _sec, re_level NULLS FIRST, _pos";
}

// The `summary` metrics of glmm_glance_by, in order (broom.mixed::glance plus the
// variance components).
const char *const GLMM_GLANCE_METRICS[] = {"n_observations", "n_groups", "n_features", "log_likelihood", "aic",
                                           "bic", "deviance", "df_residual", "var_group", "var_residual", "icc",
                                           "nb_theta", "iterations"};

static string GlmmGlanceSql() {
	string metrics;
	for (auto metric : GLMM_GLANCE_METRICS) {
		metrics += "{'metric': '" + string(metric) + "', 'value': (_m)." + metric + "::DOUBLE}, ";
	}
	metrics += "{'metric': 'sigma', 'value': sqrt((_m).var_residual)}, "
	           "{'metric': 'converged', 'value': (_m).converged::DOUBLE}";
	return "SELECT group_col, id, model_id, model_name, model_type, (_mv).metric AS metric, (_mv).value AS value "
	       "FROM (SELECT group_col, group_col::VARCHAR AS id, 'glmm' AS model_id, 'glmm' AS model_name, "
	       "(_m).model_type AS model_type, unnest([" +
	       metrics +
	       "]) AS _mv FROM (SELECT group_col, glmm_fit_agg(y_col, x_cols, re_group, options) AS _m "
	       "FROM query_table(source::VARCHAR) GROUP BY group_col) WHERE _m IS NOT NULL) "
	       "WHERE (_mv).value IS NOT NULL ORDER BY group_col, metric";
}

// The `obs` columns of <model>_augment_by, then fitted_population and
// pearson_residual. No id column: the source columns are all kept (#187).
const char *const GLMM_AUGMENT_SQL =
    "SELECT * EXCLUDE (_aug, _rn, _anofox_rid), 'glmm' AS model_id, 'glmm' AS model_name, _anofox_rid AS row_id, "
    "(_aug[_rn]).fitted AS fitted, (_aug[_rn]).residual AS residual, 'response' AS resid_type, "
    "NULL::DOUBLE AS std_residual, NULL::DOUBLE AS stud_residual, NULL::DOUBLE AS leverage, NULL::DOUBLE AS cooks_d, "
    "NULL::DOUBLE AS dffits, 1.0::DOUBLE AS weight, (_aug[_rn]).is_training AS is_training, NULL::INTEGER AS n_params, "
    "(_aug[_rn]).fitted_population AS fitted_population, (_aug[_rn]).pearson_residual AS pearson_residual "
    "FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED "
    "PRECEDING AND UNBOUNDED FOLLOWING) AS _rn, glmm_augment_agg(y_col, x_cols, re_group, options) OVER (PARTITION "
    "BY group_col ORDER BY order_by, _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS _aug "
    "FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))) "
    "ORDER BY group_col, row_id";

static void RegisterGlmmOutputMacros(ExtensionLoader &loader) {
	const vector<string> params = {"source", "group_col", "y_col", "x_cols", "re_group"};
	const vector<string> categories = {"mixed-models", "regression", "table-macro"};
	auto tidy = CreateTableMacro(
	    "glmm_tidy_by", params, {{"options", "NULL"}, {"names", "NULL"}, {"re_names", "NULL"}}, GlmmTidySql(),
	    "Table macro: fits one generalized linear mixed model (glmm_fit_agg, random effects over re_group) per "
	    "group_col and returns its terms (the integration contract's `terms` schema, broom.mixed::tidy): the group "
	    "column, id (the group), model_id and model_name ('glmm'), term, estimate, std_error, statistic, p_value, "
	    "conf_low, conf_high, conf_level, index_name, index_value, then effect ('fixed', 'ran_pars', 'ran_vals'), "
	    "re_group, re_level and df. Fixed effects carry Wald z or, for gaussian, Satterthwaite t inference (option "
	    "ddf; inference is always computed); ran_pars rows are the random-effect SDs (sd__<term>), correlations "
	    "(cor__<term>.<term>) and the residual SD; ran_vals rows are the BLUPs with their conditional standard "
	    "errors and normal intervals. names := [...] labels the fixed-effect columns, re_names := [...] the random "
	    "effects' grouping factors (default 'group').",
	    "glmm_tidy_by('sales', region, revenue, [price], store, names := ['price'])", categories);
	loader.RegisterFunction(*tidy);

	auto glance = CreateTableMacro(
	    "glmm_glance_by", params, {{"options", "NULL"}}, GlmmGlanceSql(),
	    "Table macro: fits one generalized linear mixed model (glmm_fit_agg, random effects over re_group) per "
	    "group_col and returns its fit statistics in long form (the integration contract's `summary` schema, "
	    "broom.mixed::glance): the group column, id (the group), model_id and model_name ('glmm'), model_type, "
	    "metric, value. Metrics: n_observations, n_groups, n_features, sigma, log_likelihood, aic, bic, deviance (the "
	    "REML criterion for a REML fit), df_residual, var_group, var_residual, icc, nb_theta (negative binomial), "
	    "converged (1/0), iterations.",
	    "glmm_glance_by('sales', region, revenue, [price], store)", categories);
	loader.RegisterFunction(*glance);

	auto augment = CreateTableMacro(
	    "glmm_augment_by", params, {{"options", "NULL"}, {"order_by", "NULL"}}, GLMM_AUGMENT_SQL,
	    "Table macro: fits one generalized linear mixed model (glmm_augment_agg, random effects over re_group) per "
	    "group_col and returns every source row with its per-observation results (the integration contract's `obs` "
	    "schema, broom.mixed::augment): all source columns, model_id and model_name ('glmm'), row_id, fitted "
	    "(conditional on the BLUPs, response scale), residual (y - fitted), resid_type ('response'), std_residual, "
	    "stud_residual, leverage, cooks_d, dffits (NULL), weight, is_training, n_params (NULL), then "
	    "fitted_population (random effects at zero) and pearson_residual. Rows the fit could not use get NULLs.",
	    "glmm_augment_by('sales', region, revenue, [price], store)", categories);
	loader.RegisterFunction(*augment);
}

void RegisterModelOutputMacros(ExtensionLoader &loader) {
	for (auto &m : MODEL_OUTPUT_MODELS) {
		vector<string> params = {"source", "group_col", "y_col", "x_cols"};
		if (m.ordered) {
			params.insert(params.begin() + 2, "order_col");
		}
		if (m.extra_col) {
			params.push_back(m.extra_col);
		}
		string args = string(m.ordered ? "'my_table', group_col, t, y, [x1, x2]" : "'my_table', group_col, y, [x1, x2]") +
		              (!m.extra_col ? "" : string(m.extra_col) == "weight_col" ? ", w" : ", event");
		string prefix = m.prefix;
		const string order_note = m.ordered ? " Rows are fitted in order_col order (the model is sequential). " : " ";

		auto tidy = CreateTableMacro(
		    prefix + "_tidy_by", params,
		    {{"options", "NULL"}, {"names", "NULL"}}, m.dynamic_terms ? DynamicTidySql(m) : TidySql(m),
		    m.dynamic_terms
		        ? "Table macro: fits " + string(m.label) +
		              " per group of source (rows in order_col order) and returns the time-varying coefficients of "
		              "every fitted row as `terms` rows, intercept first (the shape of <model>_rolling_tidy_by): the "
		              "group column, id (the group), model_id and model_name (the method), window_end (the row's "
		              "order_col value), term, estimate, std_error, statistic, p_value, conf_low, conf_high, "
		              "conf_level (NULL: no inference), index_name ('window_end'), index_value (the row's 1-based "
		              "position among the fitted rows). names := [...] labels the terms."
		        : "Table macro: fits " + string(m.label) +
		              " per group of source and returns one row per term (intercept first): the group column, id (the group), model_id and model_name (the method), "
		              "term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, index_name, "
		              "index_value." + order_note +
		              (m.inference ? "Inference is computed by default (compute_inference is merged under options" +
		                                 (m.tidy_defaults ? ", with " + string(m.tidy_defaults) : string()) + "). "
		               : m.credible ? "std_error is the posterior standard deviation and conf_low / conf_high the "
		                              "central Bayesian credible interval at conf_level (options' confidence_level), "
		                              "not a frequentist confidence interval; statistic and p_value are NULL. "
		                            : "The model has no coefficient inference. ") +
		              "names := [...] labels the terms.",
		    prefix + "_tidy_by(" + args + ", names := ['x1', 'x2'])");
		loader.RegisterFunction(*tidy);

		auto glance = CreateTableMacro(
		    prefix + "_glance_by", params, {{"options", "NULL"}}, GlanceSql(m),
		    "Table macro: fits " + string(m.label) +
		        " per group of source and returns its fit statistics in long form, one row per available metric: "
		        "the group column, id (the group), model_id and model_name (the method), model_type, metric, value." +
		        (m.ordered ? " Rows are fitted in order_col order (the model is sequential)." : ""),
		    prefix + "_glance_by(" + args + ")");
		loader.RegisterFunction(*glance);

		bool rolling = false;
		for (auto r : ROLLING_MODELS) {
			rolling = rolling || prefix == r;
		}
		if (rolling) {
			vector<string> rparams = {"source", "group_col", "order_col", "y_col", "x_cols"};
			if (m.extra_col) {
				rparams.push_back(m.extra_col);
			}
			auto roll = CreateTableMacro(
			    prefix + "_rolling_tidy_by", rparams,
			    {{"window_size", "30"},
			     {"min_obs", "NULL"},
			     {"options", "NULL"},
			     {"names", "NULL"}},
			    RollingTidySql(m),
			    "Table macro: fits " + string(m.label) +
			        " over a trailing window of window_size rows (ordered by order_col) within each group of source "
			        "and returns the coefficients of every window as `terms` rows (integration contract): the group "
			        "column, id (the group), model_id and model_name (the method), window_end (order_col of the window's last row), term, estimate, std_error, "
			        "statistic, p_value, conf_low, conf_high, conf_level, index_name ('window_end'), index_value (the "
			        "window's 1-based end position in its group) and n_window. Windows with fewer than min_obs rows "
			        "(default window_size) are skipped.",
			    prefix + "_rolling_tidy_by('my_table', group_col, t, y, [x1, x2]" +
			        (!m.extra_col ? "" : string(m.extra_col) == "weight_col" ? ", w" : ", event") +
			        ", window_size := 24)");
			loader.RegisterFunction(*roll);
		}
	}
	RegisterGlmmOutputMacros(loader);
}

void RegisterPathMacros(ExtensionLoader &loader) {
	for (auto &m : PATH_MACROS) {
		string name = string(m.agg) + "_by";
		string description =
		    "Table macro: computes " + string(m.label) + " per group of source with " + m.agg +
		    "_agg (same options) and returns " +
		    (m.nested ? string("one row per outer fold plus a summary row in the integration contract's `curve` "
		                       "schema: the group column, id (the group), model_id and model_name (the method), "
		                       "curve_type ('nested_cv'), series ('fold' | 'mean'), x (fold), y (test error), y_low, "
		                       "y_high (mean -/+ se on the summary row), threshold, weight, n_censor, fold, lambda, "
		                       "test_error, test_se, inner_cv_mean, inner_cv_se, n_train, n_test, n_inner_folds, "
		                       "n_folds, metric. cv_key := column (or a STRUCT of columns) passes the cv_key.")
		     : m.curve ? string("one row per lambda in the integration contract's `curve` schema: the group column, "
		                      "id (the group), model_id and model_name (the method), curve_type ('lambda_cv'), series, x (lambda), y (cv_mean), y_low, y_high "
		                      "(cv_mean -/+ cv_se), threshold, weight, n_censor, lambda, cv_mean, cv_se, n_nonzero, "
		                      "is_min, is_1se, n_folds. cv_key := column (or a STRUCT of columns) passes the cv_key "
		                      "of the cv_strategy option.")
		             : string("one row per index value and term in the `terms` schema: the group column, id (the group), model_id and model_name (the method), "
		                      "term, estimate, std_error, statistic, p_value, conf_low, conf_high, conf_level, "
		                      "index_name ('") +
		                   (string(m.agg) == "quantile_process" ? "tau" : string(m.agg) == "lars_path" ? "alpha" : "lambda") +
		                   "'), index_value. names := [...] labels the terms.");
		vector<pair<string, string>> named = {{"options", "NULL"}};
		if (!m.curve) {
			named.push_back({"names", "NULL"});
		} else {
			named.push_back({"cv_key", "NULL"});
		}
		auto macro = CreateTableMacro(name, {"source", "group_col", "y_col", "x_cols"}, named, PathSql(m),
		                              description, name + "('my_table', group_col, y, [x1, x2])");
		loader.RegisterFunction(*macro);
	}
	for (auto &m : PARTIAL_MACROS) {
		string name = string(m.prefix) + "_partial_residuals_by";
		vector<string> params = {"source", "group_col", "y_col", "x_cols"};
		if (m.weighted) {
			params.push_back("weight_col");
		}
		bool linear = string(m.prefix) == "ols" || m.weighted;
		auto macro = CreateTableMacro(
		    name, params, {{"options", "NULL"}, {"names", "NULL"}, {"order_by", "NULL"}}, PartialSql(m),
		    "Table macro: fits " + string(m.label) +
		        " per group of source and returns one row per input row and fitted term with partial-residual "
		        "(component-plus-residual, car::crPlots) and added-variable (car::avPlots) coordinates: the group "
		        "column, id (the group), model_id and model_name (the method), row_id, term_index, term, x, partial_residual (= " +
		        (linear ? "residual" : "working residual") +
		        " + component), component (b_j (x_j - mean(x_j))), x_residual, y_residual (" +
		        (linear ? "x_j and y residualised on the other columns" : "NULL for GLMs") +
		        "). Rows not used for fitting get NULL residuals. names := [...] labels the terms.",
		    name + "('my_table', group_col, y, [x1, x2]" + (m.weighted ? ", w" : "") + ")");
		loader.RegisterFunction(*macro);
	}
}

unique_ptr<CreateMacroInfo> CreateContractTableMacro(const string &name, const vector<string> &parameters,
                                                     const vector<pair<string, string>> &named_parameters,
                                                     const string &sql, const string &description,
                                                     const string &example, const vector<string> &categories) {
	return CreateTableMacro(name, parameters, named_parameters, sql, description, example, categories);
}

} // namespace duckdb
