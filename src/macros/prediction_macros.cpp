// The integration contract's `prediction` schema (#183):
//
//   <model>_prediction_by(source, group_col, x_col, y_col, x_cols [, weight_col],
//                         options := NULL, level := 0.95, split := NULL)
//       fitted values with their interval band per group, one row per input row,
//       via <model>_fit_predict_by; x_col is the plot's x position (also the
//       row order for order-sensitive models).
//   <model>_effects_by(source, group_col, y_col, x_cols [, weight_col],
//                      focal := 1, n_points := 50, at_values := NULL, options := NULL, level := 0.95)
//       a marginal-effect grid: the fitted mean over the range of feature `focal`
//       with the other features at their group means (or at_values), with a
//       confidence band from predict(model, x, {'interval': 'confidence'}).
//
// Columns: the group column, id, model_id, model_name, x, y, yhat, yhat_lower,
// yhat_upper, level, split ('train' | 'test' | 'future'; NULL on effect grids),
// as anofox-forecast's forecasts, so one fan/band plot serves both.

#include "anofox_statistics_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

namespace duckdb {

namespace {

struct PredictionModel {
	const char *prefix;
	const char *label;
	//! Extra column after x_cols (WLS weights), or nullptr.
	const char *extra_col;
	//! <prefix>_fit_predict_by takes order_by.
	bool order_by;
	//! predict() supports standard errors (OLS, WLS, Ridge, GLMs): <prefix>_effects_by exists.
	bool effects;
	//! <prefix>_fit_predict_by returns an interval band (PLS, quantile and isotonic do not).
	bool band = true;
	//! The model takes one feature, x_col itself (isotonic), instead of x_cols.
	bool single_x = false;
};

// clang-format off
const PredictionModel PREDICTION_MODELS[] = {
    {"ols", "an OLS regression", nullptr, true, true},
    {"wls", "a weighted least squares regression", "weight_col", true, true},
    {"ridge", "a Ridge regression", nullptr, true, true},
    {"elasticnet", "an Elastic Net regression", nullptr, true, false},
    {"huber", "a Huber M-estimator regression", nullptr, true, false},
    {"ransac", "a RANSAC regression", nullptr, true, false},
    {"theil_sen", "a Theil-Sen regression", nullptr, true, false},
    {"rls", "a recursive least squares regression", nullptr, true, false},
    {"bls", "a bounded least squares regression", nullptr, true, false},
    {"pls", "a partial least squares regression", nullptr, true, false, false},
    {"quantile", "a quantile regression", nullptr, true, false, false},
    {"alm", "an augmented linear model (ALM)", nullptr, true, false},
    {"poisson", "a Poisson GLM", nullptr, true, true},
    {"binomial", "a binomial GLM", nullptr, false, true},
    {"logistic", "a logistic regression", nullptr, false, true},
    {"negbinom", "a negative binomial GLM", nullptr, false, true},
    {"gamma", "a Gamma GLM", nullptr, false, true},
    {"tweedie", "a Tweedie GLM", nullptr, false, true},
    {"isotonic", "an isotonic regression", nullptr, true, false, false, true},
};
// clang-format on

//! options with confidence_level := level (level wins), as a MAP the option parser accepts.
const char *const MERGED_LEVEL_OPTIONS =
    "map_concat(coalesce(options::MAP(VARCHAR, VARCHAR), MAP {}::MAP(VARCHAR, VARCHAR)), "
    "MAP {'confidence_level': level::VARCHAR})";
const char *const MERGED_INFERENCE_OPTIONS =
    "map_concat(coalesce(options::MAP(VARCHAR, VARCHAR), MAP {}::MAP(VARCHAR, VARCHAR)), "
    "MAP {'compute_inference': 'true'})";

string ExtraArg(const PredictionModel &m) {
	return m.extra_col ? string(", ") + m.extra_col : string();
}

string PredictionSql(const PredictionModel &m) {
	const string band = m.band ? "yhat_lower, yhat_upper, CASE WHEN yhat_lower IS NULL THEN NULL ELSE level::DOUBLE END AS level"
	                           : "NULL::DOUBLE AS yhat_lower, NULL::DOUBLE AS yhat_upper, NULL::DOUBLE AS level";
	return string("SELECT group_col, group_col::VARCHAR AS id, '") + m.prefix + "' AS model_id, '" + m.prefix +
	       "' AS model_name, x_col AS x, y_col::DOUBLE AS y, yhat, " + band +
	       ", CASE WHEN is_training THEN 'train' WHEN y_col IS NULL THEN 'future' ELSE 'test' END AS split "
	       "FROM " +
	       m.prefix + "_fit_predict_by(source, group_col, y_col, " + (m.single_x ? "x_col" : "x_cols") + ExtraArg(m) +
	       ", options := " + (m.band ? MERGED_LEVEL_OPTIONS : "options") + ", split := split" +
	       (m.order_by ? ", order_by := x_col" : "") + ") ORDER BY group_col, x";
}

string EffectsSql(const PredictionModel &m) {
	return string("SELECT group_col, group_col::VARCHAR AS id, '") + m.prefix + "' AS model_id, '" + m.prefix +
	       "' AS model_name, focal::BIGINT AS focal, _xv AS x, NULL::DOUBLE AS y, (_p).yhat AS yhat, "
	       "(_p).yhat_lower AS yhat_lower, (_p).yhat_upper AS yhat_upper, level::DOUBLE AS level, "
	       "NULL::VARCHAR AS split, (_p).se_fit AS se_fit "
	       "FROM (SELECT group_col, _xv, predict(_m, list_transform(range(1, len(_base) + 1), "
	       "lambda _j: CASE WHEN _j = focal THEN _xv ELSE _base[_j] END), {'interval': 'confidence', 'level': level}) AS _p "
	       "FROM (SELECT group_col, _m, _base, _lo + _k * (_hi - _lo) / greatest(n_points - 1, 1) AS _xv "
	       "FROM (SELECT group_col, _m, "
	       "coalesce(at_values, list_transform(range(1, len(_all[1]) + 1), lambda _j: list_avg(list_transform(_all, lambda _r: "
	       "_r[_j])))) AS _base, list_min(list_transform(_all, lambda _r: _r[focal])) AS _lo, "
	       "list_max(list_transform(_all, lambda _r: _r[focal])) AS _hi "
	       "FROM (SELECT group_col, " +
	       m.prefix + "_fit_agg(y_col, x_cols" + ExtraArg(m) + ", " + MERGED_INFERENCE_OPTIONS +
	       ") AS _m, list(x_cols) FILTER (WHERE y_col IS NOT NULL) AS _all FROM query_table(source::VARCHAR) GROUP BY group_col)), "
	       "range(n_points) _grid(_k))) "
	       "ORDER BY group_col, x";
}

} // namespace

void RegisterPredictionMacros(ExtensionLoader &loader) {
	const vector<string> categories = {"regression", "prediction", "table-macro"};
	for (auto &m : PREDICTION_MODELS) {
		string prefix = m.prefix;
		string extra_example = !m.extra_col ? "" : ", w";

		vector<string> params = {"source", "group_col", "x_col", "y_col"};
		if (!m.single_x) {
			params.push_back("x_cols");
		}
		if (m.extra_col) {
			params.push_back(m.extra_col);
		}
		auto pred = CreateContractTableMacro(
		    prefix + "_prediction_by", params, {{"options", "NULL"}, {"level", "0.95"}, {"split", "NULL"}},
		    PredictionSql(m),
		    "Table macro: fits " + string(m.label) + " per group of source (" + prefix +
		        "_fit_predict_by) and returns one row per input row in the integration contract's `prediction` "
		        "schema: the group column, id, model_id, model_name ('" +
		        prefix +
		        "'), x (x_col, the plot position and row order" + (m.single_x ? ", and the model's one feature" : "") +
		        "), y, yhat, yhat_lower, yhat_upper (the interval at "
		        "level, default 0.95, overriding any confidence_level in options), level (NULL on rows without a "
		        "band), split ('train' for "
		        "fitted rows, 'future' for rows with a NULL y, 'test' for rows held out by split).",
		    m.single_x ? prefix + "_prediction_by('sales', store, price, revenue)"
		               : prefix + "_prediction_by('sales', store, week, revenue, [price, promo]" + extra_example + ")",
		    categories);
		loader.RegisterFunction(*pred);

		if (m.effects) {
			vector<string> eparams = {"source", "group_col", "y_col", "x_cols"};
			if (m.extra_col) {
				eparams.push_back(m.extra_col);
			}
			auto eff = CreateContractTableMacro(
			    prefix + "_effects_by", eparams,
			    {{"focal", "1"}, {"n_points", "50"}, {"at_values", "NULL"}, {"options", "NULL"}, {"level", "0.95"}},
			    EffectsSql(m),
			    "Table macro: marginal-effect grid of " + string(m.label) +
			        " per group of source: the fitted mean over n_points values spanning the range of feature "
			        "focal (1-based position in x_cols), with the other features at their group means over the fitted rows (or at "
			        "at_values, a list with one value per feature), and its confidence band at level from "
			        "predict(model, x, {'interval': 'confidence'}) (response scale for GLMs). Integration contract "
			        "`prediction` schema: the group column, id, model_id, model_name, focal, x (the focal value), y "
			        "(NULL), yhat, yhat_lower, yhat_upper, level, split (NULL), se_fit.",
			    prefix + "_effects_by('sales', store, revenue, [price, promo]" + extra_example +
			        ", focal := 1, n_points := 25)",
			    categories);
			loader.RegisterFunction(*eff);
		}
	}
}

} // namespace duckdb
