// Coefficient paths, cross-validation curves and the regression-quantile process.
//
//   ridge_path_agg / elasticnet_path_agg / lars_path_agg (#154) -> LIST of `terms`
//     rows, one per lambda and term: index_name 'lambda', index_value the penalty
//     (LARS: index_name 'alpha', upstream's LARS scale, one row per knot and term).
//   ridge_cv_agg / elasticnet_cv_agg (#154) -> LIST of `curve` rows, curve_type
//     'lambda_cv': x = lambda, y = cv_mean, y_low / y_high = cv_mean -/+ cv_se, with
//     lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se.
//   quantile_process_agg (#157) -> LIST of `terms` rows, one per tau and term:
//     index_name 'tau', with standard errors and confidence limits.
//
// The paths follow glmnet(family = "gaussian") (lambda on glmnet's scale), the CV
// curve cv.glmnet(type.measure = "mse"), the quantile inference
// quantreg::summary.rq(se = "iid" | "nid" | "boot"). Everything is computed by
// anofox_regression (solvers::path, FittedLars::path, solvers::quantile_inference)
// through the FFI; this file collects the rows of a group and lays the result out.
//
// Rows arrive in whatever order DuckDB's threads deliver them. The seeded folds
// and the bootstrap draw rows by position, so every aggregate here first sorts the
// rows canonically (an explicit fold id travels with its row).

#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "aggregate_finalize_guard.hpp"
#include "canonical_order.hpp"
#include "error_dispatch.hpp"
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

namespace {

enum class PathKind : uint8_t { RIDGE_PATH, ELASTICNET_PATH, LARS_PATH, RIDGE_CV, ELASTICNET_CV, QUANTILE_PROCESS };

const char *PathKindName(PathKind kind) {
	switch (kind) {
	case PathKind::RIDGE_PATH:
		return "ridge_path_agg";
	case PathKind::ELASTICNET_PATH:
		return "elasticnet_path_agg";
	case PathKind::LARS_PATH:
		return "lars_path_agg";
	case PathKind::RIDGE_CV:
		return "ridge_cv_agg";
	case PathKind::ELASTICNET_CV:
		return "elasticnet_cv_agg";
	case PathKind::QUANTILE_PROCESS:
		return "quantile_process_agg";
	}
	return "path_agg";
}

bool IsCv(PathKind kind) {
	return kind == PathKind::RIDGE_CV || kind == PathKind::ELASTICNET_CV;
}

struct PathBindData : public FunctionData {
	PathKind kind = PathKind::RIDGE_PATH;
	bool with_fold_ids = false;
	bool fit_intercept = true;
	bool standardize = false;
	double l1_ratio = 0.0;
	vector<double> lambdas;
	uint32_t n_lambda = 100;
	double lambda_min_ratio = std::numeric_limits<double>::quiet_NaN();
	uint32_t folds = 10;
	uint64_t seed = 42;
	// LARS
	bool method_lasso = true;
	int64_t n_nonzero_coefs = 0;
	// Quantile process
	vector<double> taus = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
	int32_t se_type = ANOFOX_QUANTILE_SE_NID;
	double confidence_level = 0.95;
	uint32_t n_boot = 200;
	vector<string> names;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PathBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PathBindData>();
		auto same = [](double a, double b) {
			return a == b || (std::isnan(a) && std::isnan(b));
		};
		return kind == o.kind && with_fold_ids == o.with_fold_ids && fit_intercept == o.fit_intercept &&
		       standardize == o.standardize && l1_ratio == o.l1_ratio && lambdas == o.lambdas &&
		       n_lambda == o.n_lambda && same(lambda_min_ratio, o.lambda_min_ratio) && folds == o.folds &&
		       seed == o.seed && method_lasso == o.method_lasso && n_nonzero_coefs == o.n_nonzero_coefs &&
		       taus == o.taus && se_type == o.se_type && confidence_level == o.confidence_level &&
		       n_boot == o.n_boot && names == o.names;
	}
};

//! The `terms` row of a path / process (the element type of tidy()).
LogicalType TermsRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("estimate", LogicalType::DOUBLE));
	c.push_back(make_pair("std_error", LogicalType::DOUBLE));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	c.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	c.push_back(make_pair("conf_level", LogicalType::DOUBLE));
	c.push_back(make_pair("index_name", LogicalType::VARCHAR));
	c.push_back(make_pair("index_value", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

//! The `curve` row of a lambda CV curve.
LogicalType CvRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("curve_type", LogicalType::VARCHAR));
	c.push_back(make_pair("x", LogicalType::DOUBLE));
	c.push_back(make_pair("y", LogicalType::DOUBLE));
	c.push_back(make_pair("y_low", LogicalType::DOUBLE));
	c.push_back(make_pair("y_high", LogicalType::DOUBLE));
	c.push_back(make_pair("threshold", LogicalType::DOUBLE));
	c.push_back(make_pair("lambda", LogicalType::DOUBLE));
	c.push_back(make_pair("cv_mean", LogicalType::DOUBLE));
	c.push_back(make_pair("cv_se", LogicalType::DOUBLE));
	c.push_back(make_pair("n_nonzero", LogicalType::BIGINT));
	c.push_back(make_pair("is_min", LogicalType::BOOLEAN));
	c.push_back(make_pair("is_1se", LogicalType::BOOLEAN));
	c.push_back(make_pair("n_folds", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

string TermName(const PathBindData &bind, idx_t j) {
	return j < bind.names.size() ? bind.names[j] : "x" + std::to_string(j + 1);
}

//! Reserve `n` rows of a LIST(STRUCT) result for `row` and return the first child index.
idx_t ReserveRows(Vector &result, idx_t row, idx_t n) {
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + n);
	ListVector::SetListSize(result, offset + n);
	FlatDataMut<list_entry_t>(result)[row] = {offset, n};
	return offset;
}

struct TermsWriter {
	StructEntries &f;
	explicit TermsWriter(Vector &result) : f(StructVector::GetEntries(ListVector::GetEntry(result))) {
	}
	void Write(idx_t out, const string &term, double estimate, double se, double stat, double p, double lo,
	           double hi, double level, const char *index_name, double index_value) {
		StructEntry(f[0]).SetValue(out, Value(term));
		XyRowsSetDouble(StructEntry(f[1]), out, estimate);
		XyRowsSetDouble(StructEntry(f[2]), out, se);
		XyRowsSetDouble(StructEntry(f[3]), out, stat);
		XyRowsSetDouble(StructEntry(f[4]), out, p);
		XyRowsSetDouble(StructEntry(f[5]), out, lo);
		XyRowsSetDouble(StructEntry(f[6]), out, hi);
		XyRowsSetDouble(StructEntry(f[7]), out, level);
		StructEntry(f[8]).SetValue(out, Value(index_name));
		XyRowsSetDouble(StructEntry(f[9]), out, index_value);
	}
};

AnofoxRegPathOptions PathOptions(const PathBindData &bind) {
	AnofoxRegPathOptions o {};
	o.l1_ratio = bind.l1_ratio;
	o.fit_intercept = bind.fit_intercept;
	o.standardize = bind.standardize;
	o.lambdas = bind.lambdas.empty() ? nullptr : bind.lambdas.data();
	o.lambdas_len = bind.lambdas.size();
	o.n_lambda = bind.n_lambda;
	o.lambda_min_ratio = bind.lambda_min_ratio;
	return o;
}

void WritePath(const PathBindData &bind, const AnofoxRegPathResult &path, Vector &result, idx_t row) {
	idx_t p = path.n_features;
	idx_t per = p + (bind.fit_intercept ? 1 : 0);
	idx_t out = ReserveRows(result, row, path.n_lambda * per);
	TermsWriter w(result);
	const double nan = std::numeric_limits<double>::quiet_NaN();
	// The LARS knots are upstream's alpha, not glmnet's lambda.
	const char *index_name = bind.kind == PathKind::LARS_PATH ? "alpha" : "lambda";
	for (idx_t l = 0; l < path.n_lambda; l++) {
		double lambda = path.lambdas[l];
		if (bind.fit_intercept) {
			w.Write(out++, "(Intercept)", path.intercepts[l], nan, nan, nan, nan, nan, nan, index_name, lambda);
		}
		for (idx_t j = 0; j < p; j++) {
			w.Write(out++, TermName(bind, j), path.coefficients[l * p + j], nan, nan, nan, nan, nan, nan, index_name,
			        lambda);
		}
	}
}

void WriteCv(const AnofoxCvPathResult &cv, Vector &result, idx_t row) {
	idx_t out = ReserveRows(result, row, cv.n_lambda);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t l = 0; l < cv.n_lambda; l++, out++) {
		double m = cv.cv_mean[l];
		double se = cv.cv_se[l];
		StructEntry(f[0]).SetValue(out, Value("lambda_cv"));
		XyRowsSetDouble(StructEntry(f[1]), out, cv.lambdas[l]);
		XyRowsSetDouble(StructEntry(f[2]), out, m);
		XyRowsSetDouble(StructEntry(f[3]), out, m - se);
		XyRowsSetDouble(StructEntry(f[4]), out, m + se);
		FlatVector::SetNull(StructEntry(f[5]), out, true);
		XyRowsSetDouble(StructEntry(f[6]), out, cv.lambdas[l]);
		XyRowsSetDouble(StructEntry(f[7]), out, m);
		XyRowsSetDouble(StructEntry(f[8]), out, se);
		FlatDataMut<int64_t>(StructEntry(f[9]))[out] = static_cast<int64_t>(cv.df[l]);
		FlatDataMut<bool>(StructEntry(f[10]))[out] = l == cv.index_min;
		FlatDataMut<bool>(StructEntry(f[11]))[out] = l == cv.index_1se;
		FlatDataMut<int64_t>(StructEntry(f[12]))[out] = static_cast<int64_t>(cv.n_folds);
	}
}

void WriteProcess(const PathBindData &bind, const AnofoxQuantileProcessResult &q, Vector &result, idx_t row) {
	idx_t out = ReserveRows(result, row, q.n_tau * q.k);
	TermsWriter w(result);
	idx_t shift = q.has_intercept ? 1 : 0;
	for (idx_t t = 0; t < q.n_tau; t++) {
		for (idx_t j = 0; j < q.k; j++) {
			idx_t e = t * q.k + j;
			string term = (q.has_intercept && j == 0) ? string("(Intercept)") : TermName(bind, j - shift);
			w.Write(out++, term, q.estimate[e], q.std_error[e], q.statistic[e], q.p_value[e], q.conf_low[e],
			        q.conf_high[e], std::isfinite(q.std_error[e]) ? q.conf_level : std::nan(""), "tau", q.taus[t]);
		}
	}
}

void PathFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                  idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PathBindData>();
	const char *name = PathKindName(bind.kind);
	XyRowsStatePointers states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (!bind.names.empty() && bind.names.size() != state.n_features) {
			throw InvalidInputException("%s: feature_names has %llu entries but x has %llu features", name,
			                            (unsigned long long)bind.names.size(), (unsigned long long)state.n_features);
		}
		// Canonical row order; a fold id is sorted along as one more column.
		if (bind.with_fold_ids) {
			state.x_columns.push_back(std::move(state.extra));
			SortRowsCanonically(state.y_values, state.x_columns);
			state.extra = std::move(state.x_columns.back());
			state.x_columns.pop_back();
		} else {
			SortRowsCanonically(state.y_values, state.x_columns);
		}
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto x = XyRowsColumnArrays(state);
		AnofoxError error;
		bool ok = false;
		switch (bind.kind) {
		case PathKind::RIDGE_PATH:
		case PathKind::ELASTICNET_PATH:
		case PathKind::LARS_PATH: {
			AnofoxRegPathResult path;
			if (bind.kind == PathKind::LARS_PATH) {
				AnofoxLarsOptions options {};
				options.method_lasso = bind.method_lasso;
				options.fit_intercept = bind.fit_intercept;
				options.alpha = 0.0;
				options.n_nonzero_coefs = bind.n_nonzero_coefs;
				options.standardize = bind.standardize;
				ok = anofox_lars_path(y, x.data(), x.size(), options, &path, &error);
			} else {
				ok = anofox_reg_path(y, x.data(), x.size(), PathOptions(bind), &path, &error);
			}
			if (ok) {
				WritePath(bind, path, result, row);
				anofox_free_reg_path_result(&path);
			}
			break;
		}
		case PathKind::RIDGE_CV:
		case PathKind::ELASTICNET_CV: {
			vector<int64_t> fold_ids;
			for (auto v : state.extra) {
				fold_ids.push_back(static_cast<int64_t>(v));
			}
			AnofoxCvPathResult cv;
			ok = anofox_cv_reg_path(y, x.data(), x.size(), bind.with_fold_ids ? fold_ids.data() : nullptr,
			                        bind.folds, bind.seed, PathOptions(bind), &cv, &error);
			if (ok) {
				WriteCv(cv, result, row);
				anofox_free_cv_path_result(&cv);
			}
			break;
		}
		case PathKind::QUANTILE_PROCESS: {
			AnofoxQuantileProcessResult q;
			ok = anofox_quantile_process(y, x.data(), x.size(), bind.taus.data(), bind.taus.size(),
			                             bind.fit_intercept, bind.se_type, bind.seed, bind.n_boot,
			                             bind.confidence_level, &q, &error);
			if (ok) {
				WriteProcess(bind, q, result, row);
				anofox_free_quantile_process_result(&q);
			}
			break;
		}
		}
		if (!ok) {
			ThrowUnlessDegenerate(name, error);
			FlatVector::SetNull(result, row, true);
		}
	}
}

unique_ptr<FunctionData> PathBindImpl(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &arguments, PathKind kind) {
	const string name(PathKindName(kind));
	auto bind = make_uniq<PathBindData>();
	bind->kind = kind;
	bind->with_fold_ids = IsCv(kind) && arguments.size() == 4;
	if (kind == PathKind::ELASTICNET_PATH || kind == PathKind::ELASTICNET_CV) {
		bind->l1_ratio = 0.5;
	}
	if (kind == PathKind::LARS_PATH) {
		// lars_fit_agg's default: features standardised before fitting.
		bind->standardize = true;
	}
	idx_t options_idx = bind->with_fold_ids ? 3 : 2;
	if (arguments.size() > options_idx) {
		vector<string> keys = {"fit_intercept", "feature_names"};
		switch (kind) {
		case PathKind::RIDGE_PATH:
		case PathKind::ELASTICNET_PATH:
		case PathKind::RIDGE_CV:
		case PathKind::ELASTICNET_CV:
			keys.insert(keys.end(), {"standardize", "lambdas", "n_lambda", "lambda_min_ratio"});
			if (kind == PathKind::ELASTICNET_PATH || kind == PathKind::ELASTICNET_CV) {
				keys.push_back("l1_ratio");
			}
			if (IsCv(kind)) {
				keys.insert(keys.end(), {"folds", "random_state"});
			}
			break;
		case PathKind::LARS_PATH:
			keys.insert(keys.end(), {"method", "standardize", "n_nonzero_coefs"});
			break;
		case PathKind::QUANTILE_PROCESS:
			keys.insert(keys.end(),
			            {"taus", "se", "compute_inference", "confidence_level", "n_boot", "random_state"});
			break;
		}
		auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[options_idx], name, keys);
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.feature_names.has_value()) {
			bind->names = opts.feature_names.value();
		}
		if (opts.standardize.has_value()) {
			bind->standardize = opts.standardize.value();
		}
		if (opts.lambdas.has_value()) {
			bind->lambdas = opts.lambdas.value();
		}
		if (opts.n_lambda.has_value()) {
			bind->n_lambda = opts.n_lambda.value();
		}
		if (opts.lambda_min_ratio.has_value()) {
			bind->lambda_min_ratio = opts.lambda_min_ratio.value();
		}
		if (opts.l1_ratio.has_value()) {
			bind->l1_ratio = opts.l1_ratio.value();
			if (!(bind->l1_ratio >= 0.0 && bind->l1_ratio <= 1.0)) {
				throw InvalidInputException("%s: l1_ratio must be in [0, 1], got %g", name, bind->l1_ratio);
			}
		}
		if (opts.folds.has_value()) {
			bind->folds = opts.folds.value();
		}
		if (opts.random_state.has_value()) {
			bind->seed = opts.random_state.value();
		}
		if (opts.lars_lasso.has_value()) {
			bind->method_lasso = opts.lars_lasso.value();
		}
		if (opts.n_nonzero_coefs.has_value()) {
			bind->n_nonzero_coefs = opts.n_nonzero_coefs.value();
		}
		if (opts.taus.has_value()) {
			bind->taus = opts.taus.value();
		}
		if (opts.se_type.has_value()) {
			const auto &se = opts.se_type.value();
			bind->se_type = se == "iid"    ? ANOFOX_QUANTILE_SE_IID
			                : se == "boot" ? ANOFOX_QUANTILE_SE_BOOT
			                               : ANOFOX_QUANTILE_SE_NID;
		}
		if (opts.compute_inference.has_value() && !opts.compute_inference.value()) {
			bind->se_type = ANOFOX_QUANTILE_SE_NONE;
		}
		if (opts.confidence_level.has_value()) {
			bind->confidence_level = opts.confidence_level.value();
		}
		if (opts.n_boot.has_value()) {
			bind->n_boot = opts.n_boot.value();
		}
	}
	if (IsCv(kind) && !bind->with_fold_ids && bind->folds < 3) {
		throw InvalidInputException("%s: folds must be at least 3, got %u", name, bind->folds);
	}
	for (auto tau : bind->taus) {
		if (!(tau > 0.0 && tau < 1.0)) {
			throw InvalidInputException("%s: taus must be in (0, 1), got %g", name, tau);
		}
	}
	SetReturnTypeOf(function, LogicalType::LIST(IsCv(kind) ? CvRowType() : TermsRowType()));
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

template <PathKind KIND>
unique_ptr<FunctionData> PathBind(ClientContext &context, AggregateBindFunction &function,
                                  vector<unique_ptr<Expression>> &arguments) {
	return PathBindImpl(context, function, arguments, KIND);
}

AggregateFunction MakePathAgg(const string &name, vector<LogicalType> args, aggregate_update_t update,
                              bind_aggregate_function_t bind) {
	return AggregateFunction(AsDuckName(name), std::move(args), LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
	                         ANOFOX_AGG_INIT(XyRowsInitialize), update, XyRowsCombine,
	                         ANOFOX_GUARDED_FINALIZE(PathFinalize, XyRowsDestroy, XyRowsInitialize), nullptr, bind,
	                         XyRowsDestroy);
}

FunctionDescription PathDescription(const string &description, const string &example, vector<string> names,
                                    vector<LogicalType> types) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = {"regression"};
	d.parameter_names = std::move(names);
	d.parameter_types = std::move(types);
	return d;
}

template <PathKind KIND>
void RegisterPath(ExtensionLoader &loader, const string &description, const string &example) {
	const string name = PathKindName(KIND);
	auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	AggregateFunctionSet set(AsDuckName(name));
	set.AddFunction(MakePathAgg(name, {LogicalType::DOUBLE, list_double}, XyRowsUpdate, ANOFOX_AGG_BIND(PathBind<KIND>)));
	set.AddFunction(
	    MakePathAgg(name, {LogicalType::DOUBLE, list_double, LogicalType::ANY}, XyRowsUpdate, ANOFOX_AGG_BIND(PathBind<KIND>)));
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	info.descriptions.push_back(
	    PathDescription(description, name + "(y, [x1, x2])", {"y", "x"}, {LogicalType::DOUBLE, list_double}));
	info.descriptions.push_back(PathDescription(description, example, {"y", "x", "options"},
	                                            {LogicalType::DOUBLE, list_double, LogicalType::ANY}));
	loader.RegisterFunction(std::move(info));
}

template <PathKind KIND>
void RegisterCv(ExtensionLoader &loader, const string &description, const string &example) {
	const string name = PathKindName(KIND);
	auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	AggregateFunctionSet set(AsDuckName(name));
	set.AddFunction(MakePathAgg(name, {LogicalType::DOUBLE, list_double}, XyRowsUpdate, ANOFOX_AGG_BIND(PathBind<KIND>)));
	set.AddFunction(
	    MakePathAgg(name, {LogicalType::DOUBLE, list_double, LogicalType::ANY}, XyRowsUpdate, ANOFOX_AGG_BIND(PathBind<KIND>)));
	set.AddFunction(MakePathAgg(name, {LogicalType::DOUBLE, list_double, LogicalType::BIGINT, LogicalType::ANY},
	                            XyRowsExtraUpdate, ANOFOX_AGG_BIND(PathBind<KIND>)));
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	info.descriptions.push_back(
	    PathDescription(description, name + "(y, [x1, x2])", {"y", "x"}, {LogicalType::DOUBLE, list_double}));
	info.descriptions.push_back(PathDescription(description, example, {"y", "x", "options"},
	                                            {LogicalType::DOUBLE, list_double, LogicalType::ANY}));
	info.descriptions.push_back(PathDescription(
	    description + " With fold_id, each row's fold is given (any labels, at least 3 distinct).",
	    name + "(y, [x1, x2], fold_id, {'lambdas': [1.0, 0.1, 0.01]})", {"y", "x", "fold_id", "options"},
	    {LogicalType::DOUBLE, list_double, LogicalType::BIGINT, LogicalType::ANY}));
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterPathAggregateFunctions(ExtensionLoader &loader) {
	const string path_tail =
	    " Returns a LIST of `terms` rows (term, estimate, std_error, statistic, p_value, conf_low, conf_high, "
	    "conf_level, index_name = 'lambda', index_value = lambda), one per lambda and term, intercept first. Options: "
	    "lambdas (default glmnet's grid), n_lambda (100), lambda_min_ratio, standardize (false), fit_intercept, "
	    "feature_names.";
	RegisterPath<PathKind::RIDGE_PATH>(
	    loader,
	    "Ridge coefficient path per group, as glmnet(alpha = 0): lambda is glmnet's per-observation penalty (1/(2n) "
	    "RSS + lambda/2 ||b||^2 with y scaled to unit variance internally)." +
	        path_tail,
	    "ridge_path_agg(y, [x1, x2], {'n_lambda': 20})");
	RegisterPath<PathKind::ELASTICNET_PATH>(
	    loader,
	    "Elastic-net coefficient path per group, as glmnet(alpha = l1_ratio) (default l1_ratio 0.5)." + path_tail,
	    "elasticnet_path_agg(y, [x1, x2], {'l1_ratio': 0.5, 'lambdas': [1.0, 0.1, 0.01]})");
	RegisterPath<PathKind::LARS_PATH>(
	    loader,
	    "LARS / Lasso-LARS coefficient path per group, one entry per knot (index_name 'alpha', index_value the "
	    "knot's alpha = max_j |x_j'(y - yhat)| on the centred design, columns scaled to unit norm with "
	    "standardize; this is n * glmnet's lambda without standardize and sqrt(n) * lambda with it). Options: "
	    "method ('lasso' default, 'lar'), standardize (true), n_nonzero_coefs, fit_intercept, feature_names.",
	    "lars_path_agg(y, [x1, x2, x3])");
	const string cv_tail =
	    " Returns a LIST of `curve` rows (curve_type = 'lambda_cv', x = lambda, y = cv_mean, y_low / y_high = cv_mean "
	    "-/+ cv_se, threshold NULL, lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se, n_folds), as cv.glmnet "
	    "(type.measure = 'mse'). Options: as the path, plus folds (10) and seed for the seeded folds.";
	RegisterCv<PathKind::RIDGE_CV>(loader, "k-fold cross-validation of the ridge path per group." + cv_tail,
	                               "ridge_cv_agg(y, [x1, x2], {'folds': 5, 'seed': 1})");
	RegisterCv<PathKind::ELASTICNET_CV>(loader,
	                                    "k-fold cross-validation of the elastic-net path per group." + cv_tail,
	                                    "elasticnet_cv_agg(y, [x1, x2], {'l1_ratio': 0.5, 'folds': 5})");
	RegisterPath<PathKind::QUANTILE_PROCESS>(
	    loader,
	    "Regression-quantile process per group: a quantile regression at each of taus (default 0.1, ..., 0.9) with "
	    "standard errors as quantreg::summary.rq (se 'nid' default, 'iid', 'boot'; 'compute_inference': false for "
	    "estimates only). Returns a LIST of `terms` rows with index_name = 'tau', index_value = tau.",
	    "quantile_process_agg(y, [x1, x2], {'taus': [0.25, 0.5, 0.75], 'se': 'iid'})");
}

} // namespace duckdb
