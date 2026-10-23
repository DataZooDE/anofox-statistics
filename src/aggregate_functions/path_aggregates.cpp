// Coefficient paths, cross-validation curves and the regression-quantile process.
//
//   ridge_path_agg / elasticnet_path_agg / lars_path_agg (#154) -> LIST of `terms`
//     rows, one per lambda and term: index_name 'lambda', index_value the penalty
//     (LARS: index_name 'alpha', upstream's LARS scale, one row per knot and term).
//   ridge_cv_agg / elasticnet_cv_agg (#154) -> LIST of `curve` rows, curve_type
//     'lambda_cv': x = lambda, y = cv_mean, y_low / y_high = cv_mean -/+ cv_se, with
//     lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se.
//   ridge_nested_cv_agg / elasticnet_nested_cv_agg (#204) -> LIST of `curve` rows,
//     curve_type 'nested_cv': one row per outer fold (series 'fold', x = fold,
//     y = test error at the lambda chosen by the inner CV) and a summary row
//     (series 'mean', y = mean outer error, y_low / y_high = mean -/+ se).
//
// The CV aggregates take an optional cv_key argument (#204): the fold label, group,
// stratum or time of each row, by cv_strategy. Integer keys are used as they are,
// time keys (numbers, dates, times, timestamps) by their order, other keys (text,
// STRUCTs of several group variables, ...) by their hash.
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
// rows canonically (a cv_key travels with its row).

#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
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

enum class PathKind : uint8_t {
	RIDGE_PATH,
	ELASTICNET_PATH,
	LARS_PATH,
	RIDGE_CV,
	ELASTICNET_CV,
	RIDGE_NESTED_CV,
	ELASTICNET_NESTED_CV,
	QUANTILE_PROCESS
};

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
	case PathKind::RIDGE_NESTED_CV:
		return "ridge_nested_cv_agg";
	case PathKind::ELASTICNET_NESTED_CV:
		return "elasticnet_nested_cv_agg";
	case PathKind::QUANTILE_PROCESS:
		return "quantile_process_agg";
	}
	return "path_agg";
}

bool IsCv(PathKind kind) {
	return kind == PathKind::RIDGE_CV || kind == PathKind::ELASTICNET_CV;
}

bool IsNested(PathKind kind) {
	return kind == PathKind::RIDGE_NESTED_CV || kind == PathKind::ELASTICNET_NESTED_CV;
}

bool UsesFolds(PathKind kind) {
	return IsCv(kind) || IsNested(kind);
}

bool IsElasticNet(PathKind kind) {
	return kind == PathKind::ELASTICNET_PATH || kind == PathKind::ELASTICNET_CV ||
	       kind == PathKind::ELASTICNET_NESTED_CV;
}

//! How a cv_key value becomes the DOUBLE that travels with its row.
enum class CvKeyMode : uint8_t {
	//! No key (no cv_key argument, or a NULL constant).
	NONE,
	//! Integer key, used as it is.
	INTEGER,
	//! Time key (number, date, time, timestamp): its order matters.
	ORDER,
	//! Any other key (text, STRUCT, ...): hashed; only equality matters.
	HASH
};

const char *StrategyName(int32_t strategy) {
	switch (strategy) {
	case ANOFOX_CV_FOLD_IDS:
		return "fold_id";
	case ANOFOX_CV_GROUP:
		return "group";
	case ANOFOX_CV_LOGO:
		return "logo";
	case ANOFOX_CV_STRATIFIED:
		return "stratified";
	case ANOFOX_CV_TIME_SERIES:
		return "time_series";
	default:
		return "kfold";
	}
}

struct PathBindData : public FunctionData {
	PathKind kind = PathKind::RIDGE_PATH;
	CvKeyMode key_mode = CvKeyMode::NONE;
	AnofoxCvSpec cv {};
	bool fit_intercept = true;
	bool standardize = false;
	double l1_ratio = 0.0;
	vector<double> lambdas;
	uint32_t n_lambda = 100;
	double lambda_min_ratio = std::numeric_limits<double>::quiet_NaN();
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
		return kind == o.kind && key_mode == o.key_mode && cv.strategy == o.cv.strategy &&
		       cv.n_folds == o.cv.n_folds && cv.seed == o.cv.seed && cv.repeats == o.cv.repeats &&
		       cv.test_size == o.cv.test_size && cv.gap == o.cv.gap && cv.max_train_size == o.cv.max_train_size &&
		       cv.n_bins == o.cv.n_bins && cv.mae == o.cv.mae && cv.inner_folds == o.cv.inner_folds &&
		       cv.one_se == o.cv.one_se && fit_intercept == o.fit_intercept &&
		       standardize == o.standardize && l1_ratio == o.l1_ratio && lambdas == o.lambdas &&
		       n_lambda == o.n_lambda && same(lambda_min_ratio, o.lambda_min_ratio) &&
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

//! The `curve` row of a nested CV: one per outer fold plus a summary row.
LogicalType NestedRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("curve_type", LogicalType::VARCHAR));
	c.push_back(make_pair("series", LogicalType::VARCHAR));
	c.push_back(make_pair("x", LogicalType::DOUBLE));
	c.push_back(make_pair("y", LogicalType::DOUBLE));
	c.push_back(make_pair("y_low", LogicalType::DOUBLE));
	c.push_back(make_pair("y_high", LogicalType::DOUBLE));
	c.push_back(make_pair("threshold", LogicalType::DOUBLE));
	c.push_back(make_pair("fold", LogicalType::BIGINT));
	c.push_back(make_pair("lambda", LogicalType::DOUBLE));
	c.push_back(make_pair("test_error", LogicalType::DOUBLE));
	c.push_back(make_pair("test_se", LogicalType::DOUBLE));
	c.push_back(make_pair("inner_cv_mean", LogicalType::DOUBLE));
	c.push_back(make_pair("inner_cv_se", LogicalType::DOUBLE));
	c.push_back(make_pair("n_train", LogicalType::BIGINT));
	c.push_back(make_pair("n_test", LogicalType::BIGINT));
	c.push_back(make_pair("n_inner_folds", LogicalType::BIGINT));
	c.push_back(make_pair("n_folds", LogicalType::BIGINT));
	c.push_back(make_pair("metric", LogicalType::VARCHAR));
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

void WriteNested(const PathBindData &bind, const AnofoxNestedCvResult &r, Vector &result, idx_t row) {
	idx_t out = ReserveRows(result, row, r.n_folds + 1);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	const char *metric = bind.cv.mae ? "mae" : "mse";
	auto set_null = [&](idx_t field, idx_t i) {
		FlatVector::SetNull(StructEntry(f[field]), i, true);
	};
	auto set_int = [&](idx_t field, idx_t i, double v) {
		FlatDataMut<int64_t>(StructEntry(f[field]))[i] = static_cast<int64_t>(v);
	};
	double n_tested = 0.0;
	for (idx_t k = 0; k < r.n_folds; k++, out++) {
		StructEntry(f[0]).SetValue(out, Value("nested_cv"));
		StructEntry(f[1]).SetValue(out, Value("fold"));
		XyRowsSetDouble(StructEntry(f[2]), out, static_cast<double>(k + 1));
		XyRowsSetDouble(StructEntry(f[3]), out, r.test_error[k]);
		set_null(4, out);
		set_null(5, out);
		set_null(6, out);
		set_int(7, out, static_cast<double>(k + 1));
		XyRowsSetDouble(StructEntry(f[8]), out, r.lambda[k]);
		XyRowsSetDouble(StructEntry(f[9]), out, r.test_error[k]);
		set_null(10, out);
		XyRowsSetDouble(StructEntry(f[11]), out, r.inner_cv_mean[k]);
		XyRowsSetDouble(StructEntry(f[12]), out, r.inner_cv_se[k]);
		set_int(13, out, r.n_train[k]);
		set_int(14, out, r.n_test[k]);
		set_int(15, out, r.n_inner_folds[k]);
		set_int(16, out, static_cast<double>(r.n_folds));
		StructEntry(f[17]).SetValue(out, Value(metric));
		n_tested += r.n_test[k];
	}
	// Summary row: the outer error averaged over the folds (weighted by their test rows).
	StructEntry(f[0]).SetValue(out, Value("nested_cv"));
	StructEntry(f[1]).SetValue(out, Value("mean"));
	set_null(2, out);
	XyRowsSetDouble(StructEntry(f[3]), out, r.mean);
	XyRowsSetDouble(StructEntry(f[4]), out, r.mean - r.se);
	XyRowsSetDouble(StructEntry(f[5]), out, r.mean + r.se);
	set_null(6, out);
	set_null(7, out);
	set_null(8, out);
	XyRowsSetDouble(StructEntry(f[9]), out, r.mean);
	XyRowsSetDouble(StructEntry(f[10]), out, r.se);
	set_null(11, out);
	set_null(12, out);
	set_null(13, out);
	set_int(14, out, n_tested);
	set_null(15, out);
	set_int(16, out, static_cast<double>(r.n_folds));
	StructEntry(f[17]).SetValue(out, Value(metric));
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
		// Canonical row order; a cv_key is sorted along as one more column.
		const bool with_key = bind.key_mode != CvKeyMode::NONE;
		if (with_key) {
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
			AnofoxCvPathResult cv;
			ok = anofox_cv_reg_path(y, x.data(), x.size(), with_key ? state.extra.data() : nullptr, bind.cv,
			                        PathOptions(bind), &cv, &error);
			if (ok) {
				WriteCv(cv, result, row);
				anofox_free_cv_path_result(&cv);
			}
			break;
		}
		case PathKind::RIDGE_NESTED_CV:
		case PathKind::ELASTICNET_NESTED_CV: {
			AnofoxNestedCvResult nested;
			ok = anofox_nested_cv_reg_path(y, x.data(), x.size(), with_key ? state.extra.data() : nullptr, bind.cv,
			                               PathOptions(bind), &nested, &error);
			if (ok) {
				WriteNested(bind, nested, result, row);
				anofox_free_nested_cv_result(&nested);
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

//! How the cv_key of type `type` travels with its row (see CvKeyMode).
CvKeyMode CvKeyModeFor(const string &name, const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::SQLNULL:
		return CvKeyMode::NONE;
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return CvKeyMode::INTEGER;
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_SEC:
		return CvKeyMode::ORDER;
	default:
		return CvKeyMode::HASH;
	}
}

//! The cv_key column as DOUBLEs (NULL stays NULL): integers and times by value
//! (dates as days, times and timestamps in their storage unit), anything else by
//! the top 53 bits of its hash, exact in a DOUBLE.
void CvKeysToDouble(Vector &input, Vector &keys, idx_t count, CvKeyMode mode) {
	UnifiedVectorFormat in;
	input.ToUnifiedFormat(count, in);
	auto out = FlatDataMut<double>(keys);
	auto &validity = FlatValidityMut(keys);
	if (mode == CvKeyMode::HASH) {
		Vector hashes(LogicalType::HASH, count);
		VectorOperations::Hash(input, hashes, count);
		UnifiedVectorFormat hf;
		hashes.ToUnifiedFormat(count, hf);
		auto h = UnifiedVectorFormat::GetData<hash_t>(hf);
		for (idx_t i = 0; i < count; i++) {
			if (!in.validity.RowIsValid(in.sel->get_index(i))) {
				validity.SetInvalid(i);
				continue;
			}
			out[i] = static_cast<double>(h[hf.sel->get_index(i)] >> 11);
		}
		return;
	}
	auto read = [&](auto *data) {
		for (idx_t i = 0; i < count; i++) {
			auto idx = in.sel->get_index(i);
			if (!in.validity.RowIsValid(idx)) {
				validity.SetInvalid(i);
				continue;
			}
			out[i] = static_cast<double>(data[idx]);
		}
	};
	switch (input.GetType().InternalType()) {
	case PhysicalType::INT8:
		return read(UnifiedVectorFormat::GetData<int8_t>(in));
	case PhysicalType::INT16:
		return read(UnifiedVectorFormat::GetData<int16_t>(in));
	case PhysicalType::INT32:
		return read(UnifiedVectorFormat::GetData<int32_t>(in));
	case PhysicalType::INT64:
		return read(UnifiedVectorFormat::GetData<int64_t>(in));
	case PhysicalType::UINT8:
		return read(UnifiedVectorFormat::GetData<uint8_t>(in));
	case PhysicalType::UINT16:
		return read(UnifiedVectorFormat::GetData<uint16_t>(in));
	case PhysicalType::UINT32:
		return read(UnifiedVectorFormat::GetData<uint32_t>(in));
	case PhysicalType::UINT64:
		return read(UnifiedVectorFormat::GetData<uint64_t>(in));
	case PhysicalType::FLOAT:
		return read(UnifiedVectorFormat::GetData<float>(in));
	case PhysicalType::DOUBLE:
		return read(UnifiedVectorFormat::GetData<double>(in));
	default:
		throw InternalException("cv_key: unexpected physical type %s", TypeIdToString(input.GetType().InternalType()));
	}
}

//! Update of the CV aggregates with a cv_key argument (inputs[2]).
void PathKeyUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<PathBindData>();
	if (bind.key_mode == CvKeyMode::NONE) {
		XyRowsUpdateImpl(inputs, count, state_vector);
		return;
	}
	Vector keys(LogicalType::DOUBLE, count);
	CvKeysToDouble(inputs[2], keys, count, bind.key_mode);
	XyRowsUpdateImpl(inputs, count, state_vector, &keys);
}

//! cv_strategy and its options -> the AnofoxCvSpec of the bind data.
void BindCvSpec(PathBindData &bind, const RegressionMapOptions &opts, const string &name) {
	const bool nested = IsNested(bind.kind);
	const bool with_key = bind.key_mode != CvKeyMode::NONE;
	string strategy = opts.cv_strategy.value_or(with_key ? "fold_id" : "kfold");
	auto &cv = bind.cv;
	cv.strategy = strategy == "fold_id"       ? ANOFOX_CV_FOLD_IDS
	              : strategy == "group"       ? ANOFOX_CV_GROUP
	              : strategy == "logo"        ? ANOFOX_CV_LOGO
	              : strategy == "stratified"  ? ANOFOX_CV_STRATIFIED
	              : strategy == "time_series" ? ANOFOX_CV_TIME_SERIES
	                                          : ANOFOX_CV_KFOLD;
	const bool time_series = cv.strategy == ANOFOX_CV_TIME_SERIES;
	if (with_key && cv.strategy == ANOFOX_CV_KFOLD) {
		throw InvalidInputException("%s: cv_strategy 'kfold' draws random folds and takes no cv_key; use 'fold_id' "
		                            "(the default with a cv_key) for given folds",
		                            name);
	}
	if (!with_key && cv.strategy != ANOFOX_CV_KFOLD && cv.strategy != ANOFOX_CV_STRATIFIED) {
		throw InvalidInputException("%s: cv_strategy '%s' needs a cv_key argument: %s(y, x, cv_key, options)", name,
		                            strategy, name);
	}
	if (time_series && bind.key_mode == CvKeyMode::HASH) {
		throw InvalidInputException("%s: cv_strategy 'time_series' orders the rows by cv_key, which must be a number, "
		                            "date, time or timestamp",
		                            name);
	}
	if (!time_series && (opts.test_size || opts.gap || opts.window || opts.max_train_size)) {
		throw InvalidInputException("%s: test_size, gap, window and max_train_size apply to cv_strategy 'time_series'",
		                            name);
	}
	if (opts.n_bins && (cv.strategy != ANOFOX_CV_STRATIFIED || with_key)) {
		throw InvalidInputException("%s: n_bins applies to cv_strategy 'stratified' without a cv_key (quantile strata "
		                            "of y)",
		                            name);
	}
	if (opts.repeats && opts.repeats.value() > 1 && cv.strategy != ANOFOX_CV_KFOLD &&
	    cv.strategy != ANOFOX_CV_STRATIFIED) {
		throw InvalidInputException("%s: repeats applies to cv_strategy 'kfold' and 'stratified'", name);
	}
	if (opts.window && opts.window.value() == "sliding" && !opts.max_train_size) {
		throw InvalidInputException("%s: window 'sliding' needs max_train_size", name);
	}
	if (opts.window && opts.window.value() == "expanding" && opts.max_train_size) {
		throw InvalidInputException("%s: max_train_size makes the window sliding; drop window 'expanding'", name);
	}
	// Defaults: 10 folds (cv.glmnet), 5 splits for time series (TimeSeriesSplit) and 5
	// outer folds for nested CV.
	uint32_t default_folds = (time_series || nested) ? 5 : 10;
	cv.n_folds = opts.folds.value_or(default_folds);
	uint32_t min_folds = cv.strategy == ANOFOX_CV_KFOLD ? 3 : 2;
	bool folds_used = cv.strategy != ANOFOX_CV_FOLD_IDS && cv.strategy != ANOFOX_CV_LOGO;
	if (folds_used && cv.n_folds < min_folds) {
		throw InvalidInputException("%s: folds must be at least %u for cv_strategy '%s', got %u", name, min_folds,
		                            StrategyName(cv.strategy), (unsigned)cv.n_folds);
	}
	cv.seed = bind.seed;
	cv.repeats = opts.repeats.value_or(1);
	cv.test_size = opts.test_size.value_or(0);
	cv.gap = opts.gap.value_or(0);
	cv.max_train_size = opts.max_train_size.value_or(0);
	cv.n_bins = opts.n_bins.value_or(5);
	cv.mae = opts.cv_metric.value_or("mse") == "mae";
	cv.inner_folds = opts.inner_folds.value_or(5);
	cv.one_se = opts.lambda_rule.value_or("min") == "1se";
}

unique_ptr<FunctionData> PathBindImpl(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &arguments, PathKind kind) {
	const string name(PathKindName(kind));
	auto bind = make_uniq<PathBindData>();
	bind->kind = kind;
	const bool key_arg = UsesFolds(kind) && arguments.size() == 4;
	if (key_arg) {
		bind->key_mode = CvKeyModeFor(name, arguments[2]->return_type);
	}
	if (IsElasticNet(kind)) {
		bind->l1_ratio = 0.5;
	}
	if (kind == PathKind::LARS_PATH) {
		// lars_fit_agg's default: features standardised before fitting.
		bind->standardize = true;
	}
	idx_t options_idx = key_arg ? 3 : 2;
	RegressionMapOptions cv_opts;
	if (arguments.size() > options_idx) {
		vector<string> keys = {"fit_intercept", "feature_names"};
		switch (kind) {
		case PathKind::RIDGE_PATH:
		case PathKind::ELASTICNET_PATH:
		case PathKind::RIDGE_CV:
		case PathKind::ELASTICNET_CV:
		case PathKind::RIDGE_NESTED_CV:
		case PathKind::ELASTICNET_NESTED_CV:
			keys.insert(keys.end(), {"standardize", "lambdas", "n_lambda", "lambda_min_ratio"});
			if (IsElasticNet(kind)) {
				keys.push_back("l1_ratio");
			}
			if (UsesFolds(kind)) {
				keys.insert(keys.end(), {"folds", "random_state", "cv_strategy", "test_size", "gap", "window",
				                         "max_train_size", "n_bins", "metric"});
			}
			if (IsCv(kind)) {
				keys.push_back("repeats");
			}
			if (IsNested(kind)) {
				keys.insert(keys.end(), {"inner_folds", "rule"});
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
		cv_opts = std::move(opts);
	}
	if (UsesFolds(kind)) {
		BindCvSpec(*bind, cv_opts, name);
	}
	for (auto tau : bind->taus) {
		if (!(tau > 0.0 && tau < 1.0)) {
			throw InvalidInputException("%s: taus must be in (0, 1), got %g", name, tau);
		}
	}
	SetReturnTypeOf(function, LogicalType::LIST(IsCv(kind)       ? CvRowType()
	                                            : IsNested(kind) ? NestedRowType()
	                                                             : TermsRowType()));
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
	set.AddFunction(MakePathAgg(name, {LogicalType::DOUBLE, list_double, LogicalType::ANY, LogicalType::ANY},
	                            PathKeyUpdate, ANOFOX_AGG_BIND(PathBind<KIND>)));
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	info.descriptions.push_back(
	    PathDescription(description, name + "(y, [x1, x2])", {"y", "x"}, {LogicalType::DOUBLE, list_double}));
	info.descriptions.push_back(PathDescription(description, example, {"y", "x", "options"},
	                                            {LogicalType::DOUBLE, list_double, LogicalType::ANY}));
	info.descriptions.push_back(PathDescription(
	    description +
	        " With cv_key, the folds follow cv_strategy: 'fold_id' (default; cv_key is each row's fold, any labels, "
	        "at least 3), 'group' (cv_key is a group that never spans two folds; a STRUCT of several columns for a "
	        "combined key), 'logo' (leave one group out), 'stratified' (cv_key is the stratum) or 'time_series' "
	        "(cv_key is the time: rolling-origin splits with test_size, gap, window, max_train_size).",
	    name + "(y, [x1, x2], store, {'cv_strategy': 'group', 'folds': 5})", {"y", "x", "cv_key", "options"},
	    {LogicalType::DOUBLE, list_double, LogicalType::ANY, LogicalType::ANY}));
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
	    "(type.measure = 'mse', or 'mae' with metric). Options: as the path, plus folds (10) and seed for the seeded "
	    "folds, repeats, cv_strategy and its options (see the cv_key form).";
	RegisterCv<PathKind::RIDGE_CV>(loader, "k-fold cross-validation of the ridge path per group." + cv_tail,
	                               "ridge_cv_agg(y, [x1, x2], {'folds': 5, 'seed': 1})");
	RegisterCv<PathKind::ELASTICNET_CV>(loader,
	                                    "k-fold cross-validation of the elastic-net path per group." + cv_tail,
	                                    "elasticnet_cv_agg(y, [x1, x2], {'l1_ratio': 0.5, 'folds': 5})");
	const string nested_tail =
	    " Each outer fold's training rows choose lambda by an inner CV (inner_folds folds built from those rows only, "
	    "respecting the same cv_key; rule 'min' or '1se') and the refit at that lambda is scored on the outer test "
	    "rows. Returns a LIST of `curve` rows (curve_type = 'nested_cv'): one per outer fold (series 'fold', x = "
	    "fold, y = test_error, with lambda, inner_cv_mean, inner_cv_se, n_train, n_test, n_inner_folds) and a summary "
	    "row (series 'mean', y = the mean outer error weighted by test rows, y_low / y_high = mean -/+ test_se). "
	    "Options: as the CV aggregate (cv_strategy, folds = outer folds, default 5; seed; metric) plus inner_folds "
	    "(5) and rule ('min').";
	RegisterCv<PathKind::RIDGE_NESTED_CV>(loader, "Nested cross-validation of the ridge path per group." + nested_tail,
	                                      "ridge_nested_cv_agg(y, [x1, x2], {'folds': 5, 'inner_folds': 5})");
	RegisterCv<PathKind::ELASTICNET_NESTED_CV>(
	    loader, "Nested cross-validation of the elastic-net path per group." + nested_tail,
	    "elasticnet_nested_cv_agg(y, [x1, x2], store, {'cv_strategy': 'group', 'rule': '1se'})");
	RegisterPath<PathKind::QUANTILE_PROCESS>(
	    loader,
	    "Regression-quantile process per group: a quantile regression at each of taus (default 0.1, ..., 0.9) with "
	    "standard errors as quantreg::summary.rq (se 'nid' default, 'iid', 'boot'; 'compute_inference': false for "
	    "estimates only). Returns a LIST of `terms` rows with index_name = 'tau', index_value = tau.",
	    "quantile_process_agg(y, [x1, x2], {'taus': [0.25, 0.5, 0.75], 'se': 'iid'})");
}

} // namespace duckdb
