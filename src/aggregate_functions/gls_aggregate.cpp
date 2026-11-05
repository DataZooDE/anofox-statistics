#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/anofox_statistics_extension.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "../include/canonical_order.hpp"
#include "telemetry.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Generalized least squares (#315), nlme::gls on anofox-regression's
// solvers::gls::GlsRegressor.
//
//   gls_fit_agg(y DOUBLE, x DOUBLE[], subject ANY, time DOUBLE [, options])
//   gls_fit_agg(y, x, subject, time, variance_covariate ANY, options)
//   gls_augment_agg(...)  -- same arguments, a window aggregate over a partition
//
// The errors of one subject are correlated (options.correlation) and their
// variance may depend on variance_covariate (options.variance: 'ident' takes it
// as a stratum, 'power' / 'exp' as a number). Subjects and varIdent strata are
// dictionary-encoded here, so any DuckDB type works.
//
// The fit does not depend on the order in which DuckDB delivers the rows: they
// are handed to the solver sorted by subject (in label order) and time (#226).
// Consequences, documented in docs/api/regression/gls.md: the normalized
// residuals decorrelate each subject's rows in time order, and the varIdent
// reference stratum is the stratum of the earliest row of the first subject
// (nlme takes the stratum of the first row of the data, which is the same row
// when the data are sorted by subject and time).
//===--------------------------------------------------------------------===//

namespace {

enum class GlsMode { FIT, AUGMENT };

const char *GlsModeName(GlsMode mode) {
	return mode == GlsMode::FIT ? "gls_fit_agg" : "gls_augment_agg";
}

const char *const CORRELATION_NAMES[] = {"independent", "compound_symmetry", "ar1",       "car1",        "exponential",
                                         "gaussian",    "linear",            "spherical", "unstructured"};
const char *const VARIANCE_NAMES[] = {"constant", "ident", "power", "exp"};

struct GlsBindData : public FunctionData {
	GlsMode mode = GlsMode::FIT;
	bool has_covariate = false;
	bool compute_inference = false;
	AnofoxGlsOptions options {0.95, 500, 0, 0, 0, 0, false, true, true};

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<GlsBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<GlsBindData>();
		return mode == o.mode && has_covariate == o.has_covariate && compute_inference == o.compute_inference &&
		       options.confidence_level == o.options.confidence_level &&
		       options.max_iterations == o.options.max_iterations && options.correlation == o.options.correlation &&
		       options.variance == o.options.variance && options.variogram_bins == o.options.variogram_bins &&
		       options.variogram_residuals == o.options.variogram_residuals && options.nugget == o.options.nugget &&
		       options.reml == o.options.reml && options.fit_intercept == o.options.fit_intercept;
	}
	bool UsesCovariate() const {
		return options.variance != 0;
	}
	bool IsIdent() const {
		return options.variance == 1;
	}
};

//! A string -> dense id dictionary in first-seen order. Held behind a pointer
//! because DuckDB relocates aggregate states with a raw memory copy.
struct GlsDictionary {
	unique_ptr<unordered_map<string, int32_t>> index;
	vector<string> labels;

	int32_t Intern(const string &key) {
		if (!index) {
			index = make_uniq<unordered_map<string, int32_t>>();
		}
		auto it = index->find(key);
		if (it != index->end()) {
			return it->second;
		}
		auto id = (int32_t)labels.size();
		index->emplace(key, id);
		labels.push_back(key);
		return id;
	}

	//! Renumber in sorted label order; returns old id -> new id.
	vector<int32_t> SortedRanks() {
		const idx_t n = labels.size();
		vector<idx_t> order(n);
		std::iota(order.begin(), order.end(), idx_t(0));
		std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return labels[a] < labels[b]; });
		vector<int32_t> rank(n);
		vector<string> sorted(n);
		for (idx_t r = 0; r < n; r++) {
			rank[order[r]] = (int32_t)r;
			sorted[r] = labels[order[r]];
		}
		labels = std::move(sorted);
		index = make_uniq<unordered_map<string, int32_t>>();
		for (idx_t r = 0; r < n; r++) {
			index->emplace(labels[r], (int32_t)r);
		}
		return rank;
	}
};

//! The rows of one group. The fit aggregate keeps only complete rows; the augment
//! aggregate keeps every row (NaN / -1 for what is missing) so its output lines up
//! with the frame.
struct GlsState {
	vector<double> y;
	vector<vector<double>> x_columns;
	vector<int32_t> subject; // -1: NULL
	vector<double> time;
	vector<double> covariate; // numeric covariate or stratum id (NaN: NULL)
	GlsDictionary subjects;
	GlsDictionary strata;
	idx_t n_features = 0;
	bool initialized = false;
};

void GlsInitialize(data_ptr_t state_p) {
	new (state_p) GlsState();
}

void GlsDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~GlsState();
	}
}

void EnsureFeatures(GlsState &state, idx_t n_features) {
	if (!state.initialized) {
		state.n_features = n_features;
		state.x_columns.resize(n_features);
		state.initialized = true;
	} else if (n_features != state.n_features) {
		throw InvalidInputException("Inconsistent feature count: expected %llu, got %llu",
		                            (unsigned long long)state.n_features, (unsigned long long)n_features);
	}
}

void GlsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
               idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<GlsBindData>();
	const bool keep_all = bind.mode == GlsMode::AUGMENT;
	const double nan = std::numeric_limits<double>::quiet_NaN();

	UnifiedVectorFormat y_data, x_data, t_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	inputs[3].ToUnifiedFormat(count, t_data);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto t_values = UnifiedVectorFormat::GetData<double>(t_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	// A LIST containing NULLs is not itself NULL: read the child mask and pass NaN
	// for a NULL element; the Rust side drops rows that are not finite throughout.
	auto &x_child_validity = FlatValidityMut(x_child);
	auto &subject_vec = inputs[2];
	Vector *cov_vec = (bind.has_covariate && bind.UsesCovariate()) ? &inputs[4] : nullptr;

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		auto t_idx = t_data.sel->get_index(i);
		const bool y_ok = y_data.validity.RowIsValid(y_idx);
		const bool x_ok = x_data.validity.RowIsValid(x_idx);
		const bool t_ok = t_data.validity.RowIsValid(t_idx);
		Value subject_value = subject_vec.GetValue(i);
		Value cov_value = cov_vec ? cov_vec->GetValue(i) : Value();
		const bool cov_ok = !cov_vec || !cov_value.IsNull();
		if (!keep_all && !(y_ok && x_ok && t_ok && !subject_value.IsNull() && cov_ok)) {
			continue;
		}
		if (x_ok) {
			EnsureFeatures(state, x_lists[x_idx].length);
		}

		state.y.push_back(y_ok ? y_values[y_idx] : nan);
		state.time.push_back(t_ok ? t_values[t_idx] : nan);
		state.subject.push_back(subject_value.IsNull() ? -1 : state.subjects.Intern(subject_value.ToString()));
		if (cov_vec) {
			if (cov_value.IsNull()) {
				state.covariate.push_back(nan);
			} else if (bind.IsIdent()) {
				state.covariate.push_back((double)state.strata.Intern(cov_value.ToString()));
			} else {
				double v;
				try {
					v = cov_value.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
				} catch (std::exception &) {
					throw InvalidInputException("%s: the variance covariate of variance '%s' must be numeric, got "
					                            "'%s'",
					                            GlsModeName(bind.mode), VARIANCE_NAMES[bind.options.variance],
					                            cov_value.ToString());
				}
				state.covariate.push_back(v);
			}
		}
		// x: a NULL list is NaN features (augment only; the fit skipped the row).
		const idx_t start = state.y.size() - 1;
		for (idx_t j = 0; j < state.n_features; j++) {
			while (state.x_columns[j].size() < start) {
				state.x_columns[j].push_back(nan);
			}
			if (x_ok) {
				const idx_t child = x_lists[x_idx].offset + j;
				state.x_columns[j].push_back(x_child_validity.RowIsValid(child) ? x_child_data[child] : nan);
			} else {
				state.x_columns[j].push_back(nan);
			}
		}
	}
}

void GlsCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (GlsState **)source_data.data;
	auto targets = (GlsState **)target_data.data;
	const double nan = std::numeric_limits<double>::quiet_NaN();

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.y.empty()) {
			continue;
		}
		if (source.initialized) {
			EnsureFeatures(target, source.n_features);
		}
		const bool has_cov = !source.covariate.empty();
		const bool ident = !source.strata.labels.empty();
		// The states interned subjects / strata independently: remap the source's
		// dense ids into the target's dictionaries.
		for (idx_t r = 0; r < source.y.size(); r++) {
			target.y.push_back(source.y[r]);
			target.time.push_back(source.time[r]);
			auto s = source.subject[r];
			target.subject.push_back(s < 0 ? -1 : target.subjects.Intern(source.subjects.labels[s]));
			if (has_cov) {
				double c = source.covariate[r];
				if (ident && !std::isnan(c)) {
					c = (double)target.strata.Intern(source.strata.labels[(idx_t)c]);
				}
				target.covariate.push_back(c);
			}
			for (idx_t j = 0; j < target.n_features; j++) {
				target.x_columns[j].push_back(source.initialized ? source.x_columns[j][r] : nan);
			}
		}
	}
}

//! Number subjects and strata by sorted label and sort the rows by subject, time,
//! x, y and the covariate, so the fit is independent of the delivery order. Returns
//! the input row of every sorted row.
vector<idx_t> CanonicalizeGls(GlsState &state) {
	const idx_t n = state.y.size();
	auto subject_rank = state.subjects.SortedRanks();
	for (auto &s : state.subject) {
		if (s >= 0) {
			s = subject_rank[(idx_t)s];
		}
	}
	if (!state.strata.labels.empty()) {
		auto stratum_rank = state.strata.SortedRanks();
		for (auto &c : state.covariate) {
			if (!std::isnan(c)) {
				c = (double)stratum_rank[(idx_t)c];
			}
		}
	}
	for (auto &col : state.x_columns) {
		col.resize(n, std::numeric_limits<double>::quiet_NaN());
	}
	vector<double> subject_key(state.subject.begin(), state.subject.end());
	vector<double> row_index(n);
	std::iota(row_index.begin(), row_index.end(), 0.0);
	CanonicalRows rows;
	rows.Key(subject_key).Key(state.time).Keys(state.x_columns).Key(state.y);
	if (!state.covariate.empty()) {
		rows.Key(state.covariate);
	}
	rows.Key(row_index).Carry(state.subject);
	rows.Sort();
	vector<idx_t> input_row(n);
	for (idx_t r = 0; r < n; r++) {
		input_row[r] = (idx_t)row_index[r];
	}
	return input_row;
}

//! The FFI view of a canonicalized state.
struct GlsCall {
	vector<double> x;
	AnofoxGlsData data {};
};

void PrepareGlsCall(GlsState &state, const GlsBindData &bind, GlsCall &call) {
	const idx_t n = state.y.size();
	call.x.reserve(n * state.n_features);
	for (auto &col : state.x_columns) {
		call.x.insert(call.x.end(), col.begin(), col.end());
	}
	call.data.y = state.y.data();
	call.data.x = call.x.empty() ? nullptr : call.x.data();
	call.data.subject = state.subject.data();
	call.data.time = state.time.data();
	call.data.variance_covariate = bind.UsesCovariate() ? state.covariate.data() : nullptr;
	call.data.n_rows = n;
	call.data.n_features = state.n_features;
}

void SetDoubleList(Vector &target, idx_t row, const double *values, idx_t len) {
	auto list_data = FlatDataMut<list_entry_t>(target);
	auto offset = ListVector::GetListSize(target);
	ListVector::Reserve(target, offset + len);
	auto child = FlatDataMut<double>(ListVector::GetEntry(target));
	for (idx_t j = 0; j < len; j++) {
		child[offset + j] = values[j];
	}
	ListVector::SetListSize(target, offset + len);
	list_data[row].offset = offset;
	list_data[row].length = len;
}

void SetOptionalDoubleList(Vector &target, idx_t row, const double *values, idx_t len, bool present) {
	if (!present) {
		FlatVector::SetNull(target, row, true);
		return;
	}
	SetDoubleList(target, row, values, len);
}

void SetDouble(Vector &target, idx_t row, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(target, row, true);
	} else {
		FlatDataMut<double>(target)[row] = v;
	}
}

LogicalType GlsVariogramType() {
	child_list_t<LogicalType> bin;
	bin.push_back(make_pair("dist", LogicalType::DOUBLE));
	bin.push_back(make_pair("variog", LogicalType::DOUBLE));
	bin.push_back(make_pair("n_pairs", LogicalType::BIGINT));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(bin)));
}

LogicalType GlsFitResultType() {
	auto list = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", list));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("correlation", LogicalType::VARCHAR));
	c.push_back(make_pair("variance", LogicalType::VARCHAR));
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("sigma", LogicalType::DOUBLE));
	// nlme coef(fit$modelStruct, unconstrained = FALSE): phi / rho / range, nugget /
	// the unstructured correlations r12, r13, ..., r23, ...
	c.push_back(make_pair("correlation_params", list));
	// varIdent: the SD ratio of each non-reference stratum; varPower / varExp: delta.
	c.push_back(make_pair("variance_params", list));
	c.push_back(make_pair("variance_strata", LogicalType::LIST(LogicalType::VARCHAR)));
	c.push_back(make_pair("variance_reference", LogicalType::VARCHAR));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_groups", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("n_params", LogicalType::BIGINT));
	c.push_back(make_pair("iterations", LogicalType::INTEGER));
	c.push_back(make_pair("converged", LogicalType::BOOLEAN));
	// Stable shape (#152): always present, NULL without compute_inference.
	AppendCoefficientInferenceFields(c, "t_values", false, false);
	// intervals(fit, which = "var-cov"): NULL without compute_inference, or when the
	// Hessian of the log-likelihood is not negative definite.
	c.push_back(make_pair("correlation_conf_low", list));
	c.push_back(make_pair("correlation_conf_high", list));
	c.push_back(make_pair("variance_conf_low", list));
	c.push_back(make_pair("variance_conf_high", list));
	c.push_back(make_pair("sigma_conf_low", LogicalType::DOUBLE));
	c.push_back(make_pair("sigma_conf_high", LogicalType::DOUBLE));
	// Empirical semivariogram (options.variogram = number of classes), else NULL.
	c.push_back(make_pair("variogram", GlsVariogramType()));
	AppendModelSummaryFields(c);
	AppendCovarianceFields(c);
	return LogicalType::STRUCT(std::move(c));
}

LogicalType GlsAugmentResultType() {
	child_list_t<LogicalType> row;
	row.push_back(make_pair("fitted", LogicalType::DOUBLE));
	row.push_back(make_pair("se_fit", LogicalType::DOUBLE));
	row.push_back(make_pair("residual", LogicalType::DOUBLE));
	row.push_back(make_pair("pearson_residual", LogicalType::DOUBLE));
	row.push_back(make_pair("normalized_residual", LogicalType::DOUBLE));
	row.push_back(make_pair("is_training", LogicalType::BOOLEAN));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(row)));
}

//! Field `name` of the result STRUCT.
struct FieldIndex {
	std::unordered_map<string, Vector *> fields;
	FieldIndex(Vector &result) {
		auto &entries = StructVector::GetEntries(result);
		auto &types = StructType::GetChildTypes(result.GetType());
		for (idx_t k = 0; k < types.size(); k++) {
			fields[NameString(types[k].first)] = &StructEntry(entries[k]);
		}
	}
	Vector &operator()(const char *name) const {
		return *fields.at(name);
	}
};

void WriteGlsFit(Vector &result, idx_t row, const GlsBindData &bind, const GlsState &state,
                 const AnofoxGlsFitResult &res) {
	FieldIndex f(result);
	ModelStructWriter writer(result);
	const bool has_int = res.has_intercept && res.n_coefficients > 0;
	// size_t, not idx_t: n_slopes initialises AnofoxFitResultInference::len (size_t; 4 bytes on wasm32).
	const size_t off = has_int ? 1 : 0;
	const size_t n_slopes = res.n_coefficients - off;

	SetDoubleList(f("coefficients"), row, res.coefficients + off, n_slopes);
	if (has_int) {
		FlatDataMut<double>(f("intercept"))[row] = res.coefficients[0];
	} else {
		FlatVector::SetNull(f("intercept"), row, true);
	}
	f("correlation").SetValue(row, Value(CORRELATION_NAMES[bind.options.correlation]));
	f("variance").SetValue(row, Value(VARIANCE_NAMES[bind.options.variance]));
	f("method").SetValue(row, Value(bind.options.reml ? "REML" : "ML"));
	FlatDataMut<double>(f("sigma"))[row] = res.sigma;
	SetDoubleList(f("correlation_params"), row, res.correlation_params, res.n_correlation_params);
	SetDoubleList(f("variance_params"), row, res.variance_params, res.n_variance_params);

	auto &strata = f("variance_strata");
	if (res.variance_strata) {
		auto list_data = FlatDataMut<list_entry_t>(strata);
		auto offset = ListVector::GetListSize(strata);
		ListVector::Reserve(strata, offset + res.n_variance_params);
		auto &child = ListVector::GetEntry(strata);
		auto label = [&](double code) {
			auto k = (idx_t)code;
			return k < state.strata.labels.size() ? state.strata.labels[k] : string();
		};
		for (idx_t j = 0; j < res.n_variance_params; j++) {
			child.SetValue(offset + j, Value(label(res.variance_strata[j + 1])));
		}
		ListVector::SetListSize(strata, offset + res.n_variance_params);
		list_data[row] = {offset, res.n_variance_params};
		f("variance_reference").SetValue(row, Value(label(res.variance_strata[0])));
	} else {
		FlatVector::SetNull(strata, row, true);
		FlatVector::SetNull(f("variance_reference"), row, true);
	}

	FlatDataMut<int64_t>(f("n_observations"))[row] = (int64_t)res.n_observations;
	FlatDataMut<int64_t>(f("n_groups"))[row] = (int64_t)res.n_groups;
	FlatDataMut<int64_t>(f("n_features"))[row] = (int64_t)state.n_features;
	FlatDataMut<int64_t>(f("n_params"))[row] = (int64_t)res.n_params;
	FlatDataMut<int32_t>(f("iterations"))[row] = (int32_t)res.iterations;
	FlatDataMut<bool>(f("converged"))[row] = res.converged;

	AnofoxModelSummaryFFI summary = res.summary;
	if (bind.compute_inference) {
		AnofoxFitResultInference view = {res.std_errors + off, res.t_values + off, res.p_values + off,
		                                 res.conf_low + off,   res.conf_high + off, n_slopes,
		                                 res.confidence_level, NAN,                NAN};
		writer.WriteInference(row, &view);
	} else {
		writer.WriteInference(row, nullptr);
		summary.intercept_std_error = NAN;
		summary.intercept_statistic = NAN;
		summary.intercept_p_value = NAN;
		summary.intercept_conf_low = NAN;
		summary.intercept_conf_high = NAN;
	}
	writer.WriteSummary(row, summary);

	const bool intervals = bind.compute_inference && res.has_intervals;
	SetOptionalDoubleList(f("correlation_conf_low"), row, res.correlation_conf_low, res.n_correlation_params,
	                      intervals);
	SetOptionalDoubleList(f("correlation_conf_high"), row, res.correlation_conf_high, res.n_correlation_params,
	                      intervals);
	SetOptionalDoubleList(f("variance_conf_low"), row, res.variance_conf_low, res.n_variance_params, intervals);
	SetOptionalDoubleList(f("variance_conf_high"), row, res.variance_conf_high, res.n_variance_params, intervals);
	SetDouble(f("sigma_conf_low"), row, intervals ? res.sigma_conf_low : NAN);
	SetDouble(f("sigma_conf_high"), row, intervals ? res.sigma_conf_high : NAN);

	auto &vg = f("variogram");
	if (bind.options.variogram_bins > 0 && res.n_variogram > 0) {
		auto list_data = FlatDataMut<list_entry_t>(vg);
		auto offset = ListVector::GetListSize(vg);
		ListVector::Reserve(vg, offset + res.n_variogram);
		auto &bins = StructVector::GetEntries(ListVector::GetEntry(vg));
		auto dist = FlatDataMut<double>(StructEntry(bins[0]));
		auto variog = FlatDataMut<double>(StructEntry(bins[1]));
		auto pairs = FlatDataMut<int64_t>(StructEntry(bins[2]));
		for (idx_t j = 0; j < res.n_variogram; j++) {
			dist[offset + j] = res.variogram_dist[j];
			variog[offset + j] = res.variogram_variog[j];
			pairs[offset + j] = (int64_t)res.variogram_n_pairs[j];
		}
		ListVector::SetListSize(vg, offset + res.n_variogram);
		list_data[row] = {offset, res.n_variogram};
	} else {
		FlatVector::SetNull(vg, row, true);
	}
	writer.WriteCovariance(row, res.vcov, res.n_coefficients, res.df_residual);
}

void GlsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                 idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<GlsBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlsState **)sdata.data;
	const char *fn = GlsModeName(bind.mode);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (!state.initialized || state.y.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto input_row = CanonicalizeGls(state);
		GlsCall call;
		PrepareGlsCall(state, bind, call);
		AnofoxError error;

		if (bind.mode == GlsMode::FIT) {
			AnofoxGlsFitResult res {};
			if (!anofox_gls_fit(&call.data, bind.options, &res, &error)) {
				ThrowUnlessDegenerate(fn, error);
				FlatVector::SetNull(result, row, true);
				continue;
			}
			WriteGlsFit(result, row, bind, state, res);
			anofox_free_gls_fit_result(&res);
			continue;
		}

		AnofoxGlsAugmentResult aug {};
		if (!anofox_gls_augment(&call.data, bind.options, &aug, &error)) {
			ThrowUnlessDegenerate(fn, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		const idx_t n = state.y.size();
		if (aug.n_rows != n) {
			anofox_free_gls_augment_result(&aug);
			throw InvalidInputException("%s: internal error: returned %llu rows for %llu inputs", fn,
			                            (unsigned long long)aug.n_rows, (unsigned long long)n);
		}
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n);
		ListVector::SetListSize(result, list_offset + n);
		ListVector::GetData(result)[row] = {list_offset, n};
		auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
		const double *columns[] = {aug.fitted, aug.se_fit, aug.residual, aug.pearson_residual,
		                           aug.normalized_residual};
		// Sorted row r is input (frame) row input_row[r].
		for (idx_t c = 0; c < 5; c++) {
			auto &vec = StructEntry(entries[c]);
			for (idx_t r = 0; r < n; r++) {
				SetDouble(vec, list_offset + input_row[r], columns[c][r]);
			}
		}
		auto training = FlatDataMut<bool>(StructEntry(entries[5]));
		for (idx_t r = 0; r < n; r++) {
			training[list_offset + input_row[r]] = aug.is_training[r];
		}
		anofox_free_gls_augment_result(&aug);
	}
}

//! Walk a MAP / STRUCT options value as (lower-case key, value) pairs.
template <class FN>
void VisitGlsOptions(const Value &options, FN &&fn) {
	const Value v = UnwrapVariant(options);
	if (v.IsNull()) {
		return;
	}
	if (v.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(v)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv.size() != 2 || kv[0].IsNull()) {
				throw InvalidInputException("Invalid MAP option entry");
			}
			fn(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else if (v.type().id() == LogicalTypeId::STRUCT) {
		auto &children = StructValue::GetChildren(v);
		auto &types = StructType::GetChildTypes(v.type());
		for (idx_t i = 0; i < types.size(); i++) {
			fn(StringUtil::Lower(NameString(types[i].first)), UnwrapVariant(children[i]));
		}
	} else {
		throw InvalidInputException("Expected MAP or STRUCT type for options, got %s", v.type().ToString());
	}
}

void ParseGlsOptions(const Value &value, GlsBindData &bind) {
	const string fn = GlsModeName(bind.mode);
	auto as_double = [&](const string &key, const Value &v) {
		try {
			return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
		}
	};
	auto as_bool = [&](const string &key, const Value &v) {
		try {
			return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a boolean, got '%s'", fn, key, v.ToString());
		}
	};
	auto as_count = [&](const string &key, const Value &v, double lo, double hi) {
		double d = as_double(key, v);
		if (!(d >= lo && d <= hi) || d != std::floor(d)) {
			throw InvalidInputException("%s: option '%s' must be an integer between %g and %g, got '%s'", fn, key,
			                            lo, hi, v.ToString());
		}
		return (uint32_t)d;
	};
	auto &o = bind.options;
	VisitGlsOptions(value, [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "correlation" || key == "cor") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "independent" || s == "none" || s == "identity") {
				o.correlation = 0;
			} else if (s == "compound_symmetry" || s == "compsymm" || s == "cs" || s == "exchangeable") {
				o.correlation = 1;
			} else if (s == "ar1") {
				o.correlation = 2;
			} else if (s == "car1" || s == "continuous_ar1") {
				o.correlation = 3;
			} else if (s == "exponential" || s == "exp") {
				o.correlation = 4;
			} else if (s == "gaussian" || s == "gaus") {
				o.correlation = 5;
			} else if (s == "linear" || s == "lin") {
				o.correlation = 6;
			} else if (s == "spherical" || s == "spher") {
				o.correlation = 7;
			} else if (s == "unstructured" || s == "symm" || s == "general") {
				o.correlation = 8;
			} else {
				throw InvalidInputException("%s: unknown correlation '%s'. Expected 'independent', "
				                            "'compound_symmetry', 'ar1', 'car1', 'exponential', 'gaussian', "
				                            "'linear', 'spherical' or 'unstructured'.",
				                            fn, v.ToString());
			}
		} else if (key == "nugget") {
			o.nugget = as_bool(key, v);
		} else if (key == "variance" || key == "weights") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "constant" || s == "none" || s == "identity") {
				o.variance = 0;
			} else if (s == "ident" || s == "varident") {
				o.variance = 1;
			} else if (s == "power" || s == "varpower") {
				o.variance = 2;
			} else if (s == "exp" || s == "exponential" || s == "varexp") {
				o.variance = 3;
			} else {
				throw InvalidInputException("%s: unknown variance '%s'. Expected 'constant', 'ident', 'power' or "
				                            "'exp'.",
				                            fn, v.ToString());
			}
		} else if (key == "method") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "reml") {
				o.reml = true;
			} else if (s == "ml") {
				o.reml = false;
			} else {
				throw InvalidInputException("%s: unknown method '%s'. Expected 'reml' or 'ml'.", fn, v.ToString());
			}
		} else if (key == "reml") {
			o.reml = as_bool(key, v);
		} else if (key == "fit_intercept" || key == "intercept") {
			o.fit_intercept = as_bool(key, v);
		} else if (key == "compute_inference") {
			bind.compute_inference = as_bool(key, v);
		} else if (key == "confidence_level") {
			double d = as_double(key, v);
			if (!(d > 0.0 && d < 1.0)) {
				throw InvalidInputException("%s: confidence_level must be in (0, 1), got %g", fn, d);
			}
			o.confidence_level = d;
		} else if (key == "max_iterations" || key == "max_iter") {
			o.max_iterations = as_count(key, v, 1, 1e6);
		} else if (key == "variogram") {
			o.variogram_bins = as_count(key, v, 0, 1e6);
		} else if (key == "variogram_residuals") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "pearson") {
				o.variogram_residuals = 0;
			} else if (s == "normalized") {
				o.variogram_residuals = 1;
			} else if (s == "response") {
				o.variogram_residuals = 2;
			} else {
				throw InvalidInputException("%s: unknown variogram_residuals '%s'. Expected 'pearson', "
				                            "'normalized' or 'response'.",
				                            fn, v.ToString());
			}
		} else {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: correlation, nugget, "
			                            "variance, method, reml, fit_intercept, compute_inference, "
			                            "confidence_level, max_iterations, variogram, variogram_residuals",
			                            fn, key);
		}
	});
	if (o.nugget && !(o.correlation >= 4 && o.correlation <= 7)) {
		throw InvalidInputException("%s: nugget applies to the spatial correlations (exponential, gaussian, linear, "
		                            "spherical), not '%s'",
		                            fn, CORRELATION_NAMES[o.correlation]);
	}
	if (o.variance != 0 && !bind.has_covariate) {
		throw InvalidInputException("%s: variance '%s' needs the variance_covariate argument: %s(y, x, subject, "
		                            "time, variance_covariate, options)",
		                            fn, VARIANCE_NAMES[o.variance], fn);
	}
}

template <GlsMode MODE>
unique_ptr<FunctionData> GlsBind(ClientContext &context, AggregateBindFunction &function,
                                 vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<GlsBindData>();
	bind->mode = MODE;
	bind->has_covariate = arguments.size() == 6;
	if (arguments.size() >= 5) {
		ParseGlsOptions(EvaluateConstantOptions(context, *arguments.back(), GlsModeName(MODE)), *bind);
	}
	if (bind->has_covariate && bind->UsesCovariate() &&
	    ReturnTypeOf(*arguments[4]).id() == LogicalTypeId::SQLNULL) {
		throw InvalidInputException("%s: variance '%s' needs a variance covariate, got NULL (gls_*_by: pass "
		                            "variance_col := <column>)",
		                            GlsModeName(MODE), VARIANCE_NAMES[bind->options.variance]);
	}
	SetReturnTypeOf(function, MODE == GlsMode::FIT ? GlsFitResultType() : GlsAugmentResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(GlsModeName(MODE));
	return std::move(bind);
}

void RegisterGlsAggregates(ExtensionLoader &loader, GlsMode mode) {
	const char *name = GlsModeName(mode);
	AggregateFunctionSet set(mode == GlsMode::FIT ? "gls_fit_agg" : "gls_augment_agg");
	const vector<LogicalType> base = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY,
	                                  LogicalType::DOUBLE};
	for (idx_t extra = 0; extra <= 2; extra++) {
		auto args = base;
		for (idx_t k = 0; k < extra; k++) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    name, args, LogicalType::ANY, AggregateFunction::StateSize<GlsState>, ANOFOX_AGG_INIT(GlsInitialize),
		    ANOFOX_GUARDED_UPDATE(GlsUpdate, GlsDestroy, GlsInitialize), GlsCombine,
		    ANOFOX_GUARDED_FINALIZE(GlsFinalize, GlsDestroy, GlsInitialize), nullptr,
		    mode == GlsMode::FIT ? ANOFOX_AGG_BIND(GlsBind<GlsMode::FIT>) : ANOFOX_AGG_BIND(GlsBind<GlsMode::AUGMENT>),
		    GlsDestroy));
	}
	CreateAggregateFunctionInfo info(set);
	const string what =
	    mode == GlsMode::FIT
	        ? "Fits a generalized least squares model (nlme::gls): a linear model whose errors are correlated within "
	          "each subject (options correlation: independent, compound_symmetry, ar1, car1, exponential, gaussian, "
	          "linear, spherical, unstructured; nugget) with an optional variance function of variance_covariate "
	          "(variance: ident, power, exp), estimated by REML (default) or ML (method). Returns the coefficients, "
	          "sigma, the correlation and variance parameters, the log-likelihood, AIC and BIC, and with "
	          "compute_inference the t tests and the var-cov intervals."
	        : "Window aggregate: fits the same GLS model as gls_fit_agg over the frame and returns one STRUCT per "
	          "input row (in frame order): fitted (x'b for every row with x), se_fit, residual, pearson_residual, "
	          "normalized_residual and is_training. Use it over a whole partition, as gls_augment_by does.";
	const string over = mode == GlsMode::FIT ? "" : " OVER ()";
	FunctionDescription d0;
	d0.description = what;
	d0.examples = {string(name) + "(y, [x1, x2], subject, time)" + over};
	d0.categories = {"regression", "mixed-models"};
	d0.parameter_names = {"y", "x", "subject", "time"};
	d0.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY,
	                      LogicalType::DOUBLE};
	info.descriptions.push_back(d0);
	FunctionDescription d1;
	d1.description = what;
	d1.examples = {string(name) + "(y, [x1, x2], subject, time, {'correlation': 'ar1'})" + over};
	d1.categories = {"regression", "mixed-models"};
	d1.parameter_names = {"y", "x", "subject", "time", "options"};
	d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY,
	                      LogicalType::DOUBLE, LogicalType::ANY};
	info.descriptions.push_back(d1);
	FunctionDescription d2 = d1;
	d2.examples = {string(name) + "(y, [x1], subject, time, arm, {'correlation': 'ar1', 'variance': 'ident'})" + over};
	d2.parameter_names = {"y", "x", "subject", "time", "variance_covariate", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY,
	                      LogicalType::DOUBLE, LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(d2);
	loader.RegisterFunction(info);
}

//===--------------------------------------------------------------------===//
// Table macros: gls_fit_by (one row per group with the model's fields),
// gls_tidy_by (terms), gls_glance_by (summary), gls_augment_by (obs).
// variance_col := <column> passes the variance covariate.
//===--------------------------------------------------------------------===//
const char *const GLS_CALL = "gls_fit_agg(y_col, x_cols, subject_col, time_col, variance_col, options)";

string GlsFitBySql() {
	return string("SELECT group_col, unnest(_m) FROM (SELECT group_col, ") + GLS_CALL +
	       " AS _m FROM query_table(source::VARCHAR) GROUP BY group_col) WHERE _m IS NOT NULL ORDER BY group_col";
}

const char *const GLS_INFERENCE_CALL =
    "gls_fit_agg(y_col, x_cols, subject_col, time_col, variance_col, map_concat(MAP {'compute_inference': 'true'}, "
    "coalesce(options::MAP(VARCHAR, VARCHAR), MAP {}::MAP(VARCHAR, VARCHAR))))";

string GlsTidySql() {
	const string fits = string("WITH _f AS (SELECT group_col, group_col::VARCHAR AS id, 'gls' AS model_id, 'gls' AS "
	                           "model_name, ") +
	                    GLS_INFERENCE_CALL +
	                    " AS _m FROM query_table(source::VARCHAR) GROUP BY group_col) ";
	const string no_index = "NULL::VARCHAR AS index_name, NULL::DOUBLE AS index_value, ";
	// The coefficients: tidy() of the fit.
	const string coef = "SELECT group_col, id, model_id, model_name, (_t).term AS term, (_t).estimate AS estimate, "
	                    "(_t).std_error AS std_error, (_t).statistic AS statistic, (_t).p_value AS p_value, "
	                    "(_t).conf_low AS conf_low, (_t).conf_high AS conf_high, (_t).conf_level AS conf_level, "
	                    "(_t).index_name AS index_name, (_t).index_value AS index_value, 'coefficient' AS effect, "
	                    "1 AS _sec, _i AS _pos FROM (SELECT *, unnest(tidy(_m, names)) AS _t, "
	                    "unnest(range(len(tidy(_m, names)))) AS _i FROM _f WHERE _m IS NOT NULL)";
	// The correlation and variance parameters with their var-cov intervals, then sigma.
	const string cor =
	    "SELECT group_col, id, model_id, model_name, CASE WHEN (_m).correlation = 'unstructured' THEN 'cor_' || _k "
	    "WHEN (_m).correlation IN ('exponential', 'gaussian', 'linear', 'spherical') THEN "
	    "['range', 'nugget'][_k] WHEN (_m).correlation = 'compound_symmetry' THEN 'rho' ELSE 'phi' END AS term, "
	    "(_m).correlation_params[_k] AS estimate, NULL::DOUBLE AS std_error, NULL::DOUBLE AS statistic, "
	    "NULL::DOUBLE AS p_value, (_m).correlation_conf_low[_k] AS conf_low, (_m).correlation_conf_high[_k] AS "
	    "conf_high, (_m).conf_level AS conf_level, " +
	    no_index +
	    "'correlation' AS effect, 2 AS _sec, _k AS _pos FROM (SELECT *, unnest(range(1, len((_m).correlation_params) + "
	    "1)) AS _k FROM _f WHERE _m IS NOT NULL)";
	const string var =
	    "SELECT group_col, id, model_id, model_name, CASE WHEN (_m).variance = 'ident' THEN 'sd_ratio_' || "
	    "(_m).variance_strata[_k] ELSE 'delta' END AS term, (_m).variance_params[_k] AS estimate, NULL::DOUBLE AS "
	    "std_error, NULL::DOUBLE AS statistic, NULL::DOUBLE AS p_value, (_m).variance_conf_low[_k] AS conf_low, "
	    "(_m).variance_conf_high[_k] AS conf_high, (_m).conf_level AS conf_level, " +
	    no_index +
	    "'variance' AS effect, 3 AS _sec, _k AS _pos FROM (SELECT *, unnest(range(1, len((_m).variance_params) + 1)) "
	    "AS _k FROM _f WHERE _m IS NOT NULL)";
	const string sigma = "SELECT group_col, id, model_id, model_name, 'sigma' AS term, (_m).sigma AS estimate, "
	                     "NULL::DOUBLE AS std_error, NULL::DOUBLE AS statistic, NULL::DOUBLE AS p_value, "
	                     "(_m).sigma_conf_low AS conf_low, (_m).sigma_conf_high AS conf_high, (_m).conf_level AS "
	                     "conf_level, " +
	                     no_index + "'sigma' AS effect, 4 AS _sec, 0 AS _pos FROM _f WHERE _m IS NOT NULL";
	return fits +
	       "SELECT group_col, id, model_id, model_name, term, estimate, std_error, statistic, p_value, conf_low, "
	       "conf_high, conf_level, index_name, index_value, effect FROM (" +
	       coef + " UNION ALL " + cor + " UNION ALL " + var + " UNION ALL " + sigma + ") ORDER BY group_col, _sec, _pos";
}

const char *const GLS_GLANCE_METRICS[] = {"n_observations", "n_groups", "n_features", "n_params", "sigma",
                                          "log_likelihood", "aic",      "bic",        "df_residual", "iterations"};

string GlsGlanceSql() {
	string metrics;
	for (auto metric : GLS_GLANCE_METRICS) {
		metrics += "{'metric': '" + string(metric) + "', 'value': (_m)." + metric + "::DOUBLE}, ";
	}
	metrics += "{'metric': 'converged', 'value': (_m).converged::DOUBLE}";
	return string("SELECT group_col, id, model_id, model_name, model_type, (_mv).metric AS metric, (_mv).value AS "
	              "value FROM (SELECT group_col, group_col::VARCHAR AS id, 'gls' AS model_id, 'gls' AS model_name, "
	              "(_m).model_type AS model_type, unnest([") +
	       metrics + "]) AS _mv FROM (SELECT group_col, " + GLS_CALL +
	       " AS _m FROM query_table(source::VARCHAR) GROUP BY group_col) WHERE _m IS NOT NULL) "
	       "WHERE (_mv).value IS NOT NULL ORDER BY group_col, metric";
}

const char *const GLS_AUGMENT_SQL =
    "SELECT * EXCLUDE (_aug, _rn, _anofox_rid), 'gls' AS model_id, 'gls' AS model_name, _anofox_rid AS row_id, "
    "(_aug[_rn]).fitted AS fitted, (_aug[_rn]).residual AS residual, 'response' AS resid_type, "
    "(_aug[_rn]).pearson_residual AS std_residual, NULL::DOUBLE AS stud_residual, NULL::DOUBLE AS leverage, "
    "NULL::DOUBLE AS cooks_d, NULL::DOUBLE AS dffits, 1.0::DOUBLE AS weight, (_aug[_rn]).is_training AS is_training, "
    "NULL::INTEGER AS n_params, (_aug[_rn]).se_fit AS se_fit, (_aug[_rn]).normalized_residual AS normalized_residual "
    "FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY _anofox_rid ROWS BETWEEN UNBOUNDED "
    "PRECEDING AND UNBOUNDED FOLLOWING) AS _rn, gls_augment_agg(y_col, x_cols, subject_col, time_col, variance_col, "
    "options) OVER (PARTITION BY group_col ORDER BY _anofox_rid ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED "
    "FOLLOWING) AS _aug FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))) "
    "ORDER BY group_col, row_id";

void RegisterGlsMacros(ExtensionLoader &loader) {
	const vector<string> params = {"source", "group_col", "y_col", "x_cols", "subject_col", "time_col"};
	const vector<string> categories = {"regression", "mixed-models", "table-macro"};
	const string args = "('visits', clinic, y, [dose], patient, week, options := {'correlation': 'ar1'})";

	auto fit = CreateContractTableMacro(
	    "gls_fit_by", params, {{"options", "NULL"}, {"variance_col", "NULL"}}, GlsFitBySql(),
	    "Table macro: fits one generalized least squares model (gls_fit_agg, errors correlated within subject_col, "
	    "ordered by time_col) per group_col and returns one row per group: the group column and every field of the "
	    "gls_fit_agg STRUCT. variance_col := <column> is the variance covariate of options.variance.",
	    "gls_fit_by" + args, categories);
	loader.RegisterFunction(*fit);

	auto tidy = CreateContractTableMacro(
	    "gls_tidy_by", params, {{"options", "NULL"}, {"names", "NULL"}, {"variance_col", "NULL"}}, GlsTidySql(),
	    "Table macro: fits one GLS model (gls_fit_agg) per group_col and returns its terms (the integration "
	    "contract's `terms` schema): the group column, id, model_id and model_name ('gls'), term, estimate, "
	    "std_error, statistic (t), p_value, conf_low, conf_high, conf_level, index_name, index_value, then effect: "
	    "'coefficient' rows (t tests on N - p df), 'correlation' rows (phi, rho, range / nugget or cor_<k>), "
	    "'variance' rows (sd_ratio_<stratum> or delta) and the 'sigma' row, with the approximate var-cov intervals "
	    "(nlme intervals()). Inference is always computed. names := [...] labels the coefficients.",
	    "gls_tidy_by" + args, categories);
	loader.RegisterFunction(*tidy);

	auto glance = CreateContractTableMacro(
	    "gls_glance_by", params, {{"options", "NULL"}, {"variance_col", "NULL"}}, GlsGlanceSql(),
	    "Table macro: fits one GLS model (gls_fit_agg) per group_col and returns its fit statistics in long form "
	    "(the integration contract's `summary` schema): the group column, id, model_id and model_name ('gls'), "
	    "model_type, metric, value. Metrics: n_observations, n_groups (subjects), n_features, n_params, sigma, "
	    "log_likelihood (REML or ML), aic, bic, df_residual, iterations, converged (1/0).",
	    "gls_glance_by" + args, categories);
	loader.RegisterFunction(*glance);

	auto augment = CreateContractTableMacro(
	    "gls_augment_by", params, {{"options", "NULL"}, {"variance_col", "NULL"}}, GLS_AUGMENT_SQL,
	    "Table macro: fits one GLS model (gls_augment_agg) per group_col and returns every source row with its "
	    "per-observation results (the integration contract's `obs` schema): all source columns, model_id and "
	    "model_name ('gls'), row_id, fitted (x'b), residual (y - fitted), resid_type ('response'), std_residual (the "
	    "Pearson residual), stud_residual, leverage, cooks_d, dffits (NULL), weight, is_training, n_params (NULL), "
	    "then se_fit and normalized_residual. Rows the fit could not use get NULL residuals; rows with x get "
	    "fitted and se_fit.",
	    "gls_augment_by" + args, categories);
	loader.RegisterFunction(*augment);
}

} // namespace

void RegisterGlsFunctions(ExtensionLoader &loader) {
	RegisterGlsAggregates(loader, GlsMode::FIT);
	RegisterGlsAggregates(loader, GlsMode::AUGMENT);
	RegisterGlsMacros(loader);
}

} // namespace duckdb
