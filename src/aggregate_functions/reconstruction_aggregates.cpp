// Model monitoring without labels, from upstream anofox-statistics 0.10
// `monitoring` (validated there against NannyML 0.13.1):
//   #366 multivariate drift by PCA reconstruction error (NannyML's
//   DataReconstructionDriftCalculator):
//     pca_reconstruction_fit_agg(x DOUBLE[] [, options]) -- fit_reconstruction,
//       the model as a STRUCT (rebuilt with ReconstructionModel::from_parts);
//     pca_reconstruction_error(model, x DOUBLE[]) -- one row's error (row_errors);
//     pca_reconstruction_chunk_agg(model, x DOUBLE[]) -- a chunk's mean error,
//       sampling error and confidence band (chunk_error);
//     pca_reconstruction_threshold_agg(value [, threshold]) -- the alert band
//       from the reference chunk values (reconstruction_thresholds);
//     pca_reconstruction_drift_by(source, chunk_col, x_cols, is_reference, ...).
//   #367 reference-band alert thresholds of any chunked metric:
//     alert_band_agg(value [, options]) -- threshold_band;
//     alert_band_by(source, chunk_col, metric_col, is_reference [, options]).
// A NULL element of x is a missing value (NaN): imputed by the reference column
// means unless 'impute_nan' is false. This file only buffers rows and marshals
// options and the model STRUCT.

#include <algorithm>
#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/reconstruction_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/canonical_order.hpp"
#include "anofox_statistics_banner.hpp"
#include "anofox_statistics_extension.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Options
//===--------------------------------------------------------------------===//

//! Calls f(lower-case key, value) for each entry of a MAP / STRUCT options value.
template <class F>
static void ForEachOption(const Value &options, const char *fn, F &&f) {
	if (options.IsNull()) {
		return;
	}
	auto unwrapped = UnwrapVariant(options);
	if (unwrapped.type().id() == LogicalTypeId::STRUCT) {
		auto &vals = StructValue::GetChildren(unwrapped);
		for (idx_t i = 0; i < vals.size(); i++) {
			f(StringUtil::Lower(NameString(StructType::GetChildName(unwrapped.type(), i))), UnwrapVariant(vals[i]));
		}
	} else if (unwrapped.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(unwrapped)) {
			auto &kv = StructValue::GetChildren(entry);
			f(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else {
		throw InvalidInputException("%s: options must be a MAP or STRUCT", fn);
	}
}

static double OptionDouble(const Value &v, const char *fn, const string &key) {
	if (v.IsNull()) {
		throw InvalidInputException("%s: option '%s' must be a number, not NULL", fn, key);
	}
	return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
}

static bool OptionBool(const Value &v, const char *fn, const string &key) {
	if (v.IsNull()) {
		throw InvalidInputException("%s: option '%s' must be true or false, not NULL", fn, key);
	}
	return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
}

//! A band's method and limits (NaN: no threshold / no limit on that side).
struct BandOptions {
	AnofoxThresholdMethod method {0, 3.0, 3.0};
	AnofoxValueLimits limits {std::nan(""), std::nan(""), false};

	bool operator==(const BandOptions &o) const {
		auto same = [](double a, double b) {
			return (std::isnan(a) && std::isnan(b)) || a == b;
		};
		return method.method == o.method.method && same(method.lower, o.method.lower) &&
		       same(method.upper, o.method.upper) && same(limits.lower, o.limits.lower) &&
		       same(limits.upper, o.limits.upper) && limits.override_with_none == o.limits.override_with_none;
	}
};

//! 'method' ('std' | 'constant' (alias 'fixed') | 'relative'), 'k' (both std /
//! relative multipliers), 'lower' / 'upper' (one side; NULL disables it) and,
//! with allow_limits, 'lower_limit' / 'upper_limit' / 'override_with_none'.
static BandOptions ParseBandOptions(const Value &options, const char *fn, bool allow_limits) {
	BandOptions o;
	string method = "std";
	bool has_k = false;
	double k = 3.0;
	bool has_lower = false, has_upper = false;
	double lower = std::nan(""), upper = std::nan("");
	ForEachOption(options, fn, [&](const string &key, const Value &v) {
		if (key == "method") {
			if (v.IsNull()) {
				throw InvalidInputException("%s: option 'method' must be 'std', 'constant' or 'relative'", fn);
			}
			method = StringUtil::Lower(v.ToString());
		} else if (key == "k") {
			has_k = true;
			k = OptionDouble(v, fn, key);
		} else if (key == "lower") {
			has_lower = true;
			lower = v.IsNull() ? std::nan("") : v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} else if (key == "upper") {
			has_upper = true;
			upper = v.IsNull() ? std::nan("") : v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} else if (allow_limits && key == "lower_limit") {
			o.limits.lower = v.IsNull() ? std::nan("") : OptionDouble(v, fn, key);
		} else if (allow_limits && key == "upper_limit") {
			o.limits.upper = v.IsNull() ? std::nan("") : OptionDouble(v, fn, key);
		} else if (allow_limits && key == "override_with_none") {
			o.limits.override_with_none = OptionBool(v, fn, key);
		} else {
			throw InvalidInputException("%s: unknown option '%s' (valid: method, k, lower, upper%s)", fn, key,
			                            allow_limits ? ", lower_limit, upper_limit, override_with_none" : "");
		}
	});
	if (method == "std" || method == "standard_deviation") {
		o.method.method = 0;
	} else if (method == "constant" || method == "fixed") {
		o.method.method = 1;
	} else if (method == "relative") {
		o.method.method = 2;
	} else {
		throw InvalidInputException("%s: option 'method' must be 'std', 'constant' or 'relative', got '%s'", fn,
		                            method);
	}
	if (has_k && o.method.method == 1) {
		throw InvalidInputException("%s: option 'k' applies to the 'std' and 'relative' methods; a constant band "
		                            "takes 'lower' and 'upper'",
		                            fn);
	}
	// std defaults to NannyML's 3 sd on both sides; constant and relative have no default side.
	double side = o.method.method == 0 || has_k ? k : std::nan("");
	o.method.lower = has_lower ? lower : side;
	o.method.upper = has_upper ? upper : side;
	return o;
}

//===--------------------------------------------------------------------===//
// alert_band_agg / pca_reconstruction_threshold_agg (value DOUBLE [, options])
//===--------------------------------------------------------------------===//
static constexpr int BAND_GENERIC = 0;        // alert_band_agg: threshold_band
static constexpr int BAND_RECONSTRUCTION = 1; // pca_reconstruction_threshold_agg

struct BandAggregateState {
	vector<double> values;
	bool initialized;

	BandAggregateState() : initialized(false) {
	}
};

struct BandBindData : public FunctionData {
	int kind = BAND_GENERIC;
	BandOptions options;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<BandBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<BandBindData>();
		return kind == other.kind && options == other.options;
	}
};

static const char *BandName(int kind) {
	return kind == BAND_GENERIC ? "alert_band_agg" : "pca_reconstruction_threshold_agg";
}

static bool CallBand(int kind, const BandOptions &o, const double *values, idx_t n, AnofoxThresholdBand &out,
                     AnofoxError &error) {
	if (kind == BAND_GENERIC) {
		return anofox_threshold_band(values, static_cast<size_t>(n), o.method, o.limits, &out, &error);
	}
	return anofox_reconstruction_thresholds(values, static_cast<size_t>(n), o.method, &out, &error);
}

static LogicalType BandResultType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("lower_threshold", LogicalType::DOUBLE));
	c.push_back(MakeChild("upper_threshold", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

static void SetDoubleOrNull(Vector &v, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(v, row, true);
	} else {
		FlatDataMut<double>(v)[row] = value;
	}
}

static void BandInitialize(data_ptr_t state_p) {
	new (state_p) BandAggregateState();
}

static void BandDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (BandAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~BandAggregateState();
	}
}

static void BandUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat vdata, sdata;
	inputs[0].ToUnifiedFormat(count, vdata);
	state_vector.ToUnifiedFormat(count, sdata);
	auto values = UnifiedVectorFormat::GetData<double>(vdata);
	auto states = (BandAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;
		auto idx = vdata.sel->get_index(i);
		if (vdata.validity.RowIsValid(idx)) {
			state.values.push_back(values[idx]);
		}
	}
}

static void BandCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                        idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (BandAggregateState **)source_data.data;
	auto targets = (BandAggregateState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.values = CombineTake(source.values, aggr_input_data);
			target.initialized = true;
			continue;
		}
		target.values.insert(target.values.end(), source.values.begin(), source.values.end());
	}
}

static void BandFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<BandBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (BandAggregateState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		// Canonical order: the mean and sd are bitwise identical whatever the thread count.
		std::sort(state.values.begin(), state.values.end(),
		          [](double a, double b) { return CompareDoubleTotal(a, b) < 0; });
		AnofoxThresholdBand band;
		AnofoxError error;
		if (!CallBand(bind.kind, bind.options, state.values.data(), state.values.size(), band, error)) {
			// No reference values (std / relative): NULL.
			ThrowUnlessDegenerate(BandName(bind.kind), error);
			FlatVector::SetNull(result, row, true);
			state.values.clear();
			continue;
		}
		SetDoubleOrNull(StructEntry(entries[0]), row, band.lower);
		SetDoubleOrNull(StructEntry(entries[1]), row, band.upper);
		state.values.clear();
	}
}

template <int KIND>
static unique_ptr<FunctionData> BandBind(ClientContext &context, AggregateBindFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<BandBindData>();
	bind->kind = KIND;
	const char *fn = BandName(KIND);
	Value options;
	if (arguments.size() >= 2) {
		options = EvaluateConstantOptions(context, *arguments[1], fn);
	}
	bind->options = ParseBandOptions(options, fn, KIND == BAND_GENERIC);
	// Reject invalid multipliers or a constant band with lower >= upper now: upstream
	// validates the method before it looks at the (here a single dummy) reference value.
	double dummy = 0.0;
	AnofoxThresholdBand band;
	AnofoxError error;
	if (!CallBand(KIND, bind->options, &dummy, 1, band, error)) {
		ThrowFromFfiError(fn, error);
	}
	SetReturnTypeOf(function, BandResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// The model STRUCT of pca_reconstruction_fit_agg
//===--------------------------------------------------------------------===//
static LogicalType ReconstructionModelType() {
	auto dl = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("center", dl));
	// Population column sds the rows are divided by; NULL for an unscaled model.
	c.push_back(MakeChild("scale", dl));
	c.push_back(MakeChild("sdev", dl));
	c.push_back(MakeChild("proportion_of_variance", dl));
	// One list of n_features loadings per kept component.
	c.push_back(MakeChild("loadings", LogicalType::LIST(dl)));
	c.push_back(MakeChild("n_components", LogicalType::BIGINT));
	c.push_back(MakeChild("n_features", LogicalType::BIGINT));
	c.push_back(MakeChild("n_observations", LogicalType::BIGINT));
	c.push_back(MakeChild("impute_nan", LogicalType::BOOLEAN));
	return LogicalType::STRUCT(std::move(c));
}

//! A model read from its STRUCT. Relocatable: vectors only.
struct ReconstructionModelParts {
	vector<double> center, scale, sdev, proportion_of_variance, loadings; // loadings component-major
	idx_t n_features = 0;
	idx_t n_components = 0;
	bool has_scale = false;
	bool impute_nan = true;

	AnofoxReconstructionModel View() const {
		AnofoxReconstructionModel m;
		m.center = const_cast<double *>(center.data());
		m.scale = has_scale ? const_cast<double *>(scale.data()) : nullptr;
		m.sdev = const_cast<double *>(sdev.data());
		m.proportion_of_variance = const_cast<double *>(proportion_of_variance.data());
		m.loadings = const_cast<double *>(loadings.data());
		m.n_features = static_cast<size_t>(n_features);
		m.n_components = static_cast<size_t>(n_components);
		m.n_observations = 0;
		m.impute_nan = impute_nan;
		return m;
	}
};

static bool ReadDoubles(const Value &v, vector<double> &out) {
	out.clear();
	auto list = v.DefaultCastAs(LogicalType::LIST(LogicalType::DOUBLE));
	for (auto &e : ListValue::GetChildren(list)) {
		if (e.IsNull()) {
			return false;
		}
		out.push_back(e.GetValue<double>());
	}
	return true;
}

//! Reads a model STRUCT; an empty string on success, else the error message.
static string ReadReconstructionModel(const Value &model, const char *fn, ReconstructionModelParts &out) {
	out = ReconstructionModelParts();
	auto bad = [&](const char *what) {
		return StringUtil::Format("%s: invalid model (%s); pass the STRUCT returned by pca_reconstruction_fit_agg", fn,
		                          what);
	};
	if (model.type().id() != LogicalTypeId::STRUCT) {
		return bad("not a STRUCT");
	}
	bool has_center = false, has_sdev = false, has_loadings = false, has_k = false;
	int64_t k_field = 0;
	vector<vector<double>> loadings;
	auto &children = StructValue::GetChildren(model);
	for (idx_t i = 0; i < children.size(); i++) {
		auto name = StringUtil::Lower(NameString(StructType::GetChildName(model.type(), i)));
		auto &v = children[i];
		if (name == "center") {
			if (v.IsNull() || !ReadDoubles(v, out.center)) {
				return bad("center must be a list of non-NULL values");
			}
			has_center = true;
		} else if (name == "scale") {
			if (!v.IsNull()) {
				if (!ReadDoubles(v, out.scale)) {
					return bad("scale must not hold NULLs");
				}
				out.has_scale = true;
			}
		} else if (name == "sdev") {
			if (v.IsNull() || !ReadDoubles(v, out.sdev)) {
				return bad("sdev must be a list of non-NULL values");
			}
			has_sdev = true;
		} else if (name == "proportion_of_variance") {
			if (!v.IsNull() && !ReadDoubles(v, out.proportion_of_variance)) {
				return bad("proportion_of_variance must not hold NULLs");
			}
		} else if (name == "loadings") {
			if (v.IsNull()) {
				return bad("loadings is NULL");
			}
			auto list = v.DefaultCastAs(LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)));
			for (auto &component : ListValue::GetChildren(list)) {
				vector<double> l;
				if (component.IsNull() || !ReadDoubles(component, l)) {
					return bad("loadings must not hold NULLs");
				}
				loadings.push_back(std::move(l));
			}
			has_loadings = true;
		} else if (name == "n_components") {
			if (!v.IsNull()) {
				k_field = v.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
				has_k = true;
			}
		} else if (name == "impute_nan") {
			if (!v.IsNull()) {
				out.impute_nan = v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
			}
		}
	}
	if (!has_center || !has_sdev || !has_loadings) {
		return bad("it needs the fields center, sdev and loadings");
	}
	out.n_features = out.center.size();
	out.n_components = out.sdev.size();
	if (has_k && k_field != static_cast<int64_t>(out.n_components)) {
		return bad("n_components must equal the number of sdev values");
	}
	if (loadings.size() != out.n_components) {
		return bad("there must be one loading vector per component");
	}
	if (out.has_scale && out.scale.size() != out.n_features) {
		return bad("scale must hold one value per feature");
	}
	if (out.proportion_of_variance.size() != out.n_components) {
		out.proportion_of_variance.assign(out.n_components, std::nan(""));
	}
	for (auto &l : loadings) {
		if (l.size() != out.n_features) {
			return bad("every loading vector must hold one value per feature");
		}
		out.loadings.insert(out.loadings.end(), l.begin(), l.end());
	}
	return string();
}

//! Checks at bind time that the model argument is a STRUCT with the model's fields.
static void CheckModelType(const LogicalType &type, const char *fn) {
	bool ok = type.id() == LogicalTypeId::STRUCT;
	if (ok) {
		bool c = false, s = false, l = false;
		for (auto &child : StructType::GetChildTypes(type)) {
			auto name = StringUtil::Lower(NameString(child.first));
			c = c || name == "center";
			s = s || name == "sdev";
			l = l || name == "loadings";
		}
		ok = c && s && l;
	}
	if (!ok) {
		throw InvalidInputException("%s(model, x): model must be the STRUCT returned by pca_reconstruction_fit_agg, "
		                            "got %s",
		                            fn, type.ToString());
	}
}

//! Appends a list row (NULL elements: NaN, a missing value) to column-major buffers.
static void AppendRow(vector<vector<double>> &columns, const double *data, const ValidityMask &validity,
                      const list_entry_t &entry) {
	for (idx_t j = 0; j < entry.length; j++) {
		auto idx = entry.offset + j;
		columns[j].push_back(validity.RowIsValid(idx) ? data[idx] : std::nan(""));
	}
}

static vector<double> RowMajor(const vector<vector<double>> &columns) {
	idx_t d = columns.size();
	idx_t n = d == 0 ? 0 : columns[0].size();
	vector<double> data(n * d);
	for (idx_t r = 0; r < n; r++) {
		for (idx_t j = 0; j < d; j++) {
			data[r * d + j] = columns[j][r];
		}
	}
	return data;
}

static void AppendDoubleList(Vector &list_vec, idx_t row, const double *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	auto child_data = FlatDataMut<double>(ListVector::GetEntry(list_vec));
	for (idx_t i = 0; i < len; i++) {
		child_data[offset + i] = data[i];
	}
	ListVector::SetListSize(list_vec, offset + len);
	ListVector::GetData(list_vec)[row] = {offset, len};
}

//===--------------------------------------------------------------------===//
// pca_reconstruction_fit_agg(x DOUBLE[] [, options])
//===--------------------------------------------------------------------===//
struct ReconstructionFitState {
	vector<vector<double>> x_columns;
	idx_t n_features;
	bool initialized;
	//! A data error found in Update / Combine, thrown in Finalize (a vector, not a
	//! string: DuckDB relocates states with memcpy). Holds zero or one message.
	vector<string> error;

	ReconstructionFitState() : n_features(0), initialized(false) {
	}

	void Reset() {
		x_columns.clear();
		n_features = 0;
		initialized = false;
		error.clear();
	}
};

struct ReconstructionFitBindData : public FunctionData {
	//! Share of the variance the kept components explain (NannyML's n_components = 0.65).
	double variance = 0.65;
	//! A fixed number of components; 0: by variance.
	idx_t n_components = 0;
	bool scale = true;
	bool impute_nan = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ReconstructionFitBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ReconstructionFitBindData>();
		return variance == o.variance && n_components == o.n_components && scale == o.scale &&
		       impute_nan == o.impute_nan;
	}
};

static constexpr const char *FIT_NAME = "pca_reconstruction_fit_agg";

static void FitInitialize(data_ptr_t state_p) {
	new (state_p) ReconstructionFitState();
}

static void FitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ReconstructionFitState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~ReconstructionFitState();
	}
}

static string FeatureCountError(const char *fn, idx_t expected, idx_t got) {
	return StringUtil::Format("%s: every x must have the same length, expected %llu, got %llu", fn,
	                          static_cast<unsigned long long>(expected), static_cast<unsigned long long>(got));
}

static void FitUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat x_data, sdata;
	inputs[0].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child = ListVector::GetEntry(inputs[0]);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	auto states = (ReconstructionFitState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(idx) || !state.error.empty()) {
			continue; // NULL row
		}
		auto entry = lists[idx];
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(entry.length);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			state.error = {FeatureCountError(FIT_NAME, state.n_features, entry.length)};
			continue;
		}
		AppendRow(state.x_columns, child_data, child_validity, entry);
	}
}

static void FitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (ReconstructionFitState **)source_data.data;
	auto targets = (ReconstructionFitState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.error.empty() && target.error.empty()) {
			target.error = source.error;
		}
		if (!source.initialized || !target.error.empty()) {
			continue;
		}
		if (!target.initialized) {
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			target.error = {FeatureCountError(FIT_NAME, target.n_features, source.n_features)};
			continue;
		}
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

static void FitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ReconstructionFitBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ReconstructionFitState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.error.empty()) {
			throw InvalidInputException(state.error[0]);
		}
		if (!state.initialized || state.x_columns.empty()) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Keys(state.x_columns).Sort();
		idx_t n = state.x_columns[0].size();
		auto data = RowMajor(state.x_columns);
		AnofoxReconstructionModel fit;
		AnofoxError error;
		if (!anofox_reconstruction_fit(data.data(), static_cast<size_t>(n), static_cast<size_t>(state.n_features),
		                               bind.variance, static_cast<size_t>(bind.n_components), bind.scale,
		                               bind.impute_nan, &fit, &error)) {
			ThrowUnlessDegenerate(FIT_NAME, error);
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		idx_t k = fit.n_components;
		idx_t d = fit.n_features;
		AppendDoubleList(StructEntry(entries[0]), row, fit.center, d);
		if (fit.scale) {
			AppendDoubleList(StructEntry(entries[1]), row, fit.scale, d);
		} else {
			FlatVector::SetNull(StructEntry(entries[1]), row, true);
		}
		AppendDoubleList(StructEntry(entries[2]), row, fit.sdev, k);
		AppendDoubleList(StructEntry(entries[3]), row, fit.proportion_of_variance, k);
		auto &loadings = StructEntry(entries[4]);
		auto outer_offset = ListVector::GetListSize(loadings);
		ListVector::Reserve(loadings, outer_offset + k);
		auto &inner = ListVector::GetEntry(loadings);
		for (idx_t j = 0; j < k; j++) {
			AppendDoubleList(inner, outer_offset + j, fit.loadings + j * d, d);
		}
		ListVector::SetListSize(loadings, outer_offset + k);
		ListVector::GetData(loadings)[row] = {outer_offset, k};
		FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = static_cast<int64_t>(k);
		FlatDataMut<int64_t>(StructEntry(entries[6]))[row] = static_cast<int64_t>(d);
		FlatDataMut<int64_t>(StructEntry(entries[7]))[row] = static_cast<int64_t>(fit.n_observations);
		FlatDataMut<bool>(StructEntry(entries[8]))[row] = fit.impute_nan;
		anofox_free_reconstruction_model(&fit);
		state.Reset();
	}
}

static unique_ptr<FunctionData> FitBind(ClientContext &context, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<ReconstructionFitBindData>();
	if (arguments.size() >= 2) {
		Value options = EvaluateConstantOptions(context, *arguments[1], FIT_NAME);
		bool has_variance = false;
		ForEachOption(options, FIT_NAME, [&](const string &key, const Value &v) {
			if (key == "variance") {
				bind->variance = OptionDouble(v, FIT_NAME, key);
				has_variance = true;
			} else if (key == "n_components") {
				auto k = OptionDouble(v, FIT_NAME, key);
				if (!(k >= 1) || k != std::floor(k)) {
					throw InvalidInputException("%s: option 'n_components' must be a whole number of at least 1 "
					                            "(use 'variance' for a share of the variance)",
					                            FIT_NAME);
				}
				bind->n_components = static_cast<idx_t>(k);
			} else if (key == "scale") {
				bind->scale = OptionBool(v, FIT_NAME, key);
			} else if (key == "impute_nan") {
				bind->impute_nan = OptionBool(v, FIT_NAME, key);
			} else {
				throw InvalidInputException("%s: unknown option '%s' (valid: variance, n_components, scale, "
				                            "impute_nan)",
				                            FIT_NAME, key);
			}
		});
		if (has_variance && bind->n_components > 0) {
			throw InvalidInputException("%s: give either 'variance' or 'n_components', not both", FIT_NAME);
		}
		if (!(bind->variance > 0 && bind->variance < 1)) {
			throw InvalidInputException("%s: option 'variance' must be in (0, 1), got %s", FIT_NAME,
			                            std::to_string(bind->variance));
		}
	}
	SetReturnTypeOf(function, ReconstructionModelType());
	PostHogTelemetry::Instance().RecordFunctionCall(FIT_NAME);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// pca_reconstruction_chunk_agg(model, x DOUBLE[])
//===--------------------------------------------------------------------===//
static constexpr const char *CHUNK_NAME = "pca_reconstruction_chunk_agg";

struct ReconstructionChunkState {
	ReconstructionModelParts model;
	bool has_model;
	vector<vector<double>> x_columns;
	bool initialized;
	vector<string> error; // as ReconstructionFitState

	ReconstructionChunkState() : has_model(false), initialized(false) {
	}

	void Reset() {
		model = ReconstructionModelParts();
		has_model = false;
		x_columns.clear();
		initialized = false;
		error.clear();
	}
};

static LogicalType ChunkResultType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("n", LogicalType::BIGINT));
	c.push_back(MakeChild("reconstruction_error", LogicalType::DOUBLE));
	c.push_back(MakeChild("std", LogicalType::DOUBLE));
	c.push_back(MakeChild("sampling_error", LogicalType::DOUBLE));
	c.push_back(MakeChild("lower_confidence", LogicalType::DOUBLE));
	c.push_back(MakeChild("upper_confidence", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

static void ChunkInitialize(data_ptr_t state_p) {
	new (state_p) ReconstructionChunkState();
}

static void ChunkDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ReconstructionChunkState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~ReconstructionChunkState();
	}
}

static void ChunkUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat model_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, model_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child = ListVector::GetEntry(inputs[1]);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	auto states = (ReconstructionChunkState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;
		auto x_idx = x_data.sel->get_index(i);
		if (!model_data.validity.RowIsValid(model_data.sel->get_index(i)) || !x_data.validity.RowIsValid(x_idx) ||
		    !state.error.empty()) {
			continue;
		}
		if (!state.has_model) {
			// The model is the same on every row of a chunk (a scalar subquery or a join);
			// it is read once per state.
			auto message = ReadReconstructionModel(inputs[0].GetValue(i), CHUNK_NAME, state.model);
			if (!message.empty()) {
				state.error = {message};
				continue;
			}
			state.has_model = true;
			state.x_columns.resize(state.model.n_features);
		}
		auto entry = lists[x_idx];
		if (entry.length != state.model.n_features) {
			state.error = {StringUtil::Format("%s: the model has %llu features but x has %llu values", CHUNK_NAME,
			                                  static_cast<unsigned long long>(state.model.n_features),
			                                  static_cast<unsigned long long>(entry.length))};
			continue;
		}
		AppendRow(state.x_columns, child_data, child_validity, entry);
	}
}

static void ChunkCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                         idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (ReconstructionChunkState **)source_data.data;
	auto targets = (ReconstructionChunkState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		target.initialized = true;
		if (!source.error.empty() && target.error.empty()) {
			target.error = source.error;
		}
		if (!target.error.empty() || !source.has_model) {
			continue;
		}
		if (!target.has_model) {
			target.model = CombineTake(source.model, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.has_model = true;
			continue;
		}
		if (source.model.n_features != target.model.n_features) {
			target.error = {StringUtil::Format("%s: the model must be the same on every row of a group", CHUNK_NAME)};
			continue;
		}
		for (idx_t j = 0; j < target.x_columns.size(); j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

static void ChunkFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ReconstructionChunkState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.error.empty()) {
			throw InvalidInputException(state.error[0]);
		}
		if (!state.has_model || state.x_columns.empty() || state.x_columns[0].empty()) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		CanonicalRows().Keys(state.x_columns).Sort();
		idx_t n = state.x_columns[0].size();
		auto data = RowMajor(state.x_columns);
		auto model = state.model.View();
		AnofoxReconstructionChunk chunk;
		AnofoxError error;
		if (!anofox_reconstruction_chunk(&model, data.data(), static_cast<size_t>(n), &chunk, &error)) {
			ThrowUnlessDegenerate(CHUNK_NAME, error);
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(entries[0]))[row] = static_cast<int64_t>(chunk.n);
		SetDoubleOrNull(StructEntry(entries[1]), row, chunk.value);
		SetDoubleOrNull(StructEntry(entries[2]), row, chunk.std);
		SetDoubleOrNull(StructEntry(entries[3]), row, chunk.sampling_error);
		SetDoubleOrNull(StructEntry(entries[4]), row, chunk.lower_confidence);
		SetDoubleOrNull(StructEntry(entries[5]), row, chunk.upper_confidence);
		state.Reset();
	}
}

static unique_ptr<FunctionData> ChunkBind(ClientContext &, AggregateBindFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	CheckModelType(ReturnTypeOf(*arguments[0]), CHUNK_NAME);
	SetReturnTypeOf(function, ChunkResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(CHUNK_NAME);
	return nullptr;
}

//===--------------------------------------------------------------------===//
// pca_reconstruction_error(model, x DOUBLE[]) -> DOUBLE
//===--------------------------------------------------------------------===//
static constexpr const char *ERROR_NAME = "pca_reconstruction_error";

static unique_ptr<FunctionData> ErrorBind(ClientContext &, ScalarBindFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	CheckModelType(ReturnTypeOf(*arguments[0]), ERROR_NAME);
	SetReturnTypeOf(bound_function, LogicalType::DOUBLE);
	PostHogTelemetry::Instance().RecordFunctionCall(ERROR_NAME);
	return nullptr;
}

static void ErrorFunctionImpl(DataChunk &args, ExpressionState &, Vector &result) {
	idx_t count = args.size();
	auto &model_vec = args.data[0];
	const bool constant_model = model_vec.GetVectorType() == VectorType::CONSTANT_VECTOR;
	UnifiedVectorFormat x_data;
	args.data[1].ToUnifiedFormat(count, x_data);
	auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child = ListVector::GetEntry(args.data[1]);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatDataMut<double>(result);

	ReconstructionModelParts model;
	bool have_model = false;
	vector<double> row_data;
	for (idx_t row = 0; row < count; row++) {
		auto x_idx = x_data.sel->get_index(row);
		// A constant model (the usual case: a scalar subquery or a literal) is read once.
		if (!have_model || !constant_model) {
			auto mv = model_vec.GetValue(constant_model ? 0 : row);
			if (mv.IsNull()) {
				FlatVector::SetNull(result, row, true);
				continue;
			}
			auto message = ReadReconstructionModel(mv, ERROR_NAME, model);
			if (!message.empty()) {
				throw InvalidInputException(message);
			}
			have_model = true;
		}
		if (!x_data.validity.RowIsValid(x_idx)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto entry = lists[x_idx];
		if (entry.length != model.n_features) {
			throw InvalidInputException("%s: the model has %llu features but x has %llu values", ERROR_NAME,
			                            static_cast<unsigned long long>(model.n_features),
			                            static_cast<unsigned long long>(entry.length));
		}
		row_data.resize(entry.length);
		for (idx_t j = 0; j < entry.length; j++) {
			auto idx = entry.offset + j;
			row_data[j] = child_validity.RowIsValid(idx) ? child_data[idx] : std::nan("");
		}
		auto view = model.View();
		double e = 0;
		AnofoxError error;
		if (!anofox_reconstruction_row_errors(&view, row_data.data(), 1, &e, &error)) {
			ThrowUnlessDegenerate(ERROR_NAME, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		out[row] = e;
	}
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding asserts this in debug builds).
static void ErrorFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	ErrorFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// Table macros
//===--------------------------------------------------------------------===//

// NannyML's alert: the value is outside the band (strictly); NaN never alerts.
#define ANOFOX_ALERT_SQL(v, lo, hi)                                                                                    \
	"CASE WHEN " v " IS NULL THEN NULL WHEN isnan(" v ") THEN false ELSE coalesce(" v " < " lo                         \
	", false) OR coalesce(" v " > " hi ", false) END"

// One row per (chunk, is_reference); reference chunks first. The model is fitted on the
// reference rows, the band on the reference chunks' errors. (The macro parameter
// is_reference is substituted wherever it appears, so the flag column is _anofox_ref.)
static const char *const PCA_RECONSTRUCTION_DRIFT_BY_SQL =
    "WITH _anofox_src AS ("
    " SELECT chunk_col, x_cols AS _anofox_x, coalesce((is_reference)::BOOLEAN, false) AS _anofox_ref"
    " FROM query_table(source::VARCHAR)),"
    " _anofox_model AS ("
    " SELECT pca_reconstruction_fit_agg(_anofox_x, options) AS _anofox_m FROM _anofox_src WHERE _anofox_ref),"
    " _anofox_chunks AS ("
    " SELECT chunk_col, _anofox_ref, pca_reconstruction_chunk_agg(_anofox_m, _anofox_x) AS _anofox_c"
    " FROM _anofox_src CROSS JOIN _anofox_model GROUP BY chunk_col, _anofox_ref),"
    " _anofox_band AS ("
    " SELECT pca_reconstruction_threshold_agg((_anofox_c).reconstruction_error, threshold) AS _anofox_b"
    " FROM _anofox_chunks WHERE _anofox_ref)"
    " SELECT chunk_col, _anofox_ref AS is_reference, (_anofox_c).n AS n,"
    " (_anofox_c).reconstruction_error AS reconstruction_error,"
    " (_anofox_c).sampling_error AS sampling_error,"
    " (_anofox_c).lower_confidence AS lower_confidence, (_anofox_c).upper_confidence AS upper_confidence,"
    " (_anofox_b).lower_threshold AS lower_threshold, (_anofox_b).upper_threshold AS "
    "upper_threshold, " ANOFOX_ALERT_SQL(
        "(_anofox_c).reconstruction_error", "(_anofox_b).lower_threshold",
        "(_anofox_b).upper_threshold") " AS alert FROM _anofox_chunks CROSS JOIN _anofox_band ORDER BY _anofox_ref "
                                       "DESC, chunk_col";

static const char *const ALERT_BAND_BY_SQL =
    "WITH _anofox_src AS ("
    " SELECT chunk_col, (metric_col)::DOUBLE AS _anofox_v, coalesce((is_reference)::BOOLEAN, false) AS _anofox_ref"
    " FROM query_table(source::VARCHAR)),"
    " _anofox_band AS ("
    " SELECT alert_band_agg(_anofox_v, options) AS _anofox_b FROM _anofox_src WHERE _anofox_ref)"
    " SELECT chunk_col, _anofox_v AS value, (_anofox_b).lower_threshold AS lower_threshold,"
    " (_anofox_b).upper_threshold AS upper_threshold, " ANOFOX_ALERT_SQL(
        "_anofox_v", "(_anofox_b).lower_threshold",
        "(_anofox_b).upper_threshold") " AS alert, _anofox_ref AS is_reference FROM _anofox_src CROSS JOIN _anofox_band"
                                       " ORDER BY _anofox_ref DESC, chunk_col";

#undef ANOFOX_ALERT_SQL

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
template <int KIND>
static void RegisterBand(ExtensionLoader &loader, const char *description, const char *example) {
	const char *name = BandName(KIND);
	AggregateFunctionSet set(name);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    name, args, LogicalType::ANY, AggregateFunction::StateSize<BandAggregateState>,
		    ANOFOX_AGG_INIT(BandInitialize), ANOFOX_GUARDED_UPDATE(BandUpdate, BandDestroy, BandInitialize),
		    BandCombine, ANOFOX_GUARDED_FINALIZE(BandFinalize, BandDestroy, BandInitialize), nullptr,
		    ANOFOX_AGG_BIND(BandBind<KIND>), BandDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription d1;
	d1.description = string(description) + " Default options.";
	d1.examples = {StringUtil::Format("%s(value)", name)};
	d1.categories = {"drift", "monitoring"};
	d1.parameter_names = {"value"};
	d1.parameter_types = {LogicalType::DOUBLE};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = description;
	d2.examples = {example};
	d2.categories = {"drift", "monitoring"};
	d2.parameter_names = {"value", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

void RegisterReconstructionFunctions(ExtensionLoader &loader) {
	auto list_type = LogicalType::LIST(LogicalType::DOUBLE);
	{
		AggregateFunctionSet set(FIT_NAME);
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {list_type};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			set.AddFunction(AggregateFunction(
			    FIT_NAME, args, LogicalType::ANY, AggregateFunction::StateSize<ReconstructionFitState>,
			    ANOFOX_AGG_INIT(FitInitialize), ANOFOX_GUARDED_UPDATE(FitUpdate, FitDestroy, FitInitialize), FitCombine,
			    ANOFOX_GUARDED_FINALIZE(FitFinalize, FitDestroy, FitInitialize), nullptr, ANOFOX_AGG_BIND(FitBind),
			    FitDestroy));
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d1;
		d1.description =
		    "NannyML's data-reconstruction drift model fitted on reference rows: standardise the columns "
		    "(population sd), keep the principal components explaining more than 65% of the variance. Returns "
		    "STRUCT(center, scale, sdev, proportion_of_variance, loadings, n_components, n_features, "
		    "n_observations, impute_nan). NULL rows are skipped; a NULL element is missing and imputed by the "
		    "reference column mean.";
		d1.examples = {"pca_reconstruction_fit_agg([x1, x2, x3])"};
		d1.categories = {"drift", "monitoring", "multivariate"};
		d1.parameter_names = {"x"};
		d1.parameter_types = {list_type};
		info.descriptions.push_back(std::move(d1));
		FunctionDescription d2;
		d2.description = "NannyML's data-reconstruction drift model with options: variance (share of the variance "
		                 "the kept components explain, default 0.65) or n_components (a fixed number), scale "
		                 "(default true), impute_nan (default true; false makes a missing value an error).";
		d2.examples = {"pca_reconstruction_fit_agg([x1, x2, x3], {'n_components': 2})"};
		d2.categories = {"drift", "monitoring", "multivariate"};
		d2.parameter_names = {"x", "options"};
		d2.parameter_types = {list_type, LogicalType::ANY};
		info.descriptions.push_back(std::move(d2));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set(CHUNK_NAME);
		set.AddFunction(
		    AggregateFunction(CHUNK_NAME, {LogicalType::ANY, list_type}, LogicalType::ANY,
		                      AggregateFunction::StateSize<ReconstructionChunkState>, ANOFOX_AGG_INIT(ChunkInitialize),
		                      ANOFOX_GUARDED_UPDATE(ChunkUpdate, ChunkDestroy, ChunkInitialize), ChunkCombine,
		                      ANOFOX_GUARDED_FINALIZE(ChunkFinalize, ChunkDestroy, ChunkInitialize), nullptr,
		                      ANOFOX_AGG_BIND(ChunkBind), ChunkDestroy));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = "Reconstruction error of a chunk of rows under a pca_reconstruction_fit_agg model (NannyML): "
		                "STRUCT(n, reconstruction_error (mean of the row errors), std, sampling_error (std / "
		                "sqrt(n)), lower_confidence (max(error - 3 sampling_error, 0)), upper_confidence). The model "
		                "must be the same on every row of the group.";
		d.examples = {"pca_reconstruction_chunk_agg(model, [x1, x2, x3])"};
		d.categories = {"drift", "monitoring", "multivariate"};
		d.parameter_names = {"model", "x"};
		d.parameter_types = {LogicalType::ANY, list_type};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		ScalarFunction func(ERROR_NAME, {LogicalType::ANY, list_type}, LogicalType::DOUBLE,
		                    DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, ErrorFunction), ANOFOX_SCALAR_BIND(ErrorBind));
		MarkFallible(func);
		SetNullHandlingOf(func, FunctionNullHandling::SPECIAL_HANDLING);
		ScalarFunctionSet set(ERROR_NAME);
		set.AddFunction(func);
		CreateScalarFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = "Reconstruction error of one row under a pca_reconstruction_fit_agg model: the euclidean "
		                "norm of the standardised row's residual outside the kept components. A NULL element is "
		                "imputed by the reference mean (unless the model has impute_nan false); NULL when the model "
		                "or x is NULL.";
		d.examples = {"pca_reconstruction_error(model, [1.5, 2.0, 0.3])"};
		d.categories = {"drift", "monitoring", "multivariate"};
		d.parameter_names = {"model", "x"};
		d.parameter_types = {LogicalType::ANY, list_type};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	RegisterBand<BAND_RECONSTRUCTION>(
	    loader,
	    "Alert band of the reconstruction error from its values on the reference chunks (NannyML): mean +- 3 "
	    "population sd by default, a lower threshold at or below 0 dropped (NULL). Options: method ('std', "
	    "'constant', 'relative'), k, lower, upper (NULL disables a side).",
	    "pca_reconstruction_threshold_agg(reconstruction_error, {'method': 'std', 'k': 2})");
	RegisterBand<BAND_GENERIC>(
	    loader,
	    "Alert thresholds of a metric from its values on the reference chunks (NannyML's StandardDeviationThreshold "
	    "/ ConstantThreshold, calculate_threshold_values): STRUCT(lower_threshold, upper_threshold), NULL for no "
	    "threshold. Options: method 'std' (mean -+ lower / upper population sds, default 3; 'k' sets both), "
	    "'constant' (alias 'fixed': lower / upper are the thresholds) or 'relative' (mean -+ lower / upper times "
	    "|mean|); lower_limit / upper_limit clamp a threshold beyond the metric's range, or drop it with "
	    "override_with_none. NULL values are skipped, NaN ignored.",
	    "alert_band_agg(accuracy, {'method': 'std', 'k': 3, 'lower_limit': 0, 'upper_limit': 1})");
	{
		auto macro = CreateContractTableMacro(
		    "pca_reconstruction_drift_by", {"source", "chunk_col", "x_cols", "is_reference"},
		    {{"options", "NULL"}, {"threshold", "NULL"}}, PCA_RECONSTRUCTION_DRIFT_BY_SQL,
		    "Table macro: NannyML's multivariate drift by PCA reconstruction error. Fits "
		    "pca_reconstruction_fit_agg(x_cols, options) on the rows where is_reference (a BOOLEAN expression) is "
		    "true, and returns one row per (chunk_col, is_reference), reference chunks first: n, "
		    "reconstruction_error, sampling_error, lower_confidence, upper_confidence, lower_threshold and "
		    "upper_threshold (pca_reconstruction_threshold_agg(reconstruction error of the reference chunks, "
		    "threshold)) and alert (the error is outside the band).",
		    "pca_reconstruction_drift_by('loans', chunk, [car_value, debt_to_income], period = 'reference')",
		    {"drift", "monitoring", "table-macro"});
		loader.RegisterFunction(*macro);
	}
	{
		auto macro = CreateContractTableMacro(
		    "alert_band_by", {"source", "chunk_col", "metric_col", "is_reference"}, {{"options", "NULL"}},
		    ALERT_BAND_BY_SQL,
		    "Table macro: alert thresholds for a chunked metric (one row per chunk) from its reference chunks. "
		    "alert_band_agg(metric_col, options) over the rows where is_reference (a BOOLEAN expression) is "
		    "true; returns every row with chunk_col, value, lower_threshold, upper_threshold, alert (value outside "
		    "the band; NaN never alerts) and is_reference, reference chunks first.",
		    "alert_band_by('chunk_metrics', chunk, accuracy, chunk < 5, options := {'lower_limit': 0, "
		    "'upper_limit': 1})",
		    {"drift", "monitoring", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

} // namespace duckdb
