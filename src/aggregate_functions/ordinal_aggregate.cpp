// Ordinal regression (#313): ordinal_fit_agg (cumulative link models, rms::orm /
// MASS::polr), continuation_ratio_fit_agg (rms::cr.setup + lrm), ordinal_predict and
// the ordinal_fit_predict_by / continuation_ratio_fit_predict_by table macros;
// predict(model, x) dispatches ordinal models here. The fits are upstream
// anofox-regression OrdinalRegressor / ContinuationRatioRegressor; this file buffers
// rows, parses the options and marshals the model STRUCT.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/glm_weights_arg.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/canonical_order.hpp"
#include "anofox_statistics_banner.hpp"
#include "anofox_statistics_extension.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

// Layout checks against the Rust #[repr(C)] structs, without assuming 8-byte
// pointers / size_t (wasm32).
static_assert(offsetof(AnofoxOrdinalOptions, link) == 2 * sizeof(double) &&
                  offsetof(AnofoxOrdinalOptions, po_test) == 2 * sizeof(double) + 3 * sizeof(uint32_t) &&
                  sizeof(AnofoxOrdinalOptions) == 2 * sizeof(double) + 4 * sizeof(uint32_t),
              "AnofoxOrdinalOptions must match OrdinalOptionsFFI");
static_assert(offsetof(AnofoxOrdinalFitResult, n_levels) == 8 * sizeof(double *) &&
                  offsetof(AnofoxOrdinalFitResult, log_likelihood) ==
                      ((8 * sizeof(double *) + 3 * sizeof(size_t) + alignof(double) - 1) / alignof(double)) *
                          alignof(double) &&
                  offsetof(AnofoxOrdinalFitResult, iterations) ==
                      offsetof(AnofoxOrdinalFitResult, log_likelihood) + 10 * sizeof(double) &&
                  offsetof(AnofoxOrdinalFitResult, converged) ==
                      offsetof(AnofoxOrdinalFitResult, iterations) + sizeof(uint32_t),
              "AnofoxOrdinalFitResult must match OrdinalFitResultFFI");
static_assert(offsetof(AnofoxContinuationRatioFitResult, n_levels) == 5 * sizeof(double *) &&
                  offsetof(AnofoxContinuationRatioFitResult, log_likelihood) ==
                      ((5 * sizeof(double *) + 5 * sizeof(size_t) + alignof(double) - 1) / alignof(double)) *
                          alignof(double) &&
                  offsetof(AnofoxContinuationRatioFitResult, iterations) ==
                      offsetof(AnofoxContinuationRatioFitResult, log_likelihood) + 2 * sizeof(double) &&
                  offsetof(AnofoxContinuationRatioFitResult, converged) ==
                      offsetof(AnofoxContinuationRatioFitResult, iterations) + sizeof(uint32_t),
              "AnofoxContinuationRatioFitResult must match ContinuationRatioFitResultFFI");

namespace {

const char *const ORDINAL_LINK_NAMES[] = {"logistic", "probit", "loglog", "cloglog", "cauchit"};

//! The FFI link code of an rms::orm family name (logit: alias of logistic), or -1.
int OrdinalLinkCode(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "logit") {
		return 0;
	}
	for (int i = 0; i < 5; i++) {
		if (lower == ORDINAL_LINK_NAMES[i]) {
			return i;
		}
	}
	return -1;
}

//! Visit the (key, value) pairs of a constant MAP / STRUCT options argument (keys lower-cased).
template <class F>
void VisitOptions(ClientContext &context, Expression &expr, const char *fn_name, F &&visit) {
	Value options = UnwrapVariant(EvaluateConstantOptions(context, expr, fn_name));
	if (options.IsNull()) {
		return;
	}
	if (options.type().id() == LogicalTypeId::STRUCT) {
		auto &vals = StructValue::GetChildren(options);
		for (idx_t i = 0; i < vals.size(); i++) {
			auto v = UnwrapVariant(vals[i]);
			if (!v.IsNull()) {
				visit(StringUtil::Lower(NameString(StructType::GetChildName(options.type(), i))), v);
			}
		}
	} else if (options.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(options)) {
			auto &kv = StructValue::GetChildren(entry);
			auto v = UnwrapVariant(kv[1]);
			if (!v.IsNull()) {
				visit(StringUtil::Lower(kv[0].ToString()), v);
			}
		}
	} else {
		throw InvalidInputException("%s: options must be a MAP or STRUCT", fn_name);
	}
}

double OptionDouble(const char *fn_name, const string &key, const Value &v) {
	try {
		return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn_name, key, v.ToString());
	}
}

int64_t OptionInteger(const char *fn_name, const string &key, const Value &v) {
	double d = OptionDouble(fn_name, key, v);
	if (!(std::isfinite(d) && d == std::floor(d))) {
		throw InvalidInputException("%s: option '%s' must be an integer, got '%s'", fn_name, key, v.ToString());
	}
	return static_cast<int64_t>(d);
}

bool OptionBool(const char *fn_name, const string &key, const Value &v) {
	try {
		return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a BOOLEAN, got '%s'", fn_name, key, v.ToString());
	}
}

void AppendDoubles(Vector &list_vec, idx_t row, const double *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	auto child_data = FlatDataMut<double>(ListVector::GetEntry(list_vec));
	for (idx_t i = 0; i < len; i++) {
		child_data[offset + i] = data[i];
	}
	ListVector::SetListSize(list_vec, offset + len);
	ListVector::GetData(list_vec)[row] = {offset, len};
}

//! A DOUBLE value; NaN (not computed / undefined) becomes NULL.
void SetDoubleOrNull(Vector &vec, idx_t row, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<double>(vec)[row] = v;
	}
}

void SetString(Vector &vec, idx_t row, const char *value) {
	FlatDataMut<string_t>(vec)[row] = StringVector::AddString(vec, value);
}

} // namespace

//===--------------------------------------------------------------------===//
// Shared state: buffered rows of (y, x [, weight])
//===--------------------------------------------------------------------===//
struct OrdinalAggregateState {
	vector<double> y;
	vector<vector<double>> x_columns;
	vector<double> weights; // empty unless weighted
	idx_t n_features;
	bool initialized;

	OrdinalAggregateState() : n_features(0), initialized(false) {
	}

	void Reset() {
		y.clear();
		x_columns.clear();
		weights.clear();
		n_features = 0;
		initialized = false;
	}
};

struct OrdinalBindData : public FunctionData {
	bool continuation_ratio = false;
	bool weighted = false;
	// ordinal_fit_agg
	uint32_t link = 0;
	double penalty = 0.0;
	uint32_t offset_column = 0; // 1-based, 0: none
	uint32_t max_iterations = 100;
	double tolerance = 1e-10;
	int32_t po_test = -1; // auto
	// continuation_ratio_fit_agg: 0-based columns
	vector<uint64_t> cohort_specific;

	const char *Name() const {
		return continuation_ratio ? "continuation_ratio_fit_agg" : "ordinal_fit_agg";
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<OrdinalBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<OrdinalBindData>();
		return continuation_ratio == o.continuation_ratio && weighted == o.weighted && link == o.link &&
		       penalty == o.penalty && offset_column == o.offset_column && max_iterations == o.max_iterations &&
		       tolerance == o.tolerance && po_test == o.po_test && cohort_specific == o.cohort_specific;
	}
};

static LogicalType OrdinalModelType() {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("levels", dlist));
	c.push_back(MakeChild("counts", dlist));
	c.push_back(MakeChild("intercepts", dlist));
	c.push_back(MakeChild("intercept_std_errors", dlist));
	c.push_back(MakeChild("coefficients", dlist));
	c.push_back(MakeChild("std_errors", dlist));
	c.push_back(MakeChild("z_values", dlist));
	c.push_back(MakeChild("p_values", dlist));
	c.push_back(MakeChild("log_likelihood", LogicalType::DOUBLE));
	c.push_back(MakeChild("null_log_likelihood", LogicalType::DOUBLE));
	c.push_back(MakeChild("penalized_log_likelihood", LogicalType::DOUBLE));
	c.push_back(MakeChild("lr_chisq", LogicalType::DOUBLE));
	c.push_back(MakeChild("lr_df", LogicalType::DOUBLE));
	c.push_back(MakeChild("lr_p_value", LogicalType::DOUBLE));
	c.push_back(MakeChild("aic", LogicalType::DOUBLE));
	c.push_back(MakeChild("po_test_statistic", LogicalType::DOUBLE));
	c.push_back(MakeChild("po_test_df", LogicalType::DOUBLE));
	c.push_back(MakeChild("po_test_p_value", LogicalType::DOUBLE));
	c.push_back(MakeChild("n_levels", LogicalType::BIGINT));
	c.push_back(MakeChild("n_observations", LogicalType::BIGINT));
	c.push_back(MakeChild("n_features", LogicalType::BIGINT));
	c.push_back(MakeChild("offset_column", LogicalType::BIGINT));
	c.push_back(MakeChild("penalty", LogicalType::DOUBLE));
	c.push_back(MakeChild("iterations", LogicalType::INTEGER));
	c.push_back(MakeChild("converged", LogicalType::BOOLEAN));
	c.push_back(MakeChild("model_type", LogicalType::VARCHAR));
	c.push_back(MakeChild("link", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

static LogicalType ContinuationRatioModelType() {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("levels", dlist));
	c.push_back(MakeChild("coefficients", dlist));
	c.push_back(MakeChild("std_errors", dlist));
	c.push_back(MakeChild("z_values", dlist));
	c.push_back(MakeChild("p_values", dlist));
	c.push_back(MakeChild("cohort_specific", LogicalType::LIST(LogicalType::BIGINT)));
	c.push_back(MakeChild("log_likelihood", LogicalType::DOUBLE));
	c.push_back(MakeChild("aic", LogicalType::DOUBLE));
	c.push_back(MakeChild("n_levels", LogicalType::BIGINT));
	c.push_back(MakeChild("n_observations", LogicalType::BIGINT));
	c.push_back(MakeChild("n_records", LogicalType::BIGINT));
	c.push_back(MakeChild("n_features", LogicalType::BIGINT));
	c.push_back(MakeChild("iterations", LogicalType::INTEGER));
	c.push_back(MakeChild("converged", LogicalType::BOOLEAN));
	c.push_back(MakeChild("model_type", LogicalType::VARCHAR));
	c.push_back(MakeChild("link", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

static void OrdinalAggInitialize(data_ptr_t state_p) {
	new (state_p) OrdinalAggregateState();
}

static void OrdinalAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OrdinalAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~OrdinalAggregateState();
	}
}

static void OrdinalAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                             idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<OrdinalBindData>();
	UnifiedVectorFormat y_data, x_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	GlmWeightsInput w_input;
	if (bind.weighted) {
		w_input.Load(inputs[2], count);
	}

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OrdinalAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue; // NULL y or NULL x: the row is left out
		}
		double y = y_values[y_idx];
		if (!std::isfinite(y)) {
			continue;
		}
		double w = 1.0;
		if (bind.weighted && !w_input.Get(i, w)) {
			continue; // NULL weight: the row is left out, as wls_fit_agg
		}
		auto entry = x_list_data[x_idx];
		// A row with a NULL, NaN or infinite feature is left out, as a NULL row.
		if (ListHasNullElement(x_child_validity, entry)) {
			continue;
		}
		bool finite = true;
		for (idx_t j = 0; j < entry.length; j++) {
			finite = finite && std::isfinite(x_child_data[entry.offset + j]);
		}
		if (!finite) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(entry.length);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("%s: inconsistent feature count: expected %llu, got %llu", bind.Name(),
			                            (unsigned long long)state.n_features, (unsigned long long)entry.length);
		}
		state.y.push_back(y);
		for (idx_t j = 0; j < entry.length; j++) {
			state.x_columns[j].push_back(x_child_data[entry.offset + j]);
		}
		if (bind.weighted) {
			state.weights.push_back(w);
		}
	}
}

static void OrdinalAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                              idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<OrdinalBindData>();
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (OrdinalAggregateState **)source_data.data;
	auto targets = (OrdinalAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y = CombineTake(source.y, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.weights = CombineTake(source.weights, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("%s: inconsistent feature count: expected %llu, got %llu", bind.Name(),
			                            (unsigned long long)target.n_features, (unsigned long long)source.n_features);
		}
		target.y.insert(target.y.end(), source.y.begin(), source.y.end());
		target.weights.insert(target.weights.end(), source.weights.begin(), source.weights.end());
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

//! Canonical row order (bitwise-identical results whatever the thread count, #226) and
//! the FFI column views.
static vector<AnofoxDataArray> PrepareRows(OrdinalAggregateState &state, bool weighted) {
	if (weighted) {
		CanonicalRows().Keys(state.x_columns).Key(state.y).Key(state.weights).Sort();
	} else {
		CanonicalRows().Keys(state.x_columns).Key(state.y).Sort();
	}
	vector<AnofoxDataArray> x_arrays;
	for (auto &col : state.x_columns) {
		x_arrays.push_back({col.data(), nullptr, col.size()});
	}
	return x_arrays;
}

static void OrdinalFinalizeRow(OrdinalAggregateState &state, const OrdinalBindData &bind,
                               vector<unique_ptr<Vector>> &entries, Vector &result, idx_t row) {
	auto x_arrays = PrepareRows(state, bind.weighted);
	AnofoxOrdinalOptions options {};
	options.penalty = bind.penalty;
	options.tolerance = bind.tolerance;
	options.link = bind.link;
	options.max_iterations = bind.max_iterations;
	options.offset_column = bind.offset_column;
	options.po_test = bind.po_test;
	AnofoxOrdinalFitResult fit;
	AnofoxError error;
	if (!anofox_ordinal_fit(state.y.data(), state.y.size(), x_arrays.data(), x_arrays.size(),
	                        bind.weighted ? state.weights.data() : nullptr, options, &fit, &error)) {
		ThrowUnlessDegenerate("ordinal_fit_agg", error);
		FlatVector::SetNull(result, row, true);
		return;
	}
	const idx_t k = fit.n_levels;
	const idx_t p = fit.n_features;
	AppendDoubles(StructEntry(entries[0]), row, fit.levels, k);
	AppendDoubles(StructEntry(entries[1]), row, fit.counts, k);
	AppendDoubles(StructEntry(entries[2]), row, fit.intercepts, k - 1);
	if (fit.intercept_std_errors) {
		AppendDoubles(StructEntry(entries[3]), row, fit.intercept_std_errors, k - 1);
	} else {
		FlatVector::SetNull(StructEntry(entries[3]), row, true);
	}
	AppendDoubles(StructEntry(entries[4]), row, fit.coefficients, p);
	AppendDoubles(StructEntry(entries[5]), row, fit.std_errors, p);
	AppendDoubles(StructEntry(entries[6]), row, fit.z_values, p);
	AppendDoubles(StructEntry(entries[7]), row, fit.p_values, p);
	FlatDataMut<double>(StructEntry(entries[8]))[row] = fit.log_likelihood;
	FlatDataMut<double>(StructEntry(entries[9]))[row] = fit.null_log_likelihood;
	FlatDataMut<double>(StructEntry(entries[10]))[row] = fit.penalized_log_likelihood;
	FlatDataMut<double>(StructEntry(entries[11]))[row] = fit.lr_chisq;
	FlatDataMut<double>(StructEntry(entries[12]))[row] = fit.lr_df;
	SetDoubleOrNull(StructEntry(entries[13]), row, fit.lr_p_value);
	FlatDataMut<double>(StructEntry(entries[14]))[row] = fit.aic;
	SetDoubleOrNull(StructEntry(entries[15]), row, fit.po_statistic);
	SetDoubleOrNull(StructEntry(entries[16]), row, fit.po_df);
	SetDoubleOrNull(StructEntry(entries[17]), row, fit.po_p_value);
	FlatDataMut<int64_t>(StructEntry(entries[18]))[row] = static_cast<int64_t>(k);
	FlatDataMut<int64_t>(StructEntry(entries[19]))[row] = static_cast<int64_t>(fit.n_observations);
	FlatDataMut<int64_t>(StructEntry(entries[20]))[row] = static_cast<int64_t>(p);
	if (bind.offset_column > 0) {
		FlatDataMut<int64_t>(StructEntry(entries[21]))[row] = static_cast<int64_t>(bind.offset_column);
	} else {
		FlatVector::SetNull(StructEntry(entries[21]), row, true);
	}
	FlatDataMut<double>(StructEntry(entries[22]))[row] = bind.penalty;
	FlatDataMut<int32_t>(StructEntry(entries[23]))[row] = static_cast<int32_t>(fit.iterations);
	FlatDataMut<bool>(StructEntry(entries[24]))[row] = fit.converged;
	SetString(StructEntry(entries[25]), row, "ordinal");
	SetString(StructEntry(entries[26]), row, ORDINAL_LINK_NAMES[bind.link]);
	anofox_free_ordinal_result(&fit);
}

static void ContinuationRatioFinalizeRow(OrdinalAggregateState &state, const OrdinalBindData &bind,
                                         vector<unique_ptr<Vector>> &entries, Vector &result, idx_t row) {
	auto x_arrays = PrepareRows(state, bind.weighted);
	AnofoxContinuationRatioFitResult fit;
	AnofoxError error;
	if (!anofox_continuation_ratio_fit(state.y.data(), state.y.size(), x_arrays.data(), x_arrays.size(),
	                                   bind.weighted ? state.weights.data() : nullptr,
	                                   bind.cohort_specific.empty() ? nullptr : bind.cohort_specific.data(),
	                                   bind.cohort_specific.size(), &fit, &error)) {
		ThrowUnlessDegenerate("continuation_ratio_fit_agg", error);
		FlatVector::SetNull(result, row, true);
		return;
	}
	const idx_t q = fit.n_coefficients;
	AppendDoubles(StructEntry(entries[0]), row, fit.levels, fit.n_levels);
	AppendDoubles(StructEntry(entries[1]), row, fit.coefficients, q);
	AppendDoubles(StructEntry(entries[2]), row, fit.std_errors, q);
	AppendDoubles(StructEntry(entries[3]), row, fit.z_values, q);
	AppendDoubles(StructEntry(entries[4]), row, fit.p_values, q);
	{
		auto &cs_vec = StructEntry(entries[5]);
		auto offset = ListVector::GetListSize(cs_vec);
		const idx_t len = bind.cohort_specific.size();
		ListVector::Reserve(cs_vec, offset + len);
		auto child = FlatDataMut<int64_t>(ListVector::GetEntry(cs_vec));
		for (idx_t j = 0; j < len; j++) {
			child[offset + j] = static_cast<int64_t>(bind.cohort_specific[j] + 1);
		}
		ListVector::SetListSize(cs_vec, offset + len);
		ListVector::GetData(cs_vec)[row] = {offset, len};
	}
	FlatDataMut<double>(StructEntry(entries[6]))[row] = fit.log_likelihood;
	FlatDataMut<double>(StructEntry(entries[7]))[row] = fit.aic;
	FlatDataMut<int64_t>(StructEntry(entries[8]))[row] = static_cast<int64_t>(fit.n_levels);
	FlatDataMut<int64_t>(StructEntry(entries[9]))[row] = static_cast<int64_t>(fit.n_observations);
	FlatDataMut<int64_t>(StructEntry(entries[10]))[row] = static_cast<int64_t>(fit.n_records);
	FlatDataMut<int64_t>(StructEntry(entries[11]))[row] = static_cast<int64_t>(fit.n_features);
	FlatDataMut<int32_t>(StructEntry(entries[12]))[row] = static_cast<int32_t>(fit.iterations);
	FlatDataMut<bool>(StructEntry(entries[13]))[row] = fit.converged;
	SetString(StructEntry(entries[14]), row, "continuation_ratio");
	SetString(StructEntry(entries[15]), row, "logit");
	anofox_free_continuation_ratio_result(&fit);
}

static void OrdinalAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                               idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OrdinalAggregateState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<OrdinalBindData>();
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.y.size() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (bind.offset_column > state.n_features) {
			throw InvalidInputException("ordinal_fit_agg: option 'offset' is column %u but x has %llu values",
			                            bind.offset_column, (unsigned long long)state.n_features);
		}
		for (auto c : bind.cohort_specific) {
			if (c >= state.n_features) {
				throw InvalidInputException(
				    "continuation_ratio_fit_agg: option 'cohort_specific' names column %llu but x has %llu values",
				    (unsigned long long)(c + 1), (unsigned long long)state.n_features);
			}
		}
		if (bind.continuation_ratio) {
			ContinuationRatioFinalizeRow(state, bind, entries, result, row);
		} else {
			OrdinalFinalizeRow(state, bind, entries, result, row);
		}
		state.Reset();
	}
}

static unique_ptr<FunctionData> OrdinalBindCommon(ClientContext &context, AggregateBindFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments,
                                                  bool continuation_ratio) {
	auto bind = make_uniq<OrdinalBindData>();
	bind->continuation_ratio = continuation_ratio;
	const char *fn_name = bind->Name();
	// f(y, x, w) / f(y, x, w, options): case weights, as poisson_fit_agg (#305).
	bind->weighted = GlmWeightsArgument(arguments);
	const idx_t options_idx = bind->weighted ? 3 : 2;
	if (arguments.size() > options_idx) {
		VisitOptions(context, *arguments[options_idx], fn_name, [&](const string &key, const Value &v) {
			if (continuation_ratio) {
				if (key != "cohort_specific") {
					throw InvalidInputException("%s: unknown option '%s' (supported: cohort_specific)", fn_name, key);
				}
				vector<Value> items;
				if (v.type().id() == LogicalTypeId::LIST) {
					items = ListValue::GetChildren(v);
				} else {
					items.push_back(v);
				}
				for (auto &item : items) {
					auto c = OptionInteger(fn_name, key, item);
					if (c < 1) {
						throw InvalidInputException("%s: option 'cohort_specific' holds 1-based x columns, got %lld",
						                            fn_name, (long long)c);
					}
					auto c0 = static_cast<uint64_t>(c - 1);
					if (std::find(bind->cohort_specific.begin(), bind->cohort_specific.end(), c0) ==
					    bind->cohort_specific.end()) {
						bind->cohort_specific.push_back(c0);
					}
				}
				return;
			}
			if (key == "link") {
				int code = OrdinalLinkCode(v.ToString());
				if (code < 0) {
					throw InvalidInputException("%s: option 'link' must be one of logistic, probit, loglog, cloglog, "
					                            "cauchit, got '%s'",
					                            fn_name, v.ToString());
				}
				bind->link = static_cast<uint32_t>(code);
			} else if (key == "penalty") {
				bind->penalty = OptionDouble(fn_name, key, v);
				if (!(std::isfinite(bind->penalty) && bind->penalty >= 0.0)) {
					throw InvalidInputException("%s: option 'penalty' must be finite and >= 0, got %g", fn_name,
					                            bind->penalty);
				}
			} else if (key == "offset") {
				auto c = OptionInteger(fn_name, key, v);
				if (c < 1 || c > std::numeric_limits<int32_t>::max()) {
					throw InvalidInputException("%s: option 'offset' is a 1-based column of x, got %lld", fn_name,
					                            (long long)c);
				}
				bind->offset_column = static_cast<uint32_t>(c);
			} else if (key == "max_iterations") {
				auto it = OptionInteger(fn_name, key, v);
				if (it < 1 || it > std::numeric_limits<int32_t>::max()) {
					throw InvalidInputException("%s: option 'max_iterations' must be at least 1, got %lld", fn_name,
					                            (long long)it);
				}
				bind->max_iterations = static_cast<uint32_t>(it);
			} else if (key == "tolerance") {
				bind->tolerance = OptionDouble(fn_name, key, v);
				if (!(std::isfinite(bind->tolerance) && bind->tolerance > 0.0)) {
					throw InvalidInputException("%s: option 'tolerance' must be finite and > 0, got %g", fn_name,
					                            bind->tolerance);
				}
			} else if (key == "po_test") {
				bind->po_test = OptionBool(fn_name, key, v) ? 1 : 0;
			} else {
				throw InvalidInputException("%s: unknown option '%s' (supported: link, penalty, offset, "
				                            "max_iterations, tolerance, po_test)",
				                            fn_name, key);
			}
		});
	}
	SetReturnTypeOf(function, continuation_ratio ? ContinuationRatioModelType() : OrdinalModelType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(bind);
}

static unique_ptr<FunctionData> OrdinalAggBind(ClientContext &context, AggregateBindFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	return OrdinalBindCommon(context, function, arguments, false);
}

static unique_ptr<FunctionData> ContinuationRatioAggBind(ClientContext &context, AggregateBindFunction &function,
                                                         vector<unique_ptr<Expression>> &arguments) {
	return OrdinalBindCommon(context, function, arguments, true);
}

//===--------------------------------------------------------------------===//
// ordinal_predict(model, x DOUBLE[] [, options]) and predict(model, x [, options])
//===--------------------------------------------------------------------===//
enum class OrdinalPredictType : uint8_t { CLASS, PROB, EXCEEDANCE, MEAN };

struct OrdinalPredictBindData : public FunctionData {
	OrdinalPredictType type = OrdinalPredictType::CLASS;
	bool continuation_ratio = false;
	idx_t levels_idx = DConstants::INVALID_INDEX;
	idx_t intercepts_idx = DConstants::INVALID_INDEX;
	idx_t coef_idx = DConstants::INVALID_INDEX;
	idx_t link_idx = DConstants::INVALID_INDEX;
	idx_t offset_idx = DConstants::INVALID_INDEX;
	idx_t cohort_idx = DConstants::INVALID_INDEX;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<OrdinalPredictBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<OrdinalPredictBindData>();
		return type == o.type && continuation_ratio == o.continuation_ratio && levels_idx == o.levels_idx &&
		       intercepts_idx == o.intercepts_idx && coef_idx == o.coef_idx && link_idx == o.link_idx &&
		       offset_idx == o.offset_idx && cohort_idx == o.cohort_idx;
	}
};

//! An ordinal_fit_agg model (levels + intercepts) or a continuation_ratio_fit_agg model
//! (levels + cohort_specific).
bool IsOrdinalModelType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	bool levels = false, other = false;
	for (auto &child : StructType::GetChildTypes(type)) {
		auto &name = NameString(child.first);
		if (name == "levels" && child.second.id() == LogicalTypeId::LIST) {
			levels = true;
		} else if ((name == "intercepts" || name == "cohort_specific") && child.second.id() == LogicalTypeId::LIST) {
			other = true;
		}
	}
	return levels && other;
}

static void OrdinalPredictFunction(DataChunk &args, ExpressionState &state, Vector &result);

unique_ptr<FunctionData> OrdinalPredictBind(ClientContext &context, ScalarBindFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments, const char *fn_name) {
	auto &in_type = ReturnTypeOf(*arguments[0]);
	if (!IsOrdinalModelType(in_type)) {
		throw InvalidInputException("%s(model, x): model must be the STRUCT returned by ordinal_fit_agg or "
		                            "continuation_ratio_fit_agg, got %s",
		                            fn_name, in_type.ToString());
	}
	auto data = make_uniq<OrdinalPredictBindData>();
	// Cast the model's fields to the fit's types (a hand-written literal may hold DECIMALs).
	child_list_t<LogicalType> cast_children;
	auto &children = StructType::GetChildTypes(in_type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = NameString(children[i].first);
		auto type = children[i].second;
		if (name == "levels") {
			data->levels_idx = i;
			type = LogicalType::LIST(LogicalType::DOUBLE);
		} else if (name == "intercepts") {
			data->intercepts_idx = i;
			type = LogicalType::LIST(LogicalType::DOUBLE);
		} else if (name == "coefficients") {
			data->coef_idx = i;
			type = LogicalType::LIST(LogicalType::DOUBLE);
		} else if (name == "link") {
			data->link_idx = i;
			type = LogicalType::VARCHAR;
		} else if (name == "offset_column") {
			data->offset_idx = i;
			type = LogicalType::BIGINT;
		} else if (name == "cohort_specific") {
			data->cohort_idx = i;
			type = LogicalType::LIST(LogicalType::BIGINT);
		}
		cast_children.push_back(make_pair(children[i].first, type));
	}
	data->continuation_ratio = data->intercepts_idx == DConstants::INVALID_INDEX;
	if (data->coef_idx == DConstants::INVALID_INDEX) {
		throw InvalidInputException("%s(model, x): an ordinal model needs a coefficients field, got %s", fn_name,
		                            in_type.ToString());
	}
	auto target = LogicalType::STRUCT(std::move(cast_children));
	if (target != in_type) {
		arguments[0] = BoundCastExpression::AddCastToType(context, std::move(arguments[0]), target);
	}
	if (arguments.size() >= 3) {
		VisitOptions(context, *arguments[2], fn_name, [&](const string &key, const Value &val) {
			if (key != "type") {
				throw InvalidInputException("%s: unknown option '%s' for an ordinal model (supported: type)", fn_name,
				                            key);
			}
			auto v = StringUtil::Lower(val.ToString());
			if (v == "class") {
				data->type = OrdinalPredictType::CLASS;
			} else if (v == "prob") {
				data->type = OrdinalPredictType::PROB;
			} else if (v == "exceedance") {
				data->type = OrdinalPredictType::EXCEEDANCE;
			} else if (v == "mean") {
				data->type = OrdinalPredictType::MEAN;
			} else if (v == "quantile") {
				throw InvalidInputException(
				    "%s: type 'quantile' is not available for a stored ordinal model (upstream's predicted "
				    "quantiles need the fitted model object); use 'class', 'prob', 'exceedance' or 'mean'",
				    fn_name);
			} else {
				throw InvalidInputException(
				    "%s: option 'type' must be 'class', 'prob', 'exceedance' or 'mean' for an ordinal model, got '%s'",
				    fn_name, val.ToString());
			}
		});
	}
	LogicalType ret = LogicalType::DOUBLE;
	if (data->type == OrdinalPredictType::PROB || data->type == OrdinalPredictType::EXCEEDANCE) {
		ret = LogicalType::LIST(LogicalType::DOUBLE);
	}
	SetReturnTypeOf(bound_function, ret);
	SetFunctionCallbackOf(bound_function, DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, OrdinalPredictFunction));
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(data);
}

static unique_ptr<FunctionData> OrdinalPredictOwnBind(ClientContext &context, ScalarBindFunction &bound_function,
                                                      vector<unique_ptr<Expression>> &arguments) {
	return OrdinalPredictBind(context, bound_function, arguments, "ordinal_predict");
}

//! The doubles of one list row (throws on a NULL element).
static vector<double> ListDoubles(Vector &list_vec, idx_t row, const char *field) {
	auto e = FlatDataMut<list_entry_t>(list_vec)[row];
	auto &child = ListVector::GetEntry(list_vec);
	if (ListHasNullElement(FlatValidityMut(child), e)) {
		throw InvalidInputException("ordinal_predict: the model's %s must not hold NULLs", field);
	}
	auto d = FlatDataMut<double>(child);
	return vector<double>(d + e.offset, d + e.offset + e.length);
}

static void OrdinalPredictFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = BindInfoOf(func_expr)->Cast<OrdinalPredictBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &model_vec = args.data[0];
	auto &x_vec = args.data[1];
	auto &entries = StructVector::GetEntries(model_vec);
	auto &levels_vec = StructEntry(entries[bind.levels_idx]);
	auto &coef_vec = StructEntry(entries[bind.coef_idx]);
	auto x_lists = FlatDataMut<list_entry_t>(x_vec);
	auto &x_child = ListVector::GetEntry(x_vec);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);

	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(model_vec, row) || FlatVector::IsNull(x_vec, row) ||
		    FlatVector::IsNull(levels_vec, row) || FlatVector::IsNull(coef_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto x_entry = x_lists[row];
		if (ListHasNullElement(x_child_validity, x_entry)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto levels = ListDoubles(levels_vec, row, "levels");
		auto coefficients = ListDoubles(coef_vec, row, "coefficients");
		const idx_t k = levels.size();
		vector<double> proba(k), exceedance(k > 0 ? k - 1 : 0);
		double mean = 0.0;
		size_t best = 0;
		AnofoxError error;
		bool ok;
		if (bind.continuation_ratio) {
			vector<uint64_t> cohorts;
			auto &cs_vec = StructEntry(entries[bind.cohort_idx]);
			if (!FlatVector::IsNull(cs_vec, row)) {
				auto e = FlatDataMut<list_entry_t>(cs_vec)[row];
				auto &child = ListVector::GetEntry(cs_vec);
				if (ListHasNullElement(FlatValidityMut(child), e)) {
					throw InvalidInputException("ordinal_predict: the model's cohort_specific must not hold NULLs");
				}
				auto d = FlatDataMut<int64_t>(child);
				for (idx_t j = 0; j < e.length; j++) {
					if (d[e.offset + j] < 1) {
						throw InvalidInputException("ordinal_predict: cohort_specific holds 1-based columns, got %lld",
						                            (long long)d[e.offset + j]);
					}
					cohorts.push_back(static_cast<uint64_t>(d[e.offset + j] - 1));
				}
			}
			ok = anofox_continuation_ratio_predict(levels.data(), k, coefficients.data(), coefficients.size(),
			                                       cohorts.empty() ? nullptr : cohorts.data(), cohorts.size(),
			                                       x_child_data + x_entry.offset, x_entry.length, proba.data(),
			                                       exceedance.data(), &mean, &best, &error);
		} else {
			auto &icpt_vec = StructEntry(entries[bind.intercepts_idx]);
			if (FlatVector::IsNull(icpt_vec, row)) {
				FlatVector::SetNull(result, row, true);
				continue;
			}
			auto intercepts = ListDoubles(icpt_vec, row, "intercepts");
			if (k < 2 || intercepts.size() + 1 != k) {
				throw InvalidInputException("ordinal_predict: the model has %llu levels and %llu intercepts; it needs "
				                            "one intercept fewer than levels",
				                            (unsigned long long)k, (unsigned long long)intercepts.size());
			}
			uint32_t link = 0;
			if (bind.link_idx != DConstants::INVALID_INDEX) {
				auto &link_vec = StructEntry(entries[bind.link_idx]);
				if (!FlatVector::IsNull(link_vec, row)) {
					auto name = FlatDataMut<string_t>(link_vec)[row].GetString();
					int code = OrdinalLinkCode(name);
					if (code < 0) {
						throw InvalidInputException("ordinal_predict: unknown link '%s' (logistic, probit, loglog, "
						                            "cloglog, cauchit)",
						                            name);
					}
					link = static_cast<uint32_t>(code);
				}
			}
			uint64_t offset_column = 0;
			if (bind.offset_idx != DConstants::INVALID_INDEX) {
				auto &off_vec = StructEntry(entries[bind.offset_idx]);
				if (!FlatVector::IsNull(off_vec, row)) {
					auto c = FlatDataMut<int64_t>(off_vec)[row];
					if (c < 0) {
						throw InvalidInputException("ordinal_predict: offset_column must be a 1-based column, got %lld",
						                            (long long)c);
					}
					offset_column = static_cast<uint64_t>(c);
				}
			}
			ok = anofox_ordinal_predict(link, levels.data(), k, intercepts.data(), coefficients.data(),
			                            coefficients.size(), offset_column, x_child_data + x_entry.offset,
			                            x_entry.length, proba.data(), exceedance.data(), &mean, &best, &error);
		}
		if (!ok) {
			ThrowUnlessDegenerate("ordinal_predict", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		switch (bind.type) {
		case OrdinalPredictType::CLASS:
			FlatDataMut<double>(result)[row] = levels[best];
			break;
		case OrdinalPredictType::MEAN:
			FlatDataMut<double>(result)[row] = mean;
			break;
		case OrdinalPredictType::PROB:
			AppendDoubles(result, row, proba.data(), proba.size());
			break;
		case OrdinalPredictType::EXCEEDANCE:
			AppendDoubles(result, row, exceedance.data(), exceedance.size());
			break;
		}
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding asserts this in debug builds).
static void OrdinalPredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	OrdinalPredictFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// ordinal_fit_predict_by / continuation_ratio_fit_predict_by
//===--------------------------------------------------------------------===//
// The training-row test is SPLIT_IS_TRAIN of fit_predict_macros.cpp: 'train' / 'training'
// in any case; with split omitted (the NULL literal default) every row trains.
static string OrdinalFitPredictBySql(const char *agg) {
	return string(R"(
WITH _anofox_src AS (
    SELECT *,
        coalesce(lower(split::VARCHAR) IN ('train', 'training'), replace(typeof(split), '"', '') = 'NULL') AS _anofox_train
    FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, )") +
	       agg + R"((y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND _anofox_train
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model, _anofox_train),
    ordinal_predict(_anofox_model, x_cols) AS yhat,
    ordinal_predict(_anofox_model, x_cols, {'type': 'mean'}) AS yhat_mean,
    ordinal_predict(_anofox_model, x_cols, {'type': 'prob'}) AS probabilities,
    (y_col IS NOT NULL AND _anofox_train) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)";
}

static void RegisterOrdinalFitAgg(ExtensionLoader &loader, const char *name, bool continuation_ratio,
                                  const char *description, const char *weighted_description,
                                  const vector<string> &examples) {
	AggregateFunctionSet func_set(name);
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	vector<vector<LogicalType>> signatures = {{LogicalType::DOUBLE, dlist},
	                                          {LogicalType::DOUBLE, dlist, LogicalType::ANY},
	                                          {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY}};
	for (auto &args : signatures) {
		func_set.AddFunction(AggregateFunction(
		    name, args, LogicalType::ANY, AggregateFunction::StateSize<OrdinalAggregateState>,
		    ANOFOX_AGG_INIT(OrdinalAggInitialize),
		    ANOFOX_GUARDED_UPDATE(OrdinalAggUpdate, OrdinalAggDestroy, OrdinalAggInitialize), OrdinalAggCombine,
		    ANOFOX_GUARDED_FINALIZE(OrdinalAggFinalize, OrdinalAggDestroy, OrdinalAggInitialize), nullptr,
		    continuation_ratio ? ANOFOX_AGG_BIND(ContinuationRatioAggBind) : ANOFOX_AGG_BIND(OrdinalAggBind),
		    OrdinalAggDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(func_set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	struct {
		vector<string> names;
		vector<LogicalType> types;
		string example;
		const char *text;
	} docs[] = {{{"y", "x"}, {LogicalType::DOUBLE, dlist}, examples[0], description},
	            {{"y", "x", "options"}, {LogicalType::DOUBLE, dlist, LogicalType::ANY}, examples[1], description},
	            {{"y", "x", "weights", "options"},
	             {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY},
	             examples[2],
	             weighted_description}};
	for (auto &doc : docs) {
		FunctionDescription d;
		d.description = doc.text;
		d.examples = {doc.example};
		d.categories = {"regression", "ordinal"};
		d.parameter_names = doc.names;
		d.parameter_types = doc.types;
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

void RegisterOrdinalFunctions(ExtensionLoader &loader) {
	RegisterOrdinalFitAgg(
	    loader, "ordinal_fit_agg", false,
	    "Fits a cumulative link (ordinal regression) model, rms::orm / MASS::polr: P(Y >= y_{j+1} | x) = F(alpha_j + "
	    "x'beta) for the ordered levels y_1 < ... < y_k (the distinct values of y). Returns STRUCT(levels, counts, "
	    "intercepts (alpha_j; polr's zeta_j = -alpha_j), intercept_std_errors, coefficients, std_errors, z_values, "
	    "p_values, log_likelihood, null_log_likelihood, penalized_log_likelihood, lr_chisq, lr_df, lr_p_value, aic, "
	    "po_test_statistic, po_test_df, po_test_p_value (proportional-odds score test), n_levels, n_observations, "
	    "n_features, offset_column, penalty, iterations, converged, model_type, link). Options: link (logistic, "
	    "probit, loglog, cloglog, cauchit), penalty, offset (1-based x column), max_iterations, tolerance, po_test. "
	    "Rows with a NULL y, a NULL x or a NULL / non-finite feature are left out.",
	    "Fits the same cumulative link model with non-negative case (frequency) weights (a numeric third argument "
	    "is the weight); a NULL weight leaves the row out.",
	    {"ordinal_fit_agg(rating, [age, dose])", "ordinal_fit_agg(rating, [age, dose], {'link': 'probit'})",
	     "ordinal_fit_agg(rating, [age, dose], w, {'link': 'logistic'})"});
	RegisterOrdinalFitAgg(
	    loader, "continuation_ratio_fit_agg", true,
	    "Fits the continuation-ratio ordinal model logit P(Y = y_j | Y >= y_j, x) = theta_j + x'gamma (rms "
	    "cr.setup + lrm; at least three levels). Returns STRUCT(levels, coefficients (intercept theta_1, cohort "
	    "effects theta_j - theta_1 for j = 2..k-1, slopes, then each cohort-specific column's slope differences), "
	    "std_errors, z_values, p_values, cohort_specific, log_likelihood, aic, n_levels, n_observations, n_records, "
	    "n_features, iterations, converged, model_type, link). Option: cohort_specific (1-based x columns whose slope "
	    "varies by cohort).",
	    "Fits the same continuation-ratio model with non-negative case weights (a numeric third argument is the "
	    "weight); a NULL weight leaves the row out.",
	    {"continuation_ratio_fit_agg(stage, [age, dose])",
	     "continuation_ratio_fit_agg(stage, [age, dose], {'cohort_specific': [2]})",
	     "continuation_ratio_fit_agg(stage, [age, dose], w, {'cohort_specific': [2]})"});
	{
		ScalarFunctionSet func_set("ordinal_predict");
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE)};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			ScalarFunction func(args, LogicalType::ANY, DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, OrdinalPredictFunction),
			                    ANOFOX_SCALAR_BIND(OrdinalPredictOwnBind));
			MarkFallible(func);
			SetNullHandlingOf(func, FunctionNullHandling::SPECIAL_HANDLING);
			func_set.AddFunction(func);
		}
		CreateScalarFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription desc;
		desc.description =
		    "Predicts one row from an ordinal_fit_agg or continuation_ratio_fit_agg model: the most probable level "
		    "(DOUBLE; polr's predict(type = 'class')), or with {'type': 'prob'} the level probabilities (DOUBLE[] in "
		    "the order of the model's levels), {'type': 'exceedance'} P(Y >= y_{j+1}) for j = 1..k-1 (DOUBLE[]) or "
		    "{'type': 'mean'} the predicted mean sum_j y_j P(Y = y_j) (rms Mean.orm). NULL when the model or x is "
		    "NULL or x holds a NULL. predict(model, x [, options]) does the same.";
		desc.examples = {"ordinal_predict(model, [52, 1.0])", "ordinal_predict(model, [52, 1.0], {'type': 'prob'})"};
		desc.categories = {"regression", "ordinal", "prediction"};
		desc.parameter_names = {"model", "x", "options"};
		desc.parameter_types = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
		info.descriptions.push_back(std::move(desc));
		loader.RegisterFunction(std::move(info));
	}
	struct {
		const char *name;
		const char *agg;
		const char *description;
		const char *example;
	} macros[] = {
	    {"ordinal_fit_predict_by", "ordinal_fit_agg",
	     "Table macro: fits a cumulative link model (ordinal_fit_agg, options as there) per group of source "
	     "(training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns every "
	     "source row with yhat (the most probable level), yhat_mean (the predicted mean), probabilities (DOUBLE[] in "
	     "the order of the model's levels) and is_training appended.",
	     "ordinal_fit_predict_by('ratings', site, rating, [age, dose], options := {'link': 'logistic'})"},
	    {"continuation_ratio_fit_predict_by", "continuation_ratio_fit_agg",
	     "Table macro: fits a continuation-ratio model (continuation_ratio_fit_agg, options as there) per group of "
	     "source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns "
	     "every source row with yhat (the most probable level), yhat_mean (the predicted mean), probabilities "
	     "(DOUBLE[] in the order of the model's levels) and is_training appended.",
	     "continuation_ratio_fit_predict_by('ratings', site, rating, [age, dose])"}};
	for (auto &m : macros) {
		auto macro = CreateContractTableMacro(m.name, {"source", "group_col", "y_col", "x_cols"},
		                                      {{"options", "NULL"}, {"split", "NULL"}}, OrdinalFitPredictBySql(m.agg),
		                                      m.description, m.example, {"regression", "ordinal", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

} // namespace duckdb
