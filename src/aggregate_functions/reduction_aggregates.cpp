// Data reduction and transforms (#320): spearman2_agg, hoeffding_d_agg,
// box_cox_agg, box_cox_transform, box_cox_inverse, varclus_agg and redun_agg.
// The statistics are upstream anofox-regression reduction / transform
// (Hmisc::spearman2, hoeffd, varclus, redun and MASS::boxcox); this file only
// buffers rows, parses options and marshals the result STRUCTs.

#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/canonical_order.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/test_struct.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "anofox_statistics_extension.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

//===--------------------------------------------------------------------===//
// Options
//===--------------------------------------------------------------------===//

//! Visit the entries of a constant MAP / STRUCT option bag (keys lower-cased).
template <class F>
void VisitReductionOptions(const char *fn, const Value &options, F &&visit) {
	if (options.IsNull()) {
		return;
	}
	if (options.type().id() == LogicalTypeId::STRUCT) {
		auto &vals = StructValue::GetChildren(options);
		for (idx_t i = 0; i < vals.size(); i++) {
			visit(StringUtil::Lower(NameString(StructType::GetChildName(options.type(), i))), UnwrapVariant(vals[i]));
		}
	} else if (options.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(options)) {
			auto &kv = StructValue::GetChildren(entry);
			visit(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else {
		throw InvalidInputException("%s: options must be a MAP or STRUCT", fn);
	}
}

void RequireValue(const char *fn, const string &key, const Value &v) {
	if (v.IsNull()) {
		throw InvalidInputException("%s: option '%s' must not be NULL", fn, key);
	}
}

double OptDouble(const char *fn, const string &key, const Value &v) {
	RequireValue(fn, key, v);
	return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
}

bool OptBool(const char *fn, const string &key, const Value &v) {
	RequireValue(fn, key, v);
	return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
}

int64_t OptInt(const char *fn, const string &key, const Value &v) {
	RequireValue(fn, key, v);
	return v.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
}

string OptString(const char *fn, const string &key, const Value &v) {
	RequireValue(fn, key, v);
	return StringUtil::Lower(v.ToString());
}

vector<string> OptStringList(const char *fn, const string &key, const Value &v) {
	RequireValue(fn, key, v);
	if (v.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("%s: option '%s' must be a VARCHAR[]", fn, key);
	}
	vector<string> out;
	for (auto &c : ListValue::GetChildren(v)) {
		auto u = UnwrapVariant(c);
		if (u.IsNull()) {
			throw InvalidInputException("%s: option '%s' must not hold NULLs", fn, key);
		}
		out.push_back(u.ToString());
	}
	return out;
}

vector<double> OptDoubleList(const char *fn, const string &key, const Value &v) {
	RequireValue(fn, key, v);
	if (v.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("%s: option '%s' must be a DOUBLE[]", fn, key);
	}
	vector<double> out;
	for (auto &c : ListValue::GetChildren(v)) {
		auto u = UnwrapVariant(c);
		if (u.IsNull()) {
			throw InvalidInputException("%s: option '%s' must not hold NULLs", fn, key);
		}
		out.push_back(u.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>());
	}
	return out;
}

[[noreturn]] void UnknownOption(const char *fn, const string &key, const char *supported) {
	throw InvalidInputException("%s: unknown option '%s' (supported: %s)", fn, key, supported);
}

void CheckOneOf(const char *fn, const string &key, const string &value, const vector<string> &allowed) {
	if (std::find(allowed.begin(), allowed.end(), value) == allowed.end()) {
		throw InvalidInputException("%s: option '%s' must be one of %s, got '%s'", fn, key,
		                            StringUtil::Join(allowed, ", "), value);
	}
}

//! Variable label: the given feature name, or x1, x2, ...
string Label(const vector<string> &names, idx_t j) {
	return j < names.size() ? names[j] : "x" + std::to_string(j + 1);
}

//===--------------------------------------------------------------------===//
// Shared state: rows of (y, x) values or of a feature vector
//===--------------------------------------------------------------------===//
struct ReductionState {
	vector<double> y;
	vector<double> x;
	vector<string> labels;
	vector<vector<double>> columns;
	idx_t n_features;
	bool initialized;

	ReductionState() : n_features(0), initialized(false) {
	}

	idx_t Rows() const {
		if (!columns.empty()) {
			return columns[0].size();
		}
		return y.size();
	}

	void Reset() {
		y.clear();
		x.clear();
		labels.clear();
		columns.clear();
		n_features = 0;
		initialized = false;
	}
};

struct ReductionStates {
	UnifiedVectorFormat format;
	ReductionStates(Vector &v, idx_t count) {
		v.ToUnifiedFormat(count, format);
	}
	ReductionState &operator[](idx_t i) const {
		return *reinterpret_cast<ReductionState *const *>(format.data)[format.sel->get_index(i)];
	}
};

void ReductionInitialize(data_ptr_t state_p) {
	new (state_p) ReductionState();
}

void ReductionDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	ReductionStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		states[i].~ReductionState();
	}
}

template <class V>
void AppendAll(V &target, const V &source) {
	target.insert(target.end(), source.begin(), source.end());
}

void ReductionCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	ReductionStates sources(source_vector, count);
	ReductionStates targets(target_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &source = sources[i];
		auto &target = targets[i];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y = CombineTake(source.y, aggr_input_data);
			target.x = CombineTake(source.x, aggr_input_data);
			target.labels = CombineTake(source.labels, aggr_input_data);
			target.columns = CombineTake(source.columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("inconsistent feature count: expected %llu, got %llu",
			                            (unsigned long long)target.n_features, (unsigned long long)source.n_features);
		}
		AppendAll(target.y, source.y);
		AppendAll(target.x, source.x);
		AppendAll(target.labels, source.labels);
		for (idx_t j = 0; j < target.columns.size(); j++) {
			AppendAll(target.columns[j], source.columns[j]);
		}
	}
}

//! (a DOUBLE, b DOUBLE): rows with a NULL or non-finite value are skipped.
void PairUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat a_data, b_data;
	inputs[0].ToUnifiedFormat(count, a_data);
	inputs[1].ToUnifiedFormat(count, b_data);
	auto a = UnifiedVectorFormat::GetData<double>(a_data);
	auto b = UnifiedVectorFormat::GetData<double>(b_data);
	ReductionStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		state.initialized = true;
		auto ai = a_data.sel->get_index(i);
		auto bi = b_data.sel->get_index(i);
		if (!a_data.validity.RowIsValid(ai) || !b_data.validity.RowIsValid(bi) || !std::isfinite(a[ai]) ||
		    !std::isfinite(b[bi])) {
			continue;
		}
		state.y.push_back(a[ai]);
		state.x.push_back(b[bi]);
	}
}

//! (y DOUBLE, x VARCHAR): rows with a NULL x, a NULL or non-finite y are skipped.
void LabelUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, l_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, l_data);
	auto y = UnifiedVectorFormat::GetData<double>(y_data);
	auto l = UnifiedVectorFormat::GetData<string_t>(l_data);
	ReductionStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		state.initialized = true;
		auto yi = y_data.sel->get_index(i);
		auto li = l_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(yi) || !l_data.validity.RowIsValid(li) || !std::isfinite(y[yi])) {
			continue;
		}
		state.y.push_back(y[yi]);
		state.labels.push_back(l[li].GetString());
	}
}

//! Append the feature vector of row i to the state's columns; false when the
//! row is skipped (NULL list, NULL or non-finite element).
bool AppendFeatureRow(ReductionState &state, const UnifiedVectorFormat &x_data, const list_entry_t *lists,
                      const double *child, const ValidityMask &child_validity, idx_t i, const char *fn) {
	auto xi = x_data.sel->get_index(i);
	if (!x_data.validity.RowIsValid(xi)) {
		return false;
	}
	auto entry = lists[xi];
	if (ListHasNullElement(child_validity, entry)) {
		return false;
	}
	for (idx_t j = 0; j < entry.length; j++) {
		if (!std::isfinite(child[entry.offset + j])) {
			return false;
		}
	}
	if (!state.initialized) {
		state.n_features = entry.length;
		state.columns.resize(entry.length);
		state.initialized = true;
	}
	if (entry.length != state.n_features) {
		throw InvalidInputException("%s: inconsistent feature count: expected %llu, got %llu", fn,
		                            (unsigned long long)state.n_features, (unsigned long long)entry.length);
	}
	for (idx_t j = 0; j < entry.length; j++) {
		state.columns[j].push_back(child[entry.offset + j]);
	}
	return true;
}

//! (x DOUBLE[] [, options]): the feature rows.
template <const char *FN>
void FeatureUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat x_data;
	inputs[0].ToUnifiedFormat(count, x_data);
	auto lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child = ListVector::GetEntry(inputs[0]);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	ReductionStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		AppendFeatureRow(states[i], x_data, lists, child_data, child_validity, i, FN);
	}
}

//! (y DOUBLE [, x DOUBLE[]] [, options]): y with its predictor row.
void BoxCoxUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                  idx_t count) {
	static const char *fn = "box_cox_agg";
	UnifiedVectorFormat y_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	auto y = UnifiedVectorFormat::GetData<double>(y_data);
	const bool with_x = input_count >= 2 && inputs[1].GetType().id() == LogicalTypeId::LIST;
	UnifiedVectorFormat x_data;
	const list_entry_t *lists = nullptr;
	const double *child_data = nullptr;
	const ValidityMask *child_validity = nullptr;
	if (with_x) {
		inputs[1].ToUnifiedFormat(count, x_data);
		lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
		auto &child = ListVector::GetEntry(inputs[1]);
		child_data = FlatDataMut<double>(child);
		child_validity = &FlatValidityMut(child);
	}
	ReductionStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		auto yi = y_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(yi) || !std::isfinite(y[yi])) {
			continue;
		}
		if (with_x) {
			if (!AppendFeatureRow(state, x_data, lists, child_data, *child_validity, i, fn)) {
				continue;
			}
		} else {
			state.initialized = true;
		}
		state.y.push_back(y[yi]);
	}
}

//===--------------------------------------------------------------------===//
// List writers
//===--------------------------------------------------------------------===//
template <class T>
void AppendList(Vector &list_vec, idx_t row, const T *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	auto child = FlatDataMut<T>(ListVector::GetEntry(list_vec));
	for (idx_t i = 0; i < len; i++) {
		child[offset + i] = data[i];
	}
	ListVector::SetListSize(list_vec, offset + len);
	ListVector::GetData(list_vec)[row] = {offset, len};
}

//! A DOUBLE list with NaN written as NULL.
void AppendDoubleListNullNaN(Vector &list_vec, idx_t row, const double *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	AppendList(list_vec, row, data, len);
	auto &child = ListVector::GetEntry(list_vec);
	for (idx_t i = 0; i < len; i++) {
		if (std::isnan(data[i])) {
			FlatVector::SetNull(child, offset + i, true);
		}
	}
}

void AppendStringList(Vector &list_vec, idx_t row, const vector<string> &values) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + values.size());
	auto &child = ListVector::GetEntry(list_vec);
	auto data = FlatDataMut<string_t>(child);
	for (idx_t i = 0; i < values.size(); i++) {
		data[offset + i] = StringVector::AddString(child, values[i]);
	}
	ListVector::SetListSize(list_vec, offset + values.size());
	ListVector::GetData(list_vec)[row] = {offset, values.size()};
}

//! LIST(LIST(T)) of `rows` lists of `cols` values each, row-major in data.
template <class T>
void AppendMatrix(Vector &outer, idx_t row, const T *data, idx_t rows, idx_t cols) {
	auto outer_offset = ListVector::GetListSize(outer);
	ListVector::Reserve(outer, outer_offset + rows);
	auto &inner = ListVector::GetEntry(outer);
	for (idx_t r = 0; r < rows; r++) {
		AppendList(inner, outer_offset + r, data + r * cols, cols);
	}
	ListVector::SetListSize(outer, outer_offset + rows);
	ListVector::GetData(outer)[row] = {outer_offset, rows};
}

void SetDouble(Vector &v, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(v, row, true);
	} else {
		FlatDataMut<double>(v)[row] = value;
	}
}

vector<AnofoxDataArray> ColumnArrays(vector<vector<double>> &columns) {
	vector<AnofoxDataArray> arrays;
	for (auto &col : columns) {
		AnofoxDataArray a;
		a.data = col.data();
		a.validity = nullptr;
		a.len = col.size();
		arrays.push_back(a);
	}
	return arrays;
}

Value ConstantOptions(ClientContext &context, vector<unique_ptr<Expression>> &arguments, idx_t index, const char *fn) {
	if (arguments.size() <= index) {
		return Value();
	}
	return EvaluateConstantOptions(context, *arguments[index], fn);
}

//===--------------------------------------------------------------------===//
// spearman2_agg(y, x [, options])
//===--------------------------------------------------------------------===//
struct Spearman2BindData : public FunctionData {
	idx_t p = 1;
	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<Spearman2BindData>();
		copy->p = p;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other) const override {
		return p == other.Cast<Spearman2BindData>().p;
	}
};

LogicalType Spearman2ResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("rho2", LogicalType::DOUBLE));
	c.push_back(make_pair("adjusted_rho2", LogicalType::DOUBLE));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("df2", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("n", LogicalType::BIGINT));
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void WriteSpearman2(Vector &result, idx_t row, const AnofoxSpearman2Result &r) {
	auto &e = StructVector::GetEntries(result);
	idx_t k = 0;
	SetDouble(StructEntry(e[k++]), row, r.rho2);
	SetDouble(StructEntry(e[k++]), row, r.adjusted_rho2);
	SetDouble(StructEntry(e[k++]), row, r.statistic);
	SetDouble(StructEntry(e[k++]), row, r.df1);
	SetDouble(StructEntry(e[k++]), row, r.df2);
	SetDouble(StructEntry(e[k++]), row, r.p_value);
	FlatDataMut<int64_t>(StructEntry(e[k++]))[row] = r.n;
	WriteTestResultFields(result, row, k, r.unified);
}

void Spearman2Finalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                       idx_t offset) {
	ReductionStates states(state_vector, count);
	auto &bind = aggr_input_data.bind_data->Cast<Spearman2BindData>();
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (state.y.size() < 3) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Key(state.y).Key(state.x).Sort();
		AnofoxSpearman2Result r;
		AnofoxError error;
		if (!anofox_spearman2(state.y.data(), state.x.data(), state.y.size(), bind.p, &r, &error)) {
			ThrowUnlessDegenerate("spearman2_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteSpearman2(result, row, r);
		state.Reset();
	}
}

void Spearman2CategoricalFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                  idx_t offset) {
	ReductionStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (state.y.size() < 3) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Level codes by sorted label: independent of the row order.
		std::map<string, double> levels;
		for (auto &l : state.labels) {
			levels.emplace(l, 0.0);
		}
		double next = 0.0;
		for (auto &kv : levels) {
			kv.second = next++;
		}
		vector<double> code_keys;
		code_keys.reserve(state.labels.size());
		for (auto &l : state.labels) {
			code_keys.push_back(levels[l]);
		}
		CanonicalRows().Key(state.y).Key(code_keys).Sort();
		vector<uint64_t> codes(code_keys.begin(), code_keys.end());
		AnofoxSpearman2Result r;
		AnofoxError error;
		if (!anofox_spearman2_categorical(state.y.data(), codes.data(), codes.size(), &r, &error)) {
			ThrowUnlessDegenerate("spearman2_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteSpearman2(result, row, r);
		state.Reset();
	}
}

unique_ptr<FunctionData> Spearman2Bind(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	static const char *fn = "spearman2_agg";
	auto bind = make_uniq<Spearman2BindData>();
	VisitReductionOptions(fn, ConstantOptions(context, arguments, 2, fn), [&](const string &key, const Value &v) {
		if (key == "p") {
			auto p = OptInt(fn, key, v);
			if (p != 1 && p != 2) {
				throw InvalidInputException("%s: option 'p' must be 1 or 2, got %lld", fn, (long long)p);
			}
			bind->p = static_cast<idx_t>(p);
		} else {
			UnknownOption(fn, key, "p");
		}
	});
	SetReturnTypeOf(function, Spearman2ResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// hoeffding_d_agg(x, y)
//===--------------------------------------------------------------------===//
LogicalType HoeffdingResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("d", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("n", LogicalType::BIGINT));
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void HoeffdingFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	ReductionStates states(state_vector, count);
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (state.y.size() < 5) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// state.y holds x, state.x holds y (argument order x, y).
		CanonicalRows().Key(state.y).Key(state.x).Sort();
		AnofoxHoeffdingResult r;
		AnofoxError error;
		if (!anofox_hoeffding_d(state.y.data(), state.x.data(), state.y.size(), &r, &error)) {
			ThrowUnlessDegenerate("hoeffding_d_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		SetDouble(StructEntry(e[k++]), row, r.d);
		SetDouble(StructEntry(e[k++]), row, r.p_value);
		FlatDataMut<int64_t>(StructEntry(e[k++]))[row] = r.n;
		WriteTestResultFields(result, row, k, r.unified);
		state.Reset();
	}
}

unique_ptr<FunctionData> HoeffdingBind(ClientContext &, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &) {
	SetReturnTypeOf(function, HoeffdingResultType());
	PostHogTelemetry::Instance().RecordFunctionCall("hoeffding_d_agg");
	return nullptr;
}

//===--------------------------------------------------------------------===//
// box_cox_agg(y [, x DOUBLE[]] [, options])
//===--------------------------------------------------------------------===//
struct BoxCoxBindData : public FunctionData {
	vector<double> lambdas;
	double eps = kNaN;
	bool fit_intercept = true;
	double confidence_level = 0.95;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<BoxCoxBindData>();
		copy->lambdas = lambdas;
		copy->eps = eps;
		copy->fit_intercept = fit_intercept;
		copy->confidence_level = confidence_level;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<BoxCoxBindData>();
		return lambdas == o.lambdas && (eps == o.eps || (std::isnan(eps) && std::isnan(o.eps))) &&
		       fit_intercept == o.fit_intercept && confidence_level == o.confidence_level;
	}
};

LogicalType BoxCoxResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("lambda_hat", LogicalType::DOUBLE));
	c.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
	c.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
	c.push_back(make_pair("confidence_level", LogicalType::DOUBLE));
	c.push_back(make_pair("max_log_likelihood", LogicalType::DOUBLE));
	c.push_back(make_pair("lambdas", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("log_likelihood", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

void BoxCoxFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                    idx_t offset) {
	ReductionStates states(state_vector, count);
	auto &bind = aggr_input_data.bind_data->Cast<BoxCoxBindData>();
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (state.y.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		CanonicalRows().Keys(state.columns).Key(state.y).Sort();
		auto arrays = ColumnArrays(state.columns);
		AnofoxBoxCoxOptions opts;
		opts.lambdas = bind.lambdas.empty() ? nullptr : bind.lambdas.data();
		opts.n_lambdas = bind.lambdas.size();
		opts.eps = bind.eps;
		opts.fit_intercept = bind.fit_intercept;
		opts.confidence_level = bind.confidence_level;
		AnofoxBoxCoxResult r;
		AnofoxError error;
		if (!anofox_box_cox(state.y.data(), state.y.size(), arrays.empty() ? nullptr : arrays.data(), arrays.size(),
		                    opts, &r, &error)) {
			ThrowUnlessDegenerate("box_cox_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		SetDouble(StructEntry(e[k++]), row, r.lambda);
		SetDouble(StructEntry(e[k++]), row, r.ci_lower);
		SetDouble(StructEntry(e[k++]), row, r.ci_upper);
		SetDouble(StructEntry(e[k++]), row, r.confidence_level);
		SetDouble(StructEntry(e[k++]), row, r.max_log_likelihood);
		AppendList(StructEntry(e[k++]), row, r.lambdas, r.n_lambdas);
		AppendList(StructEntry(e[k++]), row, r.log_likelihood, r.n_lambdas);
		FlatDataMut<int64_t>(StructEntry(e[k++]))[row] = static_cast<int64_t>(r.n_observations);
		anofox_free_box_cox_result(&r);
		state.Reset();
	}
}

unique_ptr<FunctionData> BoxCoxBind(ClientContext &context, AggregateBindFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	static const char *fn = "box_cox_agg";
	auto bind = make_uniq<BoxCoxBindData>();
	if (arguments.size() >= 3) {
		VisitReductionOptions(
		    fn, EvaluateConstantOptions(context, *arguments[2], fn), [&](const string &key, const Value &v) {
			    if (key == "lambdas") {
				    bind->lambdas = OptDoubleList(fn, key, v);
				    if (bind->lambdas.empty()) {
					    throw InvalidInputException("%s: option 'lambdas' must not be empty", fn);
				    }
			    } else if (key == "eps") {
				    bind->eps = OptDouble(fn, key, v);
				    if (!(bind->eps > 0.0)) {
					    throw InvalidInputException("%s: option 'eps' must be positive", fn);
				    }
			    } else if (key == "fit_intercept") {
				    bind->fit_intercept = OptBool(fn, key, v);
			    } else if (key == "confidence_level") {
				    bind->confidence_level = OptDouble(fn, key, v);
				    if (!(bind->confidence_level > 0.0 && bind->confidence_level < 1.0)) {
					    throw InvalidInputException("%s: option 'confidence_level' must be in (0, 1)", fn);
				    }
			    } else {
				    UnknownOption(fn, key, "lambdas, eps, fit_intercept, confidence_level");
			    }
		    });
	}
	SetReturnTypeOf(function, BoxCoxResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// box_cox_transform(y, lambda) / box_cox_inverse(z, lambda)
//===--------------------------------------------------------------------===//
template <bool INVERSE>
void BoxCoxScalarImpl(DataChunk &args, ExpressionState &, Vector &result) {
	const char *fn = INVERSE ? "box_cox_inverse" : "box_cox_transform";
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	idx_t count = args.size();
	UnifiedVectorFormat v_data, l_data;
	args.data[0].ToUnifiedFormat(count, v_data);
	args.data[1].ToUnifiedFormat(count, l_data);
	auto v = UnifiedVectorFormat::GetData<double>(v_data);
	auto l = UnifiedVectorFormat::GetData<double>(l_data);
	auto out = FlatDataMut<double>(result);
	for (idx_t row = 0; row < count; row++) {
		auto vi = v_data.sel->get_index(row);
		auto li = l_data.sel->get_index(row);
		if (!v_data.validity.RowIsValid(vi) || !l_data.validity.RowIsValid(li)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (!std::isfinite(l[li])) {
			throw InvalidInputException("%s: lambda must be finite", fn);
		}
		if (!INVERSE && !(v[vi] > 0.0)) {
			throw InvalidInputException("%s: y must be positive, got %g", fn, v[vi]);
		}
		double value = INVERSE ? anofox_box_cox_inverse(v[vi], l[li]) : anofox_box_cox_transform(v[vi], l[li]);
		// Outside the range of the transform (lambda z + 1 <= 0): no inverse. (pow would
		// still return a value for some lambda, e.g. (-0.5)^2, that no y maps to.)
		if (std::isnan(value) || (INVERSE && l[li] != 0.0 && !(l[li] * v[vi] + 1.0 > 0.0))) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		out[row] = value;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

template <bool INVERSE>
void BoxCoxScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	BoxCoxScalarImpl<INVERSE>(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// varclus_agg(x DOUBLE[] [, options])
//===--------------------------------------------------------------------===//
const char kVarclus[] = "varclus_agg";
const char kRedun[] = "redun_agg";

struct VarclusBindData : public FunctionData {
	string similarity = "spearman";
	string trans = "square";
	string method = "complete";
	double cut_height = kNaN;
	vector<string> names;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<VarclusBindData>();
		copy->similarity = similarity;
		copy->trans = trans;
		copy->method = method;
		copy->cut_height = cut_height;
		copy->names = names;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<VarclusBindData>();
		return similarity == o.similarity && trans == o.trans && method == o.method && names == o.names &&
		       (cut_height == o.cut_height || (std::isnan(cut_height) && std::isnan(o.cut_height)));
	}
};

LogicalType VarclusResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("variables", LogicalType::LIST(LogicalType::VARCHAR)));
	c.push_back(make_pair("similarity_matrix", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))));
	c.push_back(make_pair("merges", LogicalType::LIST(LogicalType::LIST(LogicalType::BIGINT))));
	c.push_back(make_pair("height", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("dendrogram_order", LogicalType::LIST(LogicalType::BIGINT)));
	c.push_back(make_pair("clusters", LogicalType::LIST(LogicalType::BIGINT)));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

void CheckNames(const char *fn, const vector<string> &names, idx_t p) {
	if (!names.empty() && names.size() != p) {
		throw InvalidInputException("%s: feature_names has %llu entries but x has %llu features", fn,
		                            (unsigned long long)names.size(), (unsigned long long)p);
	}
}

void VarclusFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                     idx_t offset) {
	ReductionStates states(state_vector, count);
	auto &bind = aggr_input_data.bind_data->Cast<VarclusBindData>();
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (state.columns.empty() || state.Rows() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t p = state.n_features;
		CheckNames(kVarclus, bind.names, p);
		CanonicalRows().Keys(state.columns).Sort();
		auto arrays = ColumnArrays(state.columns);
		AnofoxVarclusResult r;
		AnofoxError error;
		if (!anofox_varclus(arrays.data(), arrays.size(), bind.similarity.c_str(), bind.trans.c_str(),
		                    bind.method.c_str(), bind.cut_height, &r, &error)) {
			ThrowUnlessDegenerate(kVarclus, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		vector<string> labels;
		for (idx_t j = 0; j < p; j++) {
			labels.push_back(Label(bind.names, j));
		}
		idx_t k = 0;
		AppendStringList(StructEntry(e[k++]), row, labels);
		AppendMatrix(StructEntry(e[k++]), row, r.similarity, p, p);
		AppendMatrix(StructEntry(e[k++]), row, r.merge, p - 1, 2);
		AppendList(StructEntry(e[k++]), row, r.height, p - 1);
		AppendList(StructEntry(e[k++]), row, r.order, p);
		auto &clusters = StructEntry(e[k++]);
		if (r.clusters) {
			AppendList(clusters, row, r.clusters, p);
		} else {
			FlatVector::SetNull(clusters, row, true);
		}
		FlatDataMut<int64_t>(StructEntry(e[k++]))[row] = static_cast<int64_t>(r.n_observations);
		anofox_free_varclus_result(&r);
		state.Reset();
	}
}

unique_ptr<FunctionData> VarclusBind(ClientContext &context, AggregateBindFunction &function,
                                     vector<unique_ptr<Expression>> &arguments) {
	const char *fn = kVarclus;
	auto bind = make_uniq<VarclusBindData>();
	VisitReductionOptions(fn, ConstantOptions(context, arguments, 1, fn), [&](const string &key, const Value &v) {
		if (key == "similarity") {
			bind->similarity = OptString(fn, key, v);
			CheckOneOf(fn, key, bind->similarity, {"spearman", "pearson", "hoeffding"});
		} else if (key == "trans" || key == "transform") {
			bind->trans = OptString(fn, key, v);
			CheckOneOf(fn, key, bind->trans, {"square", "abs", "none"});
		} else if (key == "method" || key == "linkage") {
			bind->method = OptString(fn, key, v);
			CheckOneOf(fn, key, bind->method,
			           {"complete", "single", "average", "mcquitty", "median", "centroid", "ward.d", "ward.d2"});
		} else if (key == "cut_height") {
			bind->cut_height = OptDouble(fn, key, v);
			if (!std::isfinite(bind->cut_height)) {
				throw InvalidInputException("%s: option 'cut_height' must be finite", fn);
			}
		} else if (key == "feature_names") {
			bind->names = OptStringList(fn, key, v);
		} else {
			UnknownOption(fn, key, "similarity, trans, method, cut_height, feature_names");
		}
	});
	SetReturnTypeOf(function, VarclusResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// redun_agg(x DOUBLE[] [, options])
//===--------------------------------------------------------------------===//
struct RedunBindData : public FunctionData {
	AnofoxRedunOptions options {kNaN, -1, -1, -1, -1, -1, -1};
	vector<string> types;
	vector<string> names;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<RedunBindData>();
		copy->options = options;
		copy->types = types;
		copy->names = names;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<RedunBindData>();
		auto &a = options;
		auto &b = o.options;
		return (a.r2 == b.r2 || (std::isnan(a.r2) && std::isnan(b.r2))) && a.nk == b.nk && a.adjusted == b.adjusted &&
		       a.tlinear == b.tlinear && a.rank == b.rank && a.qrank == b.qrank && a.allcat == b.allcat &&
		       types == o.types && names == o.names;
	}
};

LogicalType RedunResultType() {
	auto names = LogicalType::LIST(LogicalType::VARCHAR);
	child_list_t<LogicalType> c;
	c.push_back(make_pair("variables", names));
	c.push_back(make_pair("kept", names));
	c.push_back(make_pair("removed", names));
	c.push_back(make_pair("r_squared", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("r_squared_later", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))));
	c.push_back(make_pair("r_squared_first", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("too_few", names));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

vector<string> LabelsOf(const vector<string> &names, const int64_t *one_based, idx_t len) {
	vector<string> out;
	for (idx_t i = 0; i < len; i++) {
		out.push_back(Label(names, static_cast<idx_t>(one_based[i] - 1)));
	}
	return out;
}

void RedunFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                   idx_t offset) {
	ReductionStates states(state_vector, count);
	auto &bind = aggr_input_data.bind_data->Cast<RedunBindData>();
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (state.columns.empty() || state.Rows() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t p = state.n_features;
		CheckNames(kRedun, bind.names, p);
		if (!bind.types.empty() && bind.types.size() != p) {
			throw InvalidInputException("%s: types has %llu entries but x has %llu features", kRedun,
			                            (unsigned long long)bind.types.size(), (unsigned long long)p);
		}
		CanonicalRows().Keys(state.columns).Sort();
		auto arrays = ColumnArrays(state.columns);
		vector<const char *> kinds;
		for (auto &t : bind.types) {
			kinds.push_back(t.c_str());
		}
		AnofoxRedunResult r;
		AnofoxError error;
		if (!anofox_redun(arrays.data(), arrays.size(), kinds.empty() ? nullptr : kinds.data(), bind.options, &r,
		                  &error)) {
			ThrowUnlessDegenerate(kRedun, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		vector<string> labels;
		for (idx_t j = 0; j < p; j++) {
			labels.push_back(Label(bind.names, j));
		}
		// r_squared_first aligned with the variables, NULL for the constant ones.
		vector<double> first(p, kNaN);
		for (idx_t a = 0; a < r.n_analysed; a++) {
			first[static_cast<idx_t>(r.analysed[a] - 1)] = r.r_squared_first[a];
		}
		idx_t k = 0;
		AppendStringList(StructEntry(e[k++]), row, labels);
		AppendStringList(StructEntry(e[k++]), row, LabelsOf(bind.names, r.kept, r.n_kept));
		AppendStringList(StructEntry(e[k++]), row, LabelsOf(bind.names, r.removed, r.n_removed));
		AppendList(StructEntry(e[k++]), row, r.r_squared, r.n_removed);
		// r_squared_later: removal k (from the second) has k values.
		auto &later = StructEntry(e[k++]);
		idx_t n_later = r.n_removed > 0 ? r.n_removed - 1 : 0;
		auto outer_offset = ListVector::GetListSize(later);
		ListVector::Reserve(later, outer_offset + n_later);
		auto &inner = ListVector::GetEntry(later);
		idx_t pos = 0;
		for (idx_t m = 0; m < n_later; m++) {
			AppendList(inner, outer_offset + m, r.r_squared_later + pos, m + 1);
			pos += m + 1;
		}
		ListVector::SetListSize(later, outer_offset + n_later);
		ListVector::GetData(later)[row] = {outer_offset, n_later};
		AppendDoubleListNullNaN(StructEntry(e[k++]), row, first.data(), p);
		AppendStringList(StructEntry(e[k++]), row, LabelsOf(bind.names, r.too_few, r.n_too_few));
		FlatDataMut<int64_t>(StructEntry(e[k++]))[row] = static_cast<int64_t>(r.n_observations);
		anofox_free_redun_result(&r);
		state.Reset();
	}
}

unique_ptr<FunctionData> RedunBind(ClientContext &context, AggregateBindFunction &function,
                                   vector<unique_ptr<Expression>> &arguments) {
	const char *fn = kRedun;
	auto bind = make_uniq<RedunBindData>();
	auto &o = bind->options;
	VisitReductionOptions(fn, ConstantOptions(context, arguments, 1, fn), [&](const string &key, const Value &v) {
		if (key == "r2") {
			o.r2 = OptDouble(fn, key, v);
			if (!(o.r2 > 0.0 && o.r2 < 1.0)) {
				throw InvalidInputException("%s: option 'r2' must be in (0, 1)", fn);
			}
		} else if (key == "type") {
			auto t = OptString(fn, key, v);
			CheckOneOf(fn, key, t, {"ordinary", "adjusted"});
			o.adjusted = t == "adjusted" ? 1 : 0;
		} else if (key == "nk") {
			o.nk = OptInt(fn, key, v);
			if (o.nk != 0 && (o.nk < 3 || o.nk > 7)) {
				throw InvalidInputException("%s: option 'nk' must be 0 (linear) or 3 to 7", fn);
			}
		} else if (key == "tlinear") {
			o.tlinear = OptBool(fn, key, v) ? 1 : 0;
		} else if (key == "rank") {
			o.rank = OptBool(fn, key, v) ? 1 : 0;
		} else if (key == "qrank") {
			o.qrank = OptBool(fn, key, v) ? 1 : 0;
		} else if (key == "allcat") {
			o.allcat = OptBool(fn, key, v) ? 1 : 0;
		} else if (key == "types") {
			bind->types = OptStringList(fn, key, v);
			for (auto &t : bind->types) {
				t = StringUtil::Lower(t);
				CheckOneOf(fn, key, t, {"spline", "numeric", "s", "linear", "l", "categorical", "c"});
			}
		} else if (key == "feature_names") {
			bind->names = OptStringList(fn, key, v);
		} else {
			UnknownOption(fn, key, "r2, type, nk, tlinear, rank, qrank, allcat, types, feature_names");
		}
	});
	SetReturnTypeOf(function, RedunResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
AggregateFunction MakeAggregate(const char *name, vector<LogicalType> args, aggregate_update_t update,
                                aggregate_finalize_t finalize, bind_aggregate_function_t bind) {
	return AggregateFunction(name, std::move(args), LogicalType::ANY, AggregateFunction::StateSize<ReductionState>,
	                         ANOFOX_AGG_INIT(ReductionInitialize), update, ReductionCombine, finalize, nullptr, bind,
	                         ReductionDestroy);
}

void AddDescription(CreateAggregateFunctionInfo &info, const string &description, const string &example,
                    const vector<string> &names, const vector<LogicalType> &types) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = {"data-reduction"};
	d.parameter_names = names;
	d.parameter_types = types;
	info.descriptions.push_back(std::move(d));
}

void RegisterSet(ExtensionLoader &loader, AggregateFunctionSet &set,
                 const std::function<void(CreateAggregateFunctionInfo &)> &describe) {
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	describe(info);
	loader.RegisterFunction(std::move(info));
}

} // namespace

#define REDUCTION_UPDATE(UPD)   ANOFOX_GUARDED_UPDATE(UPD, ReductionDestroy, ReductionInitialize)
#define REDUCTION_FINALIZE(FIN) ANOFOX_GUARDED_FINALIZE(FIN, ReductionDestroy, ReductionInitialize)

void RegisterReductionFunctions(ExtensionLoader &loader) {
	const auto D = LogicalType::DOUBLE;
	const auto V = LogicalType::VARCHAR;
	const auto L = LogicalType::LIST(LogicalType::DOUBLE);
	const auto A = LogicalType::ANY;

	{
		AggregateFunctionSet set("spearman2_agg");
		for (bool opts : {false, true}) {
			vector<LogicalType> num = {D, D};
			vector<LogicalType> cat = {D, V};
			if (opts) {
				num.push_back(A);
				cat.push_back(A);
			}
			set.AddFunction(MakeAggregate("spearman2_agg", num, REDUCTION_UPDATE(PairUpdate),
			                              REDUCTION_FINALIZE(Spearman2Finalize), ANOFOX_AGG_BIND(Spearman2Bind)));
			set.AddFunction(MakeAggregate("spearman2_agg", cat, REDUCTION_UPDATE(LabelUpdate),
			                              REDUCTION_FINALIZE(Spearman2CategoricalFinalize),
			                              ANOFOX_AGG_BIND(Spearman2Bind)));
		}
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			AddDescription(info,
			               "Generalized Spearman rho^2 of a response y with a predictor x (Hmisc::spearman2): the R^2 "
			               "of rank(y) on rank(x) (p = 1, the squared Spearman correlation) or on rank(x) and its "
			               "square (p = 2, detects non-monotone association). Returns rho2, adjusted_rho2, the F "
			               "statistic with df1 / df2, p_value, n and the unified test fields. Rows with a NULL or "
			               "non-finite value are skipped.",
			               "spearman2_agg(y, x)", {"y", "x"}, {D, D});
			AddDescription(info, "Generalized Spearman rho^2 with options: p (1 or 2, default 1).",
			               "spearman2_agg(y, x, {'p': 2})", {"y", "x", "options"}, {D, D, A});
			AddDescription(info,
			               "Generalized Spearman rho^2 of y with a categorical x (Hmisc::spearman2(factor(x), y)): "
			               "the R^2 of rank(y) on the level dummies, df1 = levels - 1.",
			               "spearman2_agg(y, grp)", {"y", "x"}, {D, V});
			AddDescription(info, "Generalized Spearman rho^2 of y with a categorical x, with options (p is ignored).",
			               "spearman2_agg(y, grp, {'p': 1})", {"y", "x", "options"}, {D, V, A});
		});
	}
	{
		AggregateFunctionSet set("hoeffding_d_agg");
		set.AddFunction(MakeAggregate("hoeffding_d_agg", {D, D}, REDUCTION_UPDATE(PairUpdate),
		                              REDUCTION_FINALIZE(HoeffdingFinalize), ANOFOX_AGG_BIND(HoeffdingBind)));
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			AddDescription(info,
			               "Hoeffding's D of x and y (Hmisc::hoeffd): a rank measure of dependence of any form, "
			               "scaled by 30 (range -0.5 to 1), with its asymptotic P-value. Returns d, p_value, n and "
			               "the unified test fields. Needs five or more complete rows; O(n^2).",
			               "hoeffding_d_agg(x, y)", {"x", "y"}, {D, D});
		});
	}
	{
		AggregateFunctionSet set("box_cox_agg");
		set.AddFunction(MakeAggregate("box_cox_agg", {D}, REDUCTION_UPDATE(BoxCoxUpdate),
		                              REDUCTION_FINALIZE(BoxCoxFinalize), ANOFOX_AGG_BIND(BoxCoxBind)));
		set.AddFunction(MakeAggregate("box_cox_agg", {D, L}, REDUCTION_UPDATE(BoxCoxUpdate),
		                              REDUCTION_FINALIZE(BoxCoxFinalize), ANOFOX_AGG_BIND(BoxCoxBind)));
		set.AddFunction(MakeAggregate("box_cox_agg", {D, L, A}, REDUCTION_UPDATE(BoxCoxUpdate),
		                              REDUCTION_FINALIZE(BoxCoxFinalize), ANOFOX_AGG_BIND(BoxCoxBind)));
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			AddDescription(info,
			               "Box-Cox profile likelihood of the power lambda for the intercept-only linear model of a "
			               "positive y (MASS::boxcox(y ~ 1)): the maximising lambda with its likelihood-ratio "
			               "interval and the profile on the grid seq(-2, 2, 0.1).",
			               "box_cox_agg(y)", {"y"}, {D});
			AddDescription(info,
			               "Box-Cox profile likelihood of lambda for the linear model of a positive y on x "
			               "(MASS::boxcox(lm(y ~ x))). Returns lambda, ci_lower, ci_upper (NULL when not reached "
			               "within the grid), confidence_level, max_log_likelihood, lambdas, log_likelihood and "
			               "n_observations. Rows with a NULL or non-finite value are skipped.",
			               "box_cox_agg(y, [x1, x2])", {"y", "x"}, {D, L});
			AddDescription(info,
			               "Box-Cox profile with options: lambdas (grid, default seq(-2, 2, 0.1)), "
			               "confidence_level (0.95), fit_intercept (true), eps (1/50).",
			               "box_cox_agg(y, [x1], {'confidence_level': 0.9})", {"y", "x", "options"}, {D, L, A});
		});
	}
	{
		ScalarFunctionSet set("box_cox_transform");
		ScalarFunction f({D, D}, D, BoxCoxScalar<false>);
		MarkFallible(f);
		set.AddFunction(f);
		CreateScalarFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = "Box-Cox transform of a positive y: (y^lambda - 1) / lambda, log(y) at lambda = 0.";
		d.examples = {"box_cox_transform(y, 0.5)"};
		d.categories = {"data-reduction"};
		d.parameter_names = {"y", "lambda"};
		d.parameter_types = {D, D};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		ScalarFunctionSet set("box_cox_inverse");
		ScalarFunction f({D, D}, D, BoxCoxScalar<true>);
		MarkFallible(f);
		set.AddFunction(f);
		CreateScalarFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = "Inverse Box-Cox transform: (lambda z + 1)^(1 / lambda), exp(z) at lambda = 0; NULL "
		                "where lambda z + 1 <= 0 has no real inverse.";
		d.examples = {"box_cox_inverse(z, 0.5)"};
		d.categories = {"data-reduction"};
		d.parameter_names = {"z", "lambda"};
		d.parameter_types = {D, D};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set("varclus_agg");
		set.AddFunction(MakeAggregate("varclus_agg", {L}, REDUCTION_UPDATE(FeatureUpdate<kVarclus>),
		                              REDUCTION_FINALIZE(VarclusFinalize), ANOFOX_AGG_BIND(VarclusBind)));
		set.AddFunction(MakeAggregate("varclus_agg", {L, A}, REDUCTION_UPDATE(FeatureUpdate<kVarclus>),
		                              REDUCTION_FINALIZE(VarclusFinalize), ANOFOX_AGG_BIND(VarclusBind)));
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			AddDescription(info,
			               "Variable clustering (Hmisc::varclus): the similarity of every pair of features (squared "
			               "Spearman correlation by default) clustered hierarchically on 1 - similarity (complete "
			               "linkage). Returns variables, similarity, merges (R's hclust convention: -j is variable j, "
			               "k the cluster of step k), height, dendrogram_order, clusters (NULL without cut_height) "
			               "and n_observations. Rows with a NULL or non-finite element are skipped.",
			               "varclus_agg([x1, x2, x3])", {"x"}, {L});
			AddDescription(info,
			               "Variable clustering with options: similarity (spearman, pearson, hoeffding), trans "
			               "(square, abs, none), method (complete, single, average, mcquitty, median, centroid, "
			               "ward.d, ward.d2), cut_height (cluster labels of the tree cut there), feature_names.",
			               "varclus_agg([x1, x2, x3], {'cut_height': 0.5})", {"x", "options"}, {L, A});
		});
	}
	{
		AggregateFunctionSet set("redun_agg");
		set.AddFunction(MakeAggregate("redun_agg", {L}, REDUCTION_UPDATE(FeatureUpdate<kRedun>),
		                              REDUCTION_FINALIZE(RedunFinalize), ANOFOX_AGG_BIND(RedunBind)));
		set.AddFunction(MakeAggregate("redun_agg", {L, A}, REDUCTION_UPDATE(FeatureUpdate<kRedun>),
		                              REDUCTION_FINALIZE(RedunFinalize), ANOFOX_AGG_BIND(RedunBind)));
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			AddDescription(info,
			               "Redundancy analysis (Hmisc::redun): repeatedly predicts every feature from the others "
			               "(restricted cubic splines with 3 knots) and removes the best predicted one while its R^2 "
			               "exceeds 0.9. Returns variables, kept, removed, r_squared, r_squared_later, "
			               "r_squared_first, too_few and n_observations. Rows with a NULL or non-finite element "
			               "are skipped.",
			               "redun_agg([x1, x2, x3])", {"x"}, {L});
			AddDescription(info,
			               "Redundancy analysis with options: r2 (0.9), type (ordinary, adjusted), nk (3; 0 linear), "
			               "tlinear (true), rank, qrank, allcat, types (one of spline, linear, categorical per "
			               "feature), feature_names.",
			               "redun_agg([x1, x2, x3], {'r2': 0.8})", {"x", "options"}, {L, A});
		});
	}
}

} // namespace duckdb
