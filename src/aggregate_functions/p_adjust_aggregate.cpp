// Standalone multiple-testing correction (#219): the aggregate p_adjust_agg and
// the list scalar p_adjust_list, both R's p.adjust(p, method, n) from the
// upstream anofox-statistics crate. NULL p-values pass through as NULL and are
// not counted in n. The row-aligned form is the p_adjust_by table macro and, on
// DuckDB 2.0, the window function p_adjust(p, method, n) OVER (...) (#224).

#include <cmath>
#include <string>
#include <vector>

#include "duckdb/common/map.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include "duckdb.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

static const char *const P_ADJUST_METHODS = "holm (default), hochberg, hommel, bonferroni, BH (alias fdr), BY, none";

//! Adjust `p` (NaN = NULL) with the upstream p.adjust; n < 0 means the number
//! of non-NULL p-values. Throws on an unknown method or an n that is too small.
static vector<double> AdjustPValues(const char *fn_name, const vector<double> &p, const string &method, int64_t n) {
	vector<double> out(p.size());
	AnofoxDataArray p_array;
	p_array.data = p.data();
	p_array.validity = nullptr;
	p_array.len = p.size();
	AnofoxError error;
	if (!anofox_p_adjust(p_array, method.c_str(), n, out.data(), &error)) {
		ThrowFromFfiError(fn_name, error);
	}
	return out;
}

//! Read a constant method argument; NULL means holm. Validated against upstream.
static string BindMethod(ClientContext &context, Expression &expr, const char *fn_name) {
	RequireResolvedParameters(expr);
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: method must be a constant, not a value that varies per row", fn_name);
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	string method = value.IsNull() ? "holm" : value.ToString();
	AdjustPValues(fn_name, {}, method, -1);
	return method;
}

//! Read a constant n argument; NULL means the number of non-NULL p-values (-1).
static int64_t BindN(ClientContext &context, Expression &expr, const char *fn_name) {
	RequireResolvedParameters(expr);
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: n must be a constant, not a value that varies per row", fn_name);
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		return -1;
	}
	auto n = value.GetValue<int64_t>();
	if (n < 0) {
		throw InvalidInputException("%s: n must be non-negative, got %lld", fn_name, static_cast<long long>(n));
	}
	return n;
}

//! Write `adj` as a list entry of `result` at `row`; NaN becomes a NULL element.
static void WriteAdjustedList(Vector &result, idx_t row, const vector<double> &adj) {
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + adj.size());
	auto &child = ListVector::GetEntry(result);
	auto child_data = FlatDataMut<double>(child);
	for (idx_t k = 0; k < adj.size(); k++) {
		if (std::isnan(adj[k])) {
			FlatVector::SetNull(child, offset + k, true);
		} else {
			child_data[offset + k] = adj[k];
		}
	}
	ListVector::SetListSize(result, offset + adj.size());
	ListVector::GetData(result)[row] = list_entry_t(offset, adj.size());
}

// ---------------------------------------------------------------------------
// p_adjust_agg(p [, method [, n]]) -> DOUBLE[]
// ---------------------------------------------------------------------------

//! The p-values in input order, NULL as NaN.
struct PAdjustState {
	vector<double> p;
	bool seen = false;
};

struct PAdjustBindData : public FunctionData {
	string method = "holm";
	int64_t n = -1;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PAdjustBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PAdjustBindData>();
		return method == o.method && n == o.n;
	}
};

static void PAdjustInitialize(data_ptr_t state_p) {
	new (state_p) PAdjustState();
}

static void PAdjustDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PAdjustState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~PAdjustState();
	}
}

static void PAdjustUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat p_data, sdata;
	inputs[0].ToUnifiedFormat(count, p_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto values = UnifiedVectorFormat::GetData<double>(p_data);
	auto states = (PAdjustState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto idx = p_data.sel->get_index(i);
		state.p.push_back(p_data.validity.RowIsValid(idx) ? values[idx] : NAN);
		state.seen = true;
	}
}

static void PAdjustCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                           idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (PAdjustState **)source_data.data;
	auto targets = (PAdjustState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.seen) {
			continue;
		}
		if (!target.seen) {
			target.p = CombineTake(source.p, aggr_input_data);
		} else {
			target.p.insert(target.p.end(), source.p.begin(), source.p.end());
		}
		target.seen = true;
	}
}

static void PAdjustFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PAdjustState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<PAdjustBindData>();
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (!state.seen) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		WriteAdjustedList(result, result_idx, AdjustPValues("p_adjust_agg", state.p, bind.method, bind.n));
	}
}

//! The method and n arguments of p_adjust_agg / p_adjust.
static unique_ptr<FunctionData> BindPAdjustArguments(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                                     const char *fn_name) {
	auto bind = make_uniq<PAdjustBindData>();
	if (arguments.size() >= 2) {
		bind->method = BindMethod(context, *arguments[1], fn_name);
	}
	if (arguments.size() >= 3) {
		bind->n = BindN(context, *arguments[2], fn_name);
	}
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(bind);
}

static unique_ptr<FunctionData> PAdjustAggBind(ClientContext &context, AggregateBindFunction &,
                                               vector<unique_ptr<Expression>> &arguments) {
	return BindPAdjustArguments(context, arguments, "p_adjust_agg");
}

// ---------------------------------------------------------------------------
// p_adjust(p [, method [, n]]) OVER (...) -> DOUBLE, DuckDB 2.0 only (#224)
//
// A window-only aggregate: no update / finalize, so DuckDB evaluates it with
// the custom window aggregator. The family of a row is its window frame (the
// whole partition without ORDER BY); the result is the row's own adjusted
// p-value, NULL when the row is outside its frame. window_init only reads the
// p-values, since the rows it gets (a hash group) may span several partitions;
// each distinct frame is adjusted once, on first use, and shared by the threads.
// Needs the batched window callback, the first one that is told which row it
// computes (row_idx); DuckDB 1.x has none, so there p_adjust is not registered.
// ---------------------------------------------------------------------------

//! One family: the adjusted p-values of the rows of one frame, in row order.
struct PAdjustFamily {
	mutex lock;
	bool adjusted = false;
	vector<double> adj;
};

//! Shared by the threads evaluating one hash group.
struct PAdjustWindowShared {
	//! The p-values of the hash group; NaN for NULL or filtered-out rows.
	vector<double> p;
	mutex lock;
	//! Families by frame, as [start, end) pairs of the non-empty subframes.
	map<vector<idx_t>, shared_ptr<PAdjustFamily>> families;
	//! Number of p-values in `families`; bounded, so sliding frames do not pile up.
	idx_t cached = 0;
};

//! The global state (window_init) holds `shared`; each thread's local state
//! remembers the family of its last frame.
struct PAdjustWindowState {
	unique_ptr<PAdjustWindowShared> shared;
	vector<idx_t> frame;
	vector<idx_t> scratch;
	shared_ptr<PAdjustFamily> family;
};

static void PAdjustWindowInitialize(data_ptr_t state_p) {
	new (state_p) PAdjustWindowState();
}

static void PAdjustWindowDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PAdjustWindowState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~PAdjustWindowState();
	}
}

static void PAdjustWindowInit(AggregateInputData &, const WindowPartitionInput &partition, data_ptr_t g_state) {
	auto &state = *reinterpret_cast<PAdjustWindowState *>(g_state);
	state.shared = make_uniq<PAdjustWindowShared>();
	auto &p = state.shared->p;
	p.assign(static_cast<size_t>(partition.count), NAN);
	if (!partition.inputs || partition.column_ids.empty()) {
		return;
	}
	auto &inputs = *partition.inputs;
	ColumnDataScanState scan;
	DataChunk chunk;
	vector<column_t> p_column {partition.column_ids[0]};
	inputs.InitializeScan(scan, p_column);
	inputs.InitializeScanChunk(scan, chunk);
	idx_t row = 0;
	while (inputs.Scan(scan, chunk)) {
		UnifiedVectorFormat p_data;
		chunk.data[0].ToUnifiedFormat(chunk.size(), p_data);
		auto values = UnifiedVectorFormat::GetData<double>(p_data);
		for (idx_t i = 0; i < chunk.size(); i++) {
			auto idx = p_data.sel->get_index(i);
			if (p_data.validity.RowIsValid(idx) && partition.filter_mask.RowIsValid(row + i)) {
				p[static_cast<size_t>(row + i)] = values[idx];
			}
		}
		row += chunk.size();
	}
}

//! The family of `frame`, adjusted on first use.
static shared_ptr<PAdjustFamily> GetPAdjustFamily(PAdjustWindowShared &shared, const vector<idx_t> &frame,
                                                  const PAdjustBindData &bind) {
	idx_t size = 0;
	for (idx_t k = 0; k < frame.size(); k += 2) {
		size += frame[k + 1] - frame[k];
	}
	shared_ptr<PAdjustFamily> family;
	{
		lock_guard<mutex> guard(shared.lock);
		auto entry = shared.families.find(frame);
		if (entry != shared.families.end()) {
			family = entry->second;
		} else {
			family = make_shared_ptr<PAdjustFamily>();
			if (shared.cached + size <= 2 * static_cast<idx_t>(shared.p.size())) {
				shared.families.emplace(frame, family);
				shared.cached += size;
			}
		}
	}
	lock_guard<mutex> guard(family->lock);
	if (!family->adjusted) {
		vector<double> p;
		p.reserve(static_cast<size_t>(size));
		for (idx_t k = 0; k < frame.size(); k += 2) {
			p.insert(p.end(), shared.p.begin() + static_cast<std::ptrdiff_t>(frame[k]),
			         shared.p.begin() + static_cast<std::ptrdiff_t>(frame[k + 1]));
		}
		family->adj = AdjustPValues("p_adjust", p, bind.method, bind.n);
		family->adjusted = true;
	}
	return family;
}

static void PAdjustWindowBatch(AggregateInputData &aggr_input_data, const WindowPartitionInput &, const_data_ptr_t g_state,
                               data_ptr_t l_state, const SubFrames *subframes_per_row, idx_t count, Vector &result,
                               idx_t row_idx) {
	auto &shared = *reinterpret_cast<const PAdjustWindowState *>(g_state)->shared;
	auto &local = *reinterpret_cast<PAdjustWindowState *>(l_state);
	auto &bind = aggr_input_data.bind_data->Cast<PAdjustBindData>();
	auto data = FlatDataMut<double>(result);
	for (idx_t i = 0; i < count; i++) {
		auto &frame = local.scratch;
		frame.clear();
		for (const auto &bounds : subframes_per_row[i]) {
			if (bounds.start < bounds.end) {
				frame.push_back(bounds.start);
				frame.push_back(bounds.end);
			}
		}
		if (!local.family || frame != local.frame) {
			local.family = GetPAdjustFamily(shared, frame, bind);
			local.frame = frame;
		}
		// The position of the row in its family, if it is in its frame.
		const idx_t row = row_idx + i;
		idx_t pos = 0;
		bool in_frame = false;
		for (idx_t k = 0; k < frame.size(); k += 2) {
			if (row >= frame[k] && row < frame[k + 1]) {
				pos += row - frame[k];
				in_frame = true;
				break;
			}
			pos += frame[k + 1] - frame[k];
		}
		const auto &adj = local.family->adj;
		if (!in_frame || std::isnan(adj[static_cast<size_t>(pos)])) {
			FlatVector::SetNull(result, i, true);
		} else {
			data[i] = adj[static_cast<size_t>(pos)];
		}
	}
}

static unique_ptr<FunctionData> PAdjustWindowBind(ClientContext &context, AggregateBindFunction &,
                                                  vector<unique_ptr<Expression>> &arguments) {
	return BindPAdjustArguments(context, arguments, "p_adjust");
}

// ---------------------------------------------------------------------------
// p_adjust_list(p DOUBLE[] [, method [, n]]) -> DOUBLE[]
// ---------------------------------------------------------------------------

static unique_ptr<FunctionData> PAdjustListBind(ClientContext &context, ScalarBindFunction &,
                                                vector<unique_ptr<Expression>> &arguments) {
	// Constant arguments are checked here, so a bad method fails at bind time;
	// per-row values are checked row by row.
	if (arguments.size() >= 2 && arguments[1]->IsFoldable()) {
		BindMethod(context, *arguments[1], "p_adjust_list");
	}
	if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
		BindN(context, *arguments[2], "p_adjust_list");
	}
	PostHogTelemetry::Instance().RecordFunctionCall("p_adjust_list");
	return nullptr;
}

static void PAdjustListFunction(DataChunk &args, ExpressionState &, Vector &result) {
	const bool all_constant = args.AllConstant();
	idx_t count = args.size();
	auto &list_vec = args.data[0];
	UnifiedVectorFormat list_data, child_data, method_data, n_data;
	list_vec.ToUnifiedFormat(count, list_data);
	auto &child = ListVector::GetEntry(list_vec);
	child.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);
	const bool has_method = args.ColumnCount() >= 2;
	const bool has_n = args.ColumnCount() >= 3;
	if (has_method) {
		args.data[1].ToUnifiedFormat(count, method_data);
	}
	if (has_n) {
		args.data[2].ToUnifiedFormat(count, n_data);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	vector<double> p;
	for (idx_t row = 0; row < count; row++) {
		auto l_idx = list_data.sel->get_index(row);
		if (!list_data.validity.RowIsValid(l_idx)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		string method = "holm";
		if (has_method) {
			auto m_idx = method_data.sel->get_index(row);
			if (method_data.validity.RowIsValid(m_idx)) {
				method = UnifiedVectorFormat::GetData<string_t>(method_data)[m_idx].GetString();
			}
		}
		int64_t n = -1;
		if (has_n) {
			auto n_idx = n_data.sel->get_index(row);
			if (n_data.validity.RowIsValid(n_idx)) {
				n = UnifiedVectorFormat::GetData<int64_t>(n_data)[n_idx];
				if (n < 0) {
					throw InvalidInputException("p_adjust_list: n must be non-negative, got %lld",
					                            static_cast<long long>(n));
				}
			}
		}
		auto entry = entries[l_idx];
		p.assign(entry.length, NAN);
		for (idx_t k = 0; k < entry.length; k++) {
			auto c_idx = child_data.sel->get_index(entry.offset + k);
			if (child_data.validity.RowIsValid(c_idx)) {
				p[k] = child_values[c_idx];
			}
		}
		WriteAdjustedList(result, row, AdjustPValues("p_adjust_list", p, method, n));
	}
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void RegisterPAdjustFunctions(ExtensionLoader &loader) {
	const auto list_type = LogicalType::LIST(LogicalType::DOUBLE);
	const vector<vector<LogicalType>> signatures = {
	    {LogicalType::DOUBLE},
	    {LogicalType::DOUBLE, LogicalType::VARCHAR},
	    {LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::BIGINT}};

	{
		AggregateFunctionSet set("p_adjust_agg");
		for (auto &args : signatures) {
			AggregateFunction fn("p_adjust_agg", args, list_type, AggregateFunction::StateSize<PAdjustState>,
			                     ANOFOX_AGG_INIT(PAdjustInitialize),
			                     ANOFOX_GUARDED_UPDATE(PAdjustUpdate, PAdjustDestroy, PAdjustInitialize),
			                     PAdjustCombine,
			                     ANOFOX_GUARDED_FINALIZE(PAdjustFinalize, PAdjustDestroy, PAdjustInitialize), nullptr,
			                     ANOFOX_AGG_BIND(PAdjustAggBind), PAdjustDestroy);
			set.AddFunction(fn);
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = std::string("Adjusts the p-values of a group for multiple testing (R p.adjust) and returns ") +
		                "them as a list in input order (fix the order with ORDER BY inside the call). NULL p-values " +
		                "stay NULL and are not counted. method: " + P_ADJUST_METHODS +
		                "; case-insensitive. n: number of hypotheses (default: the number of non-NULL p-values; " +
		                "may not be smaller).";
		d.examples = {"p_adjust_agg(p_value, 'BH' ORDER BY test)"};
		d.categories = {"hypothesis-testing", "multiple-testing"};
		d.parameter_names = {"p", "method", "n"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::BIGINT};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}

	if (kHasWindowBatchCallback) {
		AggregateFunctionSet set("p_adjust");
		for (auto &args : signatures) {
			AggregateFunction fn("p_adjust", args, LogicalType::DOUBLE, AggregateFunction::StateSize<PAdjustWindowState>,
			                     ANOFOX_AGG_INIT(PAdjustWindowInitialize), nullptr, nullptr, nullptr, nullptr,
			                     ANOFOX_AGG_BIND(PAdjustWindowBind), PAdjustWindowDestroy);
			fn.SetWindowInitCallback(PAdjustWindowInit);
			// Volatile keeps DuckDB 2.0's WindowSelfJoinOptimizer from rewriting
			// p_adjust(...) OVER (PARTITION BY ...) into GROUP BY + join: it does not
			// check that the aggregate has update/combine, and this one is window-only.
			fn.SetStability(FunctionStability::VOLATILE);
			SetWindowBatchCallbackOf(fn, PAdjustWindowBatch);
			// The family is the frame, whatever order the arguments come in.
			SetOrderDependentOf(fn, AggregateOrderDependent::NOT_ORDER_DEPENDENT);
			set.AddFunction(fn);
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = std::string("Window function (DuckDB 2.0+): the p-value of the row adjusted for multiple ") +
		                "testing (R p.adjust) within its window frame, i.e. within the partition when OVER has no " +
		                "ORDER BY. NULL p-values stay NULL and are not counted. method: " + P_ADJUST_METHODS +
		                "; case-insensitive. n: number of hypotheses (default: the number of non-NULL p-values; " +
		                "may not be smaller).";
		d.examples = {"p_adjust(p_value, 'BH') OVER (PARTITION BY endpoint)"};
		d.categories = {"hypothesis-testing", "multiple-testing"};
		d.parameter_names = {"p", "method", "n"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::BIGINT};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}

	{
		ScalarFunctionSet set("p_adjust_list");
		for (auto &args : signatures) {
			vector<LogicalType> list_args = args;
			list_args[0] = list_type;
			ScalarFunction fn(list_args, list_type, PAdjustListFunction, ANOFOX_SCALAR_BIND(PAdjustListBind));
			MarkFallible(fn);
			// A NULL method or n means the default, so NULL arguments must reach the function.
			SetNullHandlingOf(fn, FunctionNullHandling::SPECIAL_HANDLING);
			set.AddFunction(fn);
		}
		CreateScalarFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = std::string("Adjusts a list of p-values for multiple testing (R p.adjust); returns the ") +
		                "adjusted list in the same order. NULL elements stay NULL and are not counted. method: " +
		                P_ADJUST_METHODS + "; case-insensitive. n: number of hypotheses (default: the number of " +
		                "non-NULL p-values; may not be smaller).";
		d.examples = {"p_adjust_list([0.01, 0.02, 0.04], 'BH')"};
		d.categories = {"hypothesis-testing", "multiple-testing"};
		d.parameter_names = {"p", "method", "n"};
		d.parameter_types = {list_type, LogicalType::VARCHAR, LogicalType::BIGINT};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
