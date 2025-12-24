#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// TOST T-Test Aggregate State
//===--------------------------------------------------------------------===//
struct TostTTestAggregateState {
    vector<double> group1;
    vector<double> group2;
    bool initialized;

    TostTTestAggregateState() : initialized(false) {}

    void Reset() {
        group1.clear();
        group2.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetTostTTestAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("t_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("t_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("p_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("p_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::DOUBLE));
    children.push_back(make_pair("estimate", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("bound_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("bound_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("equivalent", LogicalType::BOOLEAN));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct TostTTestBindData : public FunctionData {
    TostMapOptions options;

    TostTTestBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<TostTTestBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<TostTTestBindData>();
        return options.delta == other.options.delta &&
               options.bound_lower == other.options.bound_lower &&
               options.bound_upper == other.options.bound_upper &&
               options.confidence_level == other.options.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void TostTTestAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) TostTTestAggregateState();
}

static void TostTTestAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostTTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TostTTestAggregateState();
    }
}

static void TostTTestAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    auto groups = UnifiedVectorFormat::GetData<int32_t>(group_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostTTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = value_data.sel->get_index(i);
        auto grp_idx = group_data.sel->get_index(i);

        if (!value_data.validity.RowIsValid(val_idx) || !group_data.validity.RowIsValid(grp_idx)) {
            continue;
        }

        double val = values[val_idx];
        int32_t group = groups[grp_idx];

        if (std::isnan(val)) {
            continue;
        }

        if (group == 0) {
            state.group1.push_back(val);
        } else {
            state.group2.push_back(val);
        }
    }
}

static void TostTTestAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TostTTestAggregateState **)source_data.data;
    auto targets = (TostTTestAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.group1 = std::move(source.group1);
            target.group2 = std::move(source.group2);
            target.initialized = true;
            continue;
        }

        target.group1.insert(target.group1.end(), source.group1.begin(), source.group1.end());
        target.group2.insert(target.group2.end(), source.group2.begin(), source.group2.end());
    }
}

static void TostTTestAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostTTestAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<TostTTestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.group1.size() < 2 || state.group2.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray group1_array;
        group1_array.data = state.group1.data();
        group1_array.validity = nullptr;
        group1_array.len = state.group1.size();

        AnofoxDataArray group2_array;
        group2_array.data = state.group2.data();
        group2_array.validity = nullptr;
        group2_array.len = state.group2.size();

        // Set options - use delta for symmetric bounds if provided
        AnofoxTostOptions options;
        if (bind_data.options.delta.has_value()) {
            double delta = bind_data.options.delta.value();
            options.bound_lower = -delta;
            options.bound_upper = delta;
        } else {
            options.bound_lower = bind_data.options.bound_lower.value_or(-1.0);
            options.bound_upper = bind_data.options.bound_upper.value_or(1.0);
        }
        options.alpha = 1.0 - bind_data.options.confidence_level.value_or(0.95);
        options.pooled = bind_data.options.kind.value_or(TTestKind::WELCH) == TTestKind::STUDENT;

        AnofoxTostResult tost_result;
        AnofoxError error;

        bool success = anofox_tost_t_test(group1_array, group2_array, options, &tost_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.t_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.t_upper;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.p_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.p_upper;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.df;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.estimate;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.ci_upper;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.bound_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.bound_upper;
        FlatVector::GetData<bool>(*struct_entries[struct_idx++])[result_idx] = tost_result.equivalent;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(tost_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], tost_result.method ? tost_result.method : "TOST Two-Sample t-test");

        anofox_free_tost_result(&tost_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> TostTTestAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetTostTTestAggResultType();
    auto bind_data = make_uniq<TostTTestBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        bind_data->options = TostMapOptions::ParseFromValue(options_val);
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("tost_t_test_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterTostTTestAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_tost_t_test_agg");

    // With options
    auto func_with_opts = AggregateFunction(
        "anofox_stats_tost_t_test_agg", {LogicalType::DOUBLE, LogicalType::INTEGER, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<TostTTestAggregateState>, TostTTestAggInitialize,
        TostTTestAggUpdate, TostTTestAggCombine, TostTTestAggFinalize,
        nullptr, TostTTestAggBind, TostTTestAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options
    auto func_no_opts = AggregateFunction(
        "anofox_stats_tost_t_test_agg", {LogicalType::DOUBLE, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<TostTTestAggregateState>, TostTTestAggInitialize,
        TostTTestAggUpdate, TostTTestAggCombine, TostTTestAggFinalize,
        nullptr, TostTTestAggBind, TostTTestAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("tost_t_test_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
