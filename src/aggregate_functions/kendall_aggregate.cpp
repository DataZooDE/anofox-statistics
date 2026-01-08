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
// Kendall Correlation Aggregate State
//===--------------------------------------------------------------------===//
struct KendallAggregateState {
    vector<double> x_values;
    vector<double> y_values;
    bool initialized;

    KendallAggregateState() : initialized(false) {}

    void Reset() {
        x_values.clear();
        y_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetKendallAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("tau", LogicalType::DOUBLE));
    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct KendallBindData : public FunctionData {
    KendallMapOptions options;

    KendallBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<KendallBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<KendallBindData>();
        return options.confidence_level == other.options.confidence_level &&
               options.variant == other.options.variant;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void KendallAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) KendallAggregateState();
}

static void KendallAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (KendallAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~KendallAggregateState();
    }
}

static void KendallAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                              Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data, y_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (KendallAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto x_idx = x_data.sel->get_index(i);
        auto y_idx = y_data.sel->get_index(i);

        if (!x_data.validity.RowIsValid(x_idx) || !y_data.validity.RowIsValid(y_idx)) {
            continue;
        }

        double x = x_values[x_idx];
        double y = y_values[y_idx];

        if (std::isnan(x) || std::isnan(y)) {
            continue;
        }

        state.x_values.push_back(x);
        state.y_values.push_back(y);
    }
}

static void KendallAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (KendallAggregateState **)source_data.data;
    auto targets = (KendallAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.x_values = std::move(source.x_values);
            target.y_values = std::move(source.y_values);
            target.initialized = true;
            continue;
        }

        target.x_values.insert(target.x_values.end(), source.x_values.begin(), source.x_values.end());
        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
    }
}

static void KendallAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (KendallAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<KendallBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.x_values.size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray x_array;
        x_array.data = state.x_values.data();
        x_array.validity = nullptr;
        x_array.len = state.x_values.size();

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxKendallOptions options;
        options.alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
        options.confidence_level = bind_data.options.confidence_level.value_or(0.95);

        // Map KendallType enum to FFI
        if (bind_data.options.variant.has_value()) {
            switch (bind_data.options.variant.value()) {
                case KendallType::TAU_A: options.tau_type = ANOFOX_KENDALL_TAU_A; break;
                case KendallType::TAU_B: options.tau_type = ANOFOX_KENDALL_TAU_B; break;
                case KendallType::TAU_C: options.tau_type = ANOFOX_KENDALL_TAU_C; break;
            }
        } else {
            options.tau_type = ANOFOX_KENDALL_TAU_B; // Default to Tau-B
        }

        AnofoxCorrelationResult cor_result;
        AnofoxError error;

        bool success = anofox_kendall_cor(x_array, y_array, options, &cor_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = cor_result.r;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = cor_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = cor_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = cor_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = cor_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(cor_result.n);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, cor_result.method ? cor_result.method : "Kendall");

        anofox_free_correlation_result(&cor_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> KendallAggBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetKendallAggResultType();
    auto bind_data = make_uniq<KendallBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        bind_data->options = KendallMapOptions::ParseFromValue(options_val);
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("kendall_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterKendallAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_kendall_agg");

    // With options
    auto func_with_opts = AggregateFunction(
        "anofox_stats_kendall_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<KendallAggregateState>, KendallAggInitialize,
        KendallAggUpdate, KendallAggCombine, KendallAggFinalize,
        nullptr, KendallAggBind, KendallAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options
    auto func_no_opts = AggregateFunction(
        "anofox_stats_kendall_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<KendallAggregateState>, KendallAggInitialize,
        KendallAggUpdate, KendallAggCombine, KendallAggFinalize,
        nullptr, KendallAggBind, KendallAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("kendall_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
