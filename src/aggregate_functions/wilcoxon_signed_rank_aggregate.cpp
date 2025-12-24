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
// Wilcoxon Signed-Rank Aggregate State
//===--------------------------------------------------------------------===//
struct WilcoxonSignedRankAggregateState {
    vector<double> x_values;
    vector<double> y_values;
    bool initialized;

    WilcoxonSignedRankAggregateState() : initialized(false) {}

    void Reset() {
        x_values.clear();
        y_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetWilcoxonSignedRankAggResultType() {
    child_list_t<LogicalType> children;

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
struct WilcoxonSignedRankBindData : public FunctionData {
    WilcoxonMapOptions options;

    WilcoxonSignedRankBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<WilcoxonSignedRankBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<WilcoxonSignedRankBindData>();
        return options.alternative == other.options.alternative &&
               options.confidence_level == other.options.confidence_level &&
               options.continuity_correction == other.options.continuity_correction;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void WilcoxonSignedRankAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) WilcoxonSignedRankAggregateState();
}

static void WilcoxonSignedRankAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WilcoxonSignedRankAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~WilcoxonSignedRankAggregateState();
    }
}

static void WilcoxonSignedRankAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                         Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data, y_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    auto x_vals = UnifiedVectorFormat::GetData<double>(x_data);
    auto y_vals = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WilcoxonSignedRankAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto x_idx = x_data.sel->get_index(i);
        auto y_idx = y_data.sel->get_index(i);

        if (!x_data.validity.RowIsValid(x_idx) || !y_data.validity.RowIsValid(y_idx)) {
            continue;
        }

        double x_val = x_vals[x_idx];
        double y_val = y_vals[y_idx];

        if (std::isnan(x_val) || std::isnan(y_val)) {
            continue;
        }

        state.x_values.push_back(x_val);
        state.y_values.push_back(y_val);
    }
}

static void WilcoxonSignedRankAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (WilcoxonSignedRankAggregateState **)source_data.data;
    auto targets = (WilcoxonSignedRankAggregateState **)target_data.data;

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

static void WilcoxonSignedRankAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                           idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WilcoxonSignedRankAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<WilcoxonSignedRankBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.x_values.size() < 2) {
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

        // Set options
        AnofoxWilcoxonOptions options;
        switch (bind_data.options.alternative.value_or(Alternative::TWO_SIDED)) {
            case Alternative::LESS:
                options.alternative = ANOFOX_ALTERNATIVE_LESS;
                break;
            case Alternative::GREATER:
                options.alternative = ANOFOX_ALTERNATIVE_GREATER;
                break;
            default:
                options.alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
        }
        options.exact = false;
        options.continuity_correction = bind_data.options.continuity_correction.value_or(true);
        options.confidence_level = bind_data.options.confidence_level.value_or(0.95);
        options.mu = 0.0;

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_wilcoxon_signed_rank(x_array, y_array, options, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Wilcoxon signed-rank test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> WilcoxonSignedRankAggBind(ClientContext &context, AggregateFunction &function,
                                                           vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetWilcoxonSignedRankAggResultType();
    auto bind_data = make_uniq<WilcoxonSignedRankBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        bind_data->options = WilcoxonMapOptions::ParseFromValue(options_val);
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("wilcoxon_signed_rank_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterWilcoxonSignedRankAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_wilcoxon_signed_rank_agg");

    // With options: (x, y, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_wilcoxon_signed_rank_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<WilcoxonSignedRankAggregateState>, WilcoxonSignedRankAggInitialize,
        WilcoxonSignedRankAggUpdate, WilcoxonSignedRankAggCombine, WilcoxonSignedRankAggFinalize,
        nullptr, WilcoxonSignedRankAggBind, WilcoxonSignedRankAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (x, y)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_wilcoxon_signed_rank_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<WilcoxonSignedRankAggregateState>, WilcoxonSignedRankAggInitialize,
        WilcoxonSignedRankAggUpdate, WilcoxonSignedRankAggCombine, WilcoxonSignedRankAggFinalize,
        nullptr, WilcoxonSignedRankAggBind, WilcoxonSignedRankAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("wilcoxon_signed_rank_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
