#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Chi-Square Goodness of Fit Aggregate State
//===--------------------------------------------------------------------===//
struct ChisqGofAggregateState {
    vector<size_t> observed;
    vector<double> expected;
    bool initialized;

    ChisqGofAggregateState() : initialized(false) {}

    void Reset() {
        observed.clear();
        expected.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetChisqGofAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void ChisqGofAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ChisqGofAggregateState();
}

static void ChisqGofAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ChisqGofAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ChisqGofAggregateState();
    }
}

static void ChisqGofAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                               Vector &state_vector, idx_t count) {
    UnifiedVectorFormat obs_data, exp_data;
    inputs[0].ToUnifiedFormat(count, obs_data);
    inputs[1].ToUnifiedFormat(count, exp_data);
    auto obs_vals = UnifiedVectorFormat::GetData<int64_t>(obs_data);
    auto exp_vals = UnifiedVectorFormat::GetData<double>(exp_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ChisqGofAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto obs_idx = obs_data.sel->get_index(i);
        auto exp_idx = exp_data.sel->get_index(i);

        if (!obs_data.validity.RowIsValid(obs_idx) || !exp_data.validity.RowIsValid(exp_idx)) {
            continue;
        }

        int64_t obs = obs_vals[obs_idx];
        double exp = exp_vals[exp_idx];

        if (obs < 0 || std::isnan(exp) || exp <= 0) {
            continue;
        }

        state.observed.push_back(static_cast<size_t>(obs));
        state.expected.push_back(exp);
    }
}

static void ChisqGofAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ChisqGofAggregateState **)source_data.data;
    auto targets = (ChisqGofAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.observed = std::move(source.observed);
            target.expected = std::move(source.expected);
            target.initialized = true;
            continue;
        }

        target.observed.insert(target.observed.end(), source.observed.begin(), source.observed.end());
        target.expected.insert(target.expected.end(), source.expected.begin(), source.expected.end());
    }
}

static void ChisqGofAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                 idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ChisqGofAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.observed.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxChiSquareResult chisq_result;
        AnofoxError error;

        bool success = anofox_chisq_goodness_of_fit(
            state.observed.data(), state.observed.size(),
            state.expected.data(), state.expected.size(),
            &chisq_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = chisq_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = chisq_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(chisq_result.df);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], chisq_result.method ? chisq_result.method : "Chi-square goodness-of-fit");

        anofox_free_chisq_result(&chisq_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> ChisqGofAggBind(ClientContext &context, AggregateFunction &function,
                                                 vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetChisqGofAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("chisq_gof_agg");
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterChisqGofAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_chisq_gof_agg");

    // (observed BIGINT, expected DOUBLE)
    auto func = AggregateFunction(
        "anofox_stats_chisq_gof_agg", {LogicalType::BIGINT, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<ChisqGofAggregateState>, ChisqGofAggInitialize,
        ChisqGofAggUpdate, ChisqGofAggCombine, ChisqGofAggFinalize,
        nullptr, ChisqGofAggBind, ChisqGofAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("chisq_gof_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
