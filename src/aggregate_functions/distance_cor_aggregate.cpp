#include <algorithm>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"


namespace duckdb {

//===--------------------------------------------------------------------===//
// Distance Correlation Aggregate State
//===--------------------------------------------------------------------===//
struct DistanceCorAggregateState {
    vector<double> x_values;
    vector<double> y_values;
    bool initialized;

    DistanceCorAggregateState() : initialized(false) {}

    void Reset() {
        x_values.clear();
        y_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetDistanceCorAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("dcor", LogicalType::DOUBLE));
    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));
    children.push_back(make_pair("alternative", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct DistanceCorBindData : public FunctionData {
    uint32_t n_permutations;
    uint64_t seed;
    bool has_seed;

    DistanceCorBindData() : n_permutations(1000), seed(0), has_seed(false) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<DistanceCorBindData>();
        copy->n_permutations = n_permutations;
        copy->seed = seed;
        copy->has_seed = has_seed;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<DistanceCorBindData>();
        return n_permutations == other.n_permutations && seed == other.seed && has_seed == other.has_seed;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void DistanceCorAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) DistanceCorAggregateState();
}

static void DistanceCorAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DistanceCorAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~DistanceCorAggregateState();
    }
}

static void DistanceCorAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data, y_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    auto x_vals = UnifiedVectorFormat::GetData<double>(x_data);
    auto y_vals = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DistanceCorAggregateState **)sdata.data;

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

static void DistanceCorAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (DistanceCorAggregateState **)source_data.data;
    auto targets = (DistanceCorAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.x_values = CombineTake(source.x_values, aggr_input_data);
            target.y_values = CombineTake(source.y_values, aggr_input_data);
            target.initialized = true;
            continue;
        }

        target.x_values.insert(target.x_values.end(), source.x_values.begin(), source.x_values.end());
        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
    }
}

static void DistanceCorAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DistanceCorAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<DistanceCorBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.x_values.size() < 4) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        if (bind_data.has_seed) {
            // Make the seeded result independent of the row order threads
            // delivered: sort the (x, y) pairs canonically.
            vector<std::pair<double, double>> pairs(state.x_values.size());
            for (idx_t k = 0; k < pairs.size(); k++) {
                pairs[k] = std::make_pair(state.x_values[k], state.y_values[k]);
            }
            std::sort(pairs.begin(), pairs.end());
            for (idx_t k = 0; k < pairs.size(); k++) {
                state.x_values[k] = pairs[k].first;
                state.y_values[k] = pairs[k].second;
            }
        }

        // First compute distance correlation
        AnofoxDataArray x_array;
        x_array.data = state.x_values.data();
        x_array.validity = nullptr;
        x_array.len = state.x_values.size();

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxDistanceCorResult dcor_result;
        AnofoxError error;

        bool dcor_success = anofox_distance_cor(x_array, y_array, &dcor_result, &error);
        if (!dcor_success) {
            ThrowUnlessDegenerate("distance_cor_agg", error);
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Then compute the test with permutations
        AnofoxTestResult test_result;
        bool test_success = anofox_distance_cor_test_seeded(x_array, y_array, bind_data.n_permutations, bind_data.seed,
                                                            bind_data.has_seed, &test_result, &error);

        if (!test_success) {
            ThrowUnlessDegenerate("distance_cor_agg", error);
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = dcor_result.dcor;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, test_result.method ? test_result.method : "Distance correlation test");
        SetResultNull(*struct_entries[struct_idx++], result_idx); // alternative: omnibus test

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> DistanceCorAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetDistanceCorAggResultType();
    auto bind_data = make_uniq<DistanceCorBindData>();

    // Parse n_permutations from options if provided
    if (arguments.size() >= 3) {
        Value options_val = EvaluateConstantOptions(context, *arguments[2], "distance_cor_agg");
        auto opts = DistanceCorMapOptions::ParseFromValue(options_val, "distance_cor_agg");
        if (opts.n_permutations.has_value()) {
            bind_data->n_permutations = opts.n_permutations.value();
        }
        if (opts.seed.has_value()) {
            bind_data->seed = opts.seed.value();
            bind_data->has_seed = true;
        }
    }

    PostHogTelemetry::Instance().RecordFunctionCall("distance_cor_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterDistanceCorAggregateFunction(ExtensionLoader &loader) {
    // With options: (x, y, options)
    auto func_with_opts = AggregateFunction(
        "distance_cor_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<DistanceCorAggregateState>, DistanceCorAggInitialize,
        ANOFOX_GUARDED_UPDATE(DistanceCorAggUpdate, DistanceCorAggDestroy, DistanceCorAggInitialize), DistanceCorAggCombine, ANOFOX_GUARDED_FINALIZE(DistanceCorAggFinalize, DistanceCorAggDestroy, DistanceCorAggInitialize),
        nullptr, DistanceCorAggBind, DistanceCorAggDestroy);

    // Without options: (x, y)
    auto func_no_opts = AggregateFunction(
        "distance_cor_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<DistanceCorAggregateState>, DistanceCorAggInitialize,
        ANOFOX_GUARDED_UPDATE(DistanceCorAggUpdate, DistanceCorAggDestroy, DistanceCorAggInitialize), DistanceCorAggCombine, ANOFOX_GUARDED_FINALIZE(DistanceCorAggFinalize, DistanceCorAggDestroy, DistanceCorAggInitialize),
        nullptr, DistanceCorAggBind, DistanceCorAggDestroy);

    {
        AggregateFunctionSet func_set("distance_cor_agg");
        func_set.AddFunction(func_with_opts);
        func_set.AddFunction(func_no_opts);
        CreateAggregateFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        FunctionDescription d1;
        d1.description     = "Computes the distance correlation between two variables, detecting both linear and nonlinear dependence.";
        d1.examples        = {"distance_cor_agg(x, y, {'n_permutations': 1000})"};
        d1.categories      = {"correlation"};
        d1.parameter_names = {"x", "y", "options"};
        d1.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY};
        info.descriptions.push_back(std::move(d1));
        FunctionDescription d2;
        d2.description     = "Computes the distance correlation between two variables, detecting both linear and nonlinear dependence, using default options.";
        d2.examples        = {"distance_cor_agg(x, y)"};
        d2.categories      = {"correlation"};
        d2.parameter_names = {"x", "y"};
        d2.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
        info.descriptions.push_back(std::move(d2));
        loader.RegisterFunction(std::move(info));
    }

}

} // namespace duckdb
