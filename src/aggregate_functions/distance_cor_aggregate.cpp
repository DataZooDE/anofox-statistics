#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

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

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct DistanceCorBindData : public FunctionData {
    uint32_t n_permutations;

    DistanceCorBindData() : n_permutations(1000) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<DistanceCorBindData>();
        copy->n_permutations = n_permutations;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<DistanceCorBindData>();
        return n_permutations == other.n_permutations;
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

static void DistanceCorAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
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
            target.x_values = std::move(source.x_values);
            target.y_values = std::move(source.y_values);
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
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Then compute the test with permutations
        AnofoxTestResult test_result;
        bool test_success = anofox_distance_cor_test(x_array, y_array, bind_data.n_permutations, &test_result, &error);

        if (!test_success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = dcor_result.dcor;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Distance correlation test");

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
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "n_permutations") == 0 || strcasecmp(key, "permutations") == 0) {
                        bind_data->n_permutations = static_cast<uint32_t>(key_list[1].GetValue<int64_t>());
                    }
                }
            }
        }
    }

    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterDistanceCorAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_distance_cor_agg");

    // With options: (x, y, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_distance_cor_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<DistanceCorAggregateState>, DistanceCorAggInitialize,
        DistanceCorAggUpdate, DistanceCorAggCombine, DistanceCorAggFinalize,
        nullptr, DistanceCorAggBind, DistanceCorAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (x, y)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_distance_cor_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<DistanceCorAggregateState>, DistanceCorAggInitialize,
        DistanceCorAggUpdate, DistanceCorAggCombine, DistanceCorAggFinalize,
        nullptr, DistanceCorAggBind, DistanceCorAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("distance_cor_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
