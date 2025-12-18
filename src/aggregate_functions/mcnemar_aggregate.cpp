#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// McNemar Test Aggregate State
//===--------------------------------------------------------------------===//
struct McNemarAggregateState {
    vector<double> var1_values;
    vector<double> var2_values;
    bool initialized;

    McNemarAggregateState() : initialized(false) {}

    void Reset() {
        var1_values.clear();
        var2_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetMcNemarAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct McNemarBindData : public FunctionData {
    bool correction;

    McNemarBindData() : correction(true) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<McNemarBindData>();
        copy->correction = correction;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<McNemarBindData>();
        return correction == other.correction;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void McNemarAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) McNemarAggregateState();
}

static void McNemarAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (McNemarAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~McNemarAggregateState();
    }
}

static void McNemarAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                              Vector &state_vector, idx_t count) {
    UnifiedVectorFormat v1_data, v2_data;
    inputs[0].ToUnifiedFormat(count, v1_data);
    inputs[1].ToUnifiedFormat(count, v2_data);
    auto v1_vals = UnifiedVectorFormat::GetData<int64_t>(v1_data);
    auto v2_vals = UnifiedVectorFormat::GetData<int64_t>(v2_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (McNemarAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto v1_idx = v1_data.sel->get_index(i);
        auto v2_idx = v2_data.sel->get_index(i);

        if (!v1_data.validity.RowIsValid(v1_idx) || !v2_data.validity.RowIsValid(v2_idx)) {
            continue;
        }

        state.var1_values.push_back(static_cast<double>(v1_vals[v1_idx]));
        state.var2_values.push_back(static_cast<double>(v2_vals[v2_idx]));
    }
}

static void McNemarAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (McNemarAggregateState **)source_data.data;
    auto targets = (McNemarAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.var1_values = std::move(source.var1_values);
            target.var2_values = std::move(source.var2_values);
            target.initialized = true;
            continue;
        }

        target.var1_values.insert(target.var1_values.end(), source.var1_values.begin(), source.var1_values.end());
        target.var2_values.insert(target.var2_values.end(), source.var2_values.begin(), source.var2_values.end());
    }
}

static void McNemarAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (McNemarAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<McNemarBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.var1_values.size() < 4) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build 2x2 contingency table from paired binary values
        // Table: [[a, b], [c, d]] where:
        // a = both 0 (--), b = var1=0, var2=1 (-+)
        // c = var1=1, var2=0 (+-), d = both 1 (++)
        size_t a = 0, b = 0, c = 0, d = 0;
        for (size_t j = 0; j < state.var1_values.size(); j++) {
            int64_t v1 = static_cast<int64_t>(state.var1_values[j]);
            int64_t v2 = static_cast<int64_t>(state.var2_values[j]);
            if (v1 == 0 && v2 == 0) a++;
            else if (v1 == 0 && v2 != 0) b++;
            else if (v1 != 0 && v2 == 0) c++;
            else d++;
        }

        AnofoxChiSquareResult mcnemar_result;
        AnofoxError error;

        bool success = anofox_mcnemar_test(a, b, c, d, bind_data.correction, false,
                                            &mcnemar_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = mcnemar_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = mcnemar_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(mcnemar_result.df);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], mcnemar_result.method ? mcnemar_result.method : "McNemar's test");

        anofox_free_chisq_result(&mcnemar_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> McNemarAggBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetMcNemarAggResultType();
    auto bind_data = make_uniq<McNemarBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "correction") == 0) {
                        bind_data->correction = key_list[1].GetValue<bool>();
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
void RegisterMcNemarAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_mcnemar_agg");

    // With options: (var1 BIGINT, var2 BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_mcnemar_agg", {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<McNemarAggregateState>, McNemarAggInitialize,
        McNemarAggUpdate, McNemarAggCombine, McNemarAggFinalize,
        nullptr, McNemarAggBind, McNemarAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (var1 BIGINT, var2 BIGINT)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_mcnemar_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<McNemarAggregateState>, McNemarAggInitialize,
        McNemarAggUpdate, McNemarAggCombine, McNemarAggFinalize,
        nullptr, McNemarAggBind, McNemarAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("mcnemar_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
