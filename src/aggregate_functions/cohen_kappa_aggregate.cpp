#include <vector>
#include <map>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

namespace duckdb {

//===--------------------------------------------------------------------===//
// Cohen's Kappa Aggregate State
//===--------------------------------------------------------------------===//
struct CohenKappaAggregateState {
    vector<int64_t> rater1_values;
    vector<int64_t> rater2_values;
    bool initialized;

    CohenKappaAggregateState() : initialized(false) {}

    void Reset() {
        rater1_values.clear();
        rater2_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetCohenKappaAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("kappa", LogicalType::DOUBLE));
    children.push_back(make_pair("se", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("z", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct CohenKappaBindData : public FunctionData {
    bool weighted;

    CohenKappaBindData() : weighted(false) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<CohenKappaBindData>();
        copy->weighted = weighted;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<CohenKappaBindData>();
        return weighted == other.weighted;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void CohenKappaAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) CohenKappaAggregateState();
}

static void CohenKappaAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (CohenKappaAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~CohenKappaAggregateState();
    }
}

static void CohenKappaAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
    UnifiedVectorFormat r1_data, r2_data;
    inputs[0].ToUnifiedFormat(count, r1_data);
    inputs[1].ToUnifiedFormat(count, r2_data);
    auto r1_vals = UnifiedVectorFormat::GetData<int64_t>(r1_data);
    auto r2_vals = UnifiedVectorFormat::GetData<int64_t>(r2_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (CohenKappaAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto r1_idx = r1_data.sel->get_index(i);
        auto r2_idx = r2_data.sel->get_index(i);

        if (!r1_data.validity.RowIsValid(r1_idx) || !r2_data.validity.RowIsValid(r2_idx)) {
            continue;
        }

        state.rater1_values.push_back(r1_vals[r1_idx]);
        state.rater2_values.push_back(r2_vals[r2_idx]);
    }
}

static void CohenKappaAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (CohenKappaAggregateState **)source_data.data;
    auto targets = (CohenKappaAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.rater1_values = std::move(source.rater1_values);
            target.rater2_values = std::move(source.rater2_values);
            target.initialized = true;
            continue;
        }

        target.rater1_values.insert(target.rater1_values.end(), source.rater1_values.begin(), source.rater1_values.end());
        target.rater2_values.insert(target.rater2_values.end(), source.rater2_values.begin(), source.rater2_values.end());
    }
}

static void CohenKappaAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (CohenKappaAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<CohenKappaBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.rater1_values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build confusion matrix from rater values
        // Get unique categories
        std::map<int64_t, size_t> cat_map;
        for (auto r : state.rater1_values) {
            if (cat_map.find(r) == cat_map.end()) {
                cat_map[r] = cat_map.size();
            }
        }
        for (auto r : state.rater2_values) {
            if (cat_map.find(r) == cat_map.end()) {
                cat_map[r] = cat_map.size();
            }
        }

        size_t n_cats = cat_map.size();
        if (n_cats < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build the confusion matrix (row = rater1, col = rater2)
        vector<size_t> table(n_cats * n_cats, 0);
        for (size_t j = 0; j < state.rater1_values.size(); j++) {
            size_t r = cat_map[state.rater1_values[j]];
            size_t c = cat_map[state.rater2_values[j]];
            table[r * n_cats + c]++;
        }

        // Build row lengths array (all rows have same length = n_cats)
        vector<size_t> row_lengths(n_cats, n_cats);

        AnofoxKappaResult kappa_result;
        AnofoxError error;

        bool success = anofox_cohen_kappa(table.data(), row_lengths.data(), n_cats,
                                           bind_data.weighted, &kappa_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = kappa_result.kappa;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = kappa_result.se;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = kappa_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = kappa_result.ci_upper;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = kappa_result.z;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = kappa_result.p_value;

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> CohenKappaAggBind(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetCohenKappaAggResultType();
    auto bind_data = make_uniq<CohenKappaBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "weighted") == 0) {
                        bind_data->weighted = key_list[1].GetValue<bool>();
                    }
                }
            }
        }
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("cohen_kappa_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterCohenKappaAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_cohen_kappa_agg");

    // With options: (rater1 BIGINT, rater2 BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_cohen_kappa_agg", {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<CohenKappaAggregateState>, CohenKappaAggInitialize,
        CohenKappaAggUpdate, CohenKappaAggCombine, CohenKappaAggFinalize,
        nullptr, CohenKappaAggBind, CohenKappaAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (rater1 BIGINT, rater2 BIGINT)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_cohen_kappa_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<CohenKappaAggregateState>, CohenKappaAggInitialize,
        CohenKappaAggUpdate, CohenKappaAggCombine, CohenKappaAggFinalize,
        nullptr, CohenKappaAggBind, CohenKappaAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("cohen_kappa_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
