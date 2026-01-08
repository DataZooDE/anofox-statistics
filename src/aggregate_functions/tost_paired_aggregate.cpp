#include <vector>

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
// TOST Paired T-Test Aggregate State
//===--------------------------------------------------------------------===//
struct TostPairedAggregateState {
    vector<double> x_values;
    vector<double> y_values;
    bool initialized;

    TostPairedAggregateState() : initialized(false) {}

    void Reset() {
        x_values.clear();
        y_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetTostPairedAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("estimate", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("equivalent", LogicalType::BOOLEAN));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct TostPairedBindData : public FunctionData {
    double bound_lower;
    double bound_upper;
    double alpha;

    TostPairedBindData() : bound_lower(-0.5), bound_upper(0.5), alpha(0.05) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<TostPairedBindData>();
        copy->bound_lower = bound_lower;
        copy->bound_upper = bound_upper;
        copy->alpha = alpha;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<TostPairedBindData>();
        return bound_lower == other.bound_lower &&
               bound_upper == other.bound_upper &&
               alpha == other.alpha;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void TostPairedAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) TostPairedAggregateState();
}

static void TostPairedAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostPairedAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TostPairedAggregateState();
    }
}

static void TostPairedAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data, y_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    auto x_vals = UnifiedVectorFormat::GetData<double>(x_data);
    auto y_vals = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostPairedAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto x_idx = x_data.sel->get_index(i);
        auto y_idx = y_data.sel->get_index(i);

        if (!x_data.validity.RowIsValid(x_idx) || !y_data.validity.RowIsValid(y_idx)) {
            continue;
        }

        double x = x_vals[x_idx];
        double y = y_vals[y_idx];

        if (std::isnan(x) || std::isnan(y)) {
            continue;
        }

        state.x_values.push_back(x);
        state.y_values.push_back(y);
    }
}

static void TostPairedAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TostPairedAggregateState **)source_data.data;
    auto targets = (TostPairedAggregateState **)target_data.data;

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

static void TostPairedAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostPairedAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<TostPairedBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.x_values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray x_array;
        x_array.data = state.x_values.data();
        x_array.validity = nullptr;
        x_array.len = state.x_values.size();

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxTostOptions options;
        options.bound_lower = bind_data.bound_lower;
        options.bound_upper = bind_data.bound_upper;
        options.alpha = bind_data.alpha;
        options.pooled = false;

        AnofoxTostResult tost_result;
        AnofoxError error;

        bool success = anofox_tost_t_test_paired(x_array, y_array, options, &tost_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.estimate;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.ci_upper;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = tost_result.p_value;
        FlatVector::GetData<bool>(*struct_entries[struct_idx++])[result_idx] = tost_result.equivalent;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(tost_result.n);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, tost_result.method ? tost_result.method : "TOST paired t-test");

        anofox_free_tost_result(&tost_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> TostPairedAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetTostPairedAggResultType();
    auto bind_data = make_uniq<TostPairedBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "bound_lower") == 0 || strcasecmp(key, "delta") == 0) {
                        double val = key_list[1].GetValue<double>();
                        if (strcasecmp(key, "delta") == 0) {
                            bind_data->bound_lower = -val;
                            bind_data->bound_upper = val;
                        } else {
                            bind_data->bound_lower = val;
                        }
                    } else if (strcasecmp(key, "bound_upper") == 0) {
                        bind_data->bound_upper = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "alpha") == 0) {
                        bind_data->alpha = key_list[1].GetValue<double>();
                    }
                }
            }
        }
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("tost_paired_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterTostPairedAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_tost_paired_agg");

    // With options: (x, y, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_tost_paired_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<TostPairedAggregateState>, TostPairedAggInitialize,
        TostPairedAggUpdate, TostPairedAggCombine, TostPairedAggFinalize,
        nullptr, TostPairedAggBind, TostPairedAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (x, y)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_tost_paired_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<TostPairedAggregateState>, TostPairedAggInitialize,
        TostPairedAggUpdate, TostPairedAggCombine, TostPairedAggFinalize,
        nullptr, TostPairedAggBind, TostPairedAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("tost_paired_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
