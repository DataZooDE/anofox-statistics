#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// TOST Correlation Aggregate State
//===--------------------------------------------------------------------===//
struct TostCorrelationAggregateState {
    vector<double> x_values;
    vector<double> y_values;
    bool initialized;

    TostCorrelationAggregateState() : initialized(false) {}

    void Reset() {
        x_values.clear();
        y_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetTostCorrelationAggResultType() {
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
struct TostCorrelationBindData : public FunctionData {
    double rho_null;
    double bound_lower;
    double bound_upper;
    double alpha;
    AnofoxTostCorrelationMethod method;

    TostCorrelationBindData() : rho_null(0.0), bound_lower(-0.1), bound_upper(0.1),
                                 alpha(0.05), method(ANOFOX_TOST_COR_PEARSON) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<TostCorrelationBindData>();
        copy->rho_null = rho_null;
        copy->bound_lower = bound_lower;
        copy->bound_upper = bound_upper;
        copy->alpha = alpha;
        copy->method = method;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<TostCorrelationBindData>();
        return rho_null == other.rho_null &&
               bound_lower == other.bound_lower &&
               bound_upper == other.bound_upper &&
               alpha == other.alpha &&
               method == other.method;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void TostCorrelationAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) TostCorrelationAggregateState();
}

static void TostCorrelationAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostCorrelationAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TostCorrelationAggregateState();
    }
}

static void TostCorrelationAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data, y_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    auto x_vals = UnifiedVectorFormat::GetData<double>(x_data);
    auto y_vals = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostCorrelationAggregateState **)sdata.data;

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

static void TostCorrelationAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TostCorrelationAggregateState **)source_data.data;
    auto targets = (TostCorrelationAggregateState **)target_data.data;

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

static void TostCorrelationAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TostCorrelationAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<TostCorrelationBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.x_values.size() < 4) {
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

        AnofoxTostResult tost_result;
        AnofoxError error;

        bool success = anofox_tost_correlation(x_array, y_array, bind_data.rho_null,
                                                bind_data.bound_lower, bind_data.bound_upper,
                                                bind_data.alpha, bind_data.method,
                                                &tost_result, &error);

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
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], tost_result.method ? tost_result.method : "TOST correlation");

        anofox_free_tost_result(&tost_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> TostCorrelationAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetTostCorrelationAggResultType();
    auto bind_data = make_uniq<TostCorrelationBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "rho_null") == 0 || strcasecmp(key, "rho") == 0) {
                        bind_data->rho_null = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "bound_lower") == 0) {
                        bind_data->bound_lower = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "bound_upper") == 0) {
                        bind_data->bound_upper = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "delta") == 0) {
                        double val = key_list[1].GetValue<double>();
                        bind_data->bound_lower = -val;
                        bind_data->bound_upper = val;
                    } else if (strcasecmp(key, "alpha") == 0) {
                        bind_data->alpha = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "method") == 0) {
                        auto method_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(method_str.c_str(), "spearman") == 0) {
                            bind_data->method = ANOFOX_TOST_COR_SPEARMAN;
                        } else {
                            bind_data->method = ANOFOX_TOST_COR_PEARSON;
                        }
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
void RegisterTostCorrelationAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_tost_correlation_agg");

    // With options: (x, y, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_tost_correlation_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<TostCorrelationAggregateState>, TostCorrelationAggInitialize,
        TostCorrelationAggUpdate, TostCorrelationAggCombine, TostCorrelationAggFinalize,
        nullptr, TostCorrelationAggBind, TostCorrelationAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (x, y)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_tost_correlation_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<TostCorrelationAggregateState>, TostCorrelationAggInitialize,
        TostCorrelationAggUpdate, TostCorrelationAggCombine, TostCorrelationAggFinalize,
        nullptr, TostCorrelationAggBind, TostCorrelationAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("tost_correlation_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
