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
// Diebold-Mariano Test Aggregate State
//===--------------------------------------------------------------------===//
struct DieboldMarianoAggregateState {
    vector<double> actual;
    vector<double> forecast1;
    vector<double> forecast2;
    bool initialized;

    DieboldMarianoAggregateState() : initialized(false) {}

    void Reset() {
        actual.clear();
        forecast1.clear();
        forecast2.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetDieboldMarianoAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct DieboldMarianoBindData : public FunctionData {
    AnofoxForecastLoss loss;
    AnofoxForecastVarEstimator var_estimator;
    size_t horizon;
    AnofoxAlternative alternative;

    DieboldMarianoBindData()
        : loss(ANOFOX_FORECAST_LOSS_SQUARED),
          var_estimator(ANOFOX_FORECAST_VAR_ACF),
          horizon(1),
          alternative(ANOFOX_ALTERNATIVE_TWO_SIDED) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<DieboldMarianoBindData>();
        copy->loss = loss;
        copy->var_estimator = var_estimator;
        copy->horizon = horizon;
        copy->alternative = alternative;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<DieboldMarianoBindData>();
        return loss == other.loss &&
               var_estimator == other.var_estimator &&
               horizon == other.horizon &&
               alternative == other.alternative;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void DieboldMarianoAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) DieboldMarianoAggregateState();
}

static void DieboldMarianoAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DieboldMarianoAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~DieboldMarianoAggregateState();
    }
}

static void DieboldMarianoAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                     Vector &state_vector, idx_t count) {
    UnifiedVectorFormat actual_data, f1_data, f2_data;
    inputs[0].ToUnifiedFormat(count, actual_data);
    inputs[1].ToUnifiedFormat(count, f1_data);
    inputs[2].ToUnifiedFormat(count, f2_data);
    auto actuals = UnifiedVectorFormat::GetData<double>(actual_data);
    auto f1s = UnifiedVectorFormat::GetData<double>(f1_data);
    auto f2s = UnifiedVectorFormat::GetData<double>(f2_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DieboldMarianoAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto act_idx = actual_data.sel->get_index(i);
        auto f1_idx = f1_data.sel->get_index(i);
        auto f2_idx = f2_data.sel->get_index(i);

        if (!actual_data.validity.RowIsValid(act_idx) ||
            !f1_data.validity.RowIsValid(f1_idx) ||
            !f2_data.validity.RowIsValid(f2_idx)) {
            continue;
        }

        double act = actuals[act_idx];
        double f1 = f1s[f1_idx];
        double f2 = f2s[f2_idx];

        if (std::isnan(act) || std::isnan(f1) || std::isnan(f2)) {
            continue;
        }

        state.actual.push_back(act);
        state.forecast1.push_back(f1);
        state.forecast2.push_back(f2);
    }
}

static void DieboldMarianoAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (DieboldMarianoAggregateState **)source_data.data;
    auto targets = (DieboldMarianoAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.actual = std::move(source.actual);
            target.forecast1 = std::move(source.forecast1);
            target.forecast2 = std::move(source.forecast2);
            target.initialized = true;
            continue;
        }

        target.actual.insert(target.actual.end(), source.actual.begin(), source.actual.end());
        target.forecast1.insert(target.forecast1.end(), source.forecast1.begin(), source.forecast1.end());
        target.forecast2.insert(target.forecast2.end(), source.forecast2.begin(), source.forecast2.end());
    }
}

static void DieboldMarianoAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                       idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DieboldMarianoAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<DieboldMarianoBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.actual.size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray actual_array;
        actual_array.data = state.actual.data();
        actual_array.validity = nullptr;
        actual_array.len = state.actual.size();

        AnofoxDataArray f1_array;
        f1_array.data = state.forecast1.data();
        f1_array.validity = nullptr;
        f1_array.len = state.forecast1.size();

        AnofoxDataArray f2_array;
        f2_array.data = state.forecast2.data();
        f2_array.validity = nullptr;
        f2_array.len = state.forecast2.size();

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_diebold_mariano(actual_array, f1_array, f2_array,
                                               bind_data.loss, bind_data.var_estimator,
                                               bind_data.horizon, bind_data.alternative,
                                               &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Diebold-Mariano test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> DieboldMarianoAggBind(ClientContext &context, AggregateFunction &function,
                                                        vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetDieboldMarianoAggResultType();
    auto bind_data = make_uniq<DieboldMarianoBindData>();

    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[3]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "loss") == 0) {
                        auto loss_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(loss_str.c_str(), "absolute") == 0) {
                            bind_data->loss = ANOFOX_FORECAST_LOSS_ABSOLUTE;
                        }
                    } else if (strcasecmp(key, "var_estimator") == 0) {
                        auto var_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(var_str.c_str(), "bartlett") == 0) {
                            bind_data->var_estimator = ANOFOX_FORECAST_VAR_BARTLETT;
                        }
                    } else if (strcasecmp(key, "horizon") == 0) {
                        bind_data->horizon = static_cast<size_t>(key_list[1].GetValue<int64_t>());
                    } else if (strcasecmp(key, "alternative") == 0) {
                        auto alt_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(alt_str.c_str(), "less") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_LESS;
                        } else if (strcasecmp(alt_str.c_str(), "greater") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_GREATER;
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
void RegisterDieboldMarianoAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_diebold_mariano_agg");

    // With options: (actual, forecast1, forecast2, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_diebold_mariano_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<DieboldMarianoAggregateState>, DieboldMarianoAggInitialize,
        DieboldMarianoAggUpdate, DieboldMarianoAggCombine, DieboldMarianoAggFinalize,
        nullptr, DieboldMarianoAggBind, DieboldMarianoAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (actual, forecast1, forecast2)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_diebold_mariano_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<DieboldMarianoAggregateState>, DieboldMarianoAggInitialize,
        DieboldMarianoAggUpdate, DieboldMarianoAggCombine, DieboldMarianoAggFinalize,
        nullptr, DieboldMarianoAggBind, DieboldMarianoAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("diebold_mariano_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
