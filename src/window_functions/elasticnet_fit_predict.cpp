#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

struct ElasticNetFitPredictState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    vector<double> current_x;
    bool has_current_x;

    bool fit_intercept;
    double confidence_level;
    double alpha;
    double l1_ratio;
    uint32_t max_iterations;
    double tolerance;

    ElasticNetFitPredictState()
        : n_features(0), initialized(false), has_current_x(false), fit_intercept(true), confidence_level(0.95),
          alpha(1.0), l1_ratio(0.5), max_iterations(1000), tolerance(1e-6) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        current_x.clear();
        n_features = 0;
        initialized = false;
        has_current_x = false;
    }
};

struct ElasticNetFitPredictBindData : public FunctionData {
    bool fit_intercept = true;
    double confidence_level = 0.95;
    double alpha = 1.0;
    double l1_ratio = 0.5;
    uint32_t max_iterations = 1000;
    double tolerance = 1e-6;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<ElasticNetFitPredictBindData>();
        result->fit_intercept = fit_intercept;
        result->confidence_level = confidence_level;
        result->alpha = alpha;
        result->l1_ratio = l1_ratio;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ElasticNetFitPredictBindData>();
        return fit_intercept == other.fit_intercept && confidence_level == other.confidence_level &&
               alpha == other.alpha && l1_ratio == other.l1_ratio && max_iterations == other.max_iterations &&
               tolerance == other.tolerance;
    }
};

static LogicalType GetElasticNetFitPredictResultType() {
    child_list_t<LogicalType> children;
    children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    return LogicalType::STRUCT(std::move(children));
}

static void ElasticNetFitPredictInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ElasticNetFitPredictState();
}

static void ElasticNetFitPredictDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetFitPredictState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ElasticNetFitPredictState();
    }
}

static void ElasticNetFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                        Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<ElasticNetFitPredictBindData>();

    UnifiedVectorFormat y_data, x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.confidence_level = bind_data.confidence_level;
        state.alpha = bind_data.alpha;
        state.l1_ratio = bind_data.l1_ratio;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            state.has_current_x = false;
            continue;
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        if (!state.initialized) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
            state.initialized = true;
        }

        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count");
        }

        state.current_x.resize(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            state.current_x[j] = x_child_data[list_entry.offset + j];
        }
        state.has_current_x = true;

        auto y_idx = y_data.sel->get_index(i);
        if (y_data.validity.RowIsValid(y_idx)) {
            state.y_values.push_back(y_values[y_idx]);
            for (idx_t j = 0; j < n_features; j++) {
                state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
            }
        }
    }
}

static void ElasticNetFitPredictCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &,
                                         idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ElasticNetFitPredictState **)source_data.data;
    auto targets = (ElasticNetFitPredictState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized)
            continue;

        if (!target.initialized) {
            target.y_values = std::move(source.y_values);
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            target.current_x = std::move(source.current_x);
            target.has_current_x = source.has_current_x;
            target.fit_intercept = source.fit_intercept;
            target.confidence_level = source.confidence_level;
            target.alpha = source.alpha;
            target.l1_ratio = source.l1_ratio;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts");
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
                                       source.x_columns[j].end());
        }

        if (source.has_current_x) {
            target.current_x = std::move(source.current_x);
            target.has_current_x = true;
        }
    }
}

static void ElasticNetFitPredictFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                          idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetFitPredictState **)sdata.data;
    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || !state.has_current_x) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
        if (state.y_values.size() <= min_obs) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray y_array = {state.y_values.data(), nullptr, state.y_values.size()};
        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_columns) {
            x_arrays.push_back({col.data(), nullptr, col.size()});
        }

        AnofoxElasticNetOptions options = {state.alpha, state.l1_ratio, state.fit_intercept, state.max_iterations,
                                           state.tolerance};
        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_elasticnet_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxPredictionResult pred_result;
        bool pred_success =
            anofox_predict_with_interval(core_result.coefficients, core_result.coefficients_len, core_result.intercept,
                                         state.current_x.data(), state.current_x.size(), core_result.residual_std_error,
                                         core_result.n_observations, state.confidence_level, &pred_result);

        anofox_free_result_core(&core_result);

        if (!pred_success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        FlatVector::GetData<double>(*struct_entries[0])[result_idx] = pred_result.yhat;
        FlatVector::GetData<double>(*struct_entries[1])[result_idx] = pred_result.yhat_lower;
        FlatVector::GetData<double>(*struct_entries[2])[result_idx] = pred_result.yhat_upper;

        state.Reset();
    }
}

static unique_ptr<FunctionData> ElasticNetFitPredictBind(ClientContext &context, AggregateFunction &function,
                                                          vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<ElasticNetFitPredictBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value())
            result->fit_intercept = opts.fit_intercept.value();
        if (opts.confidence_level.has_value())
            result->confidence_level = opts.confidence_level.value();
        auto reg = opts.GetRegularizationStrength();
        if (reg.has_value())
            result->alpha = reg.value();
        if (opts.l1_ratio.has_value())
            result->l1_ratio = opts.l1_ratio.value();
        if (opts.max_iterations.has_value())
            result->max_iterations = opts.max_iterations.value();
        if (opts.tolerance.has_value())
            result->tolerance = opts.tolerance.value();
    }

    function.return_type = GetElasticNetFitPredictResultType();
    return std::move(result);
}

void RegisterElasticNetFitPredictFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_elasticnet_fit_predict");

    auto basic_func = AggregateFunction(
        "anofox_stats_elasticnet_fit_predict", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        GetElasticNetFitPredictResultType(), AggregateFunction::StateSize<ElasticNetFitPredictState>,
        ElasticNetFitPredictInitialize, ElasticNetFitPredictUpdate, ElasticNetFitPredictCombine,
        ElasticNetFitPredictFinalize, nullptr, ElasticNetFitPredictBind, ElasticNetFitPredictDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_elasticnet_fit_predict",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
        GetElasticNetFitPredictResultType(), AggregateFunction::StateSize<ElasticNetFitPredictState>,
        ElasticNetFitPredictInitialize, ElasticNetFitPredictUpdate, ElasticNetFitPredictCombine,
        ElasticNetFitPredictFinalize, nullptr, ElasticNetFitPredictBind, ElasticNetFitPredictDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    AggregateFunctionSet alias_set("elasticnet_fit_predict");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
