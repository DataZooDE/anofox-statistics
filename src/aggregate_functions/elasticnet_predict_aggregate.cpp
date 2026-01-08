#include <cmath>
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
// ElasticNet Predict Aggregate State
//===--------------------------------------------------------------------===//
struct ElasticNetPredictAggState {
    vector<double> y_train;
    vector<vector<double>> x_train;
    vector<double> y_all;
    vector<bool> y_is_null;
    vector<bool> is_training;
    vector<vector<double>> x_all;

    idx_t n_features;
    bool initialized;

    double alpha;
    double l1_ratio;
    bool fit_intercept;
    uint32_t max_iterations;
    double tolerance;
    double confidence_level;
    NullPolicy null_policy;

    ElasticNetPredictAggState()
        : n_features(0), initialized(false), alpha(1.0), l1_ratio(0.5), fit_intercept(true),
          max_iterations(1000), tolerance(1e-6), confidence_level(0.95), null_policy(NullPolicy::DROP) {}

    void Reset() {
        y_train.clear();
        x_train.clear();
        y_all.clear();
        y_is_null.clear();
        is_training.clear();
        x_all.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data
//===--------------------------------------------------------------------===//
struct ElasticNetPredictAggBindData : public FunctionData {
    double alpha = 1.0;
    double l1_ratio = 0.5;
    bool fit_intercept = true;
    uint32_t max_iterations = 1000;
    double tolerance = 1e-6;
    double confidence_level = 0.95;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<ElasticNetPredictAggBindData>();
        result->alpha = alpha;
        result->l1_ratio = l1_ratio;
        result->fit_intercept = fit_intercept;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->confidence_level = confidence_level;
        result->null_policy = null_policy;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ElasticNetPredictAggBindData>();
        return alpha == other.alpha && l1_ratio == other.l1_ratio && fit_intercept == other.fit_intercept &&
               max_iterations == other.max_iterations && tolerance == other.tolerance &&
               confidence_level == other.confidence_level && null_policy == other.null_policy;
    }
};

//===--------------------------------------------------------------------===//
// Result type
//===--------------------------------------------------------------------===//
static LogicalType GetElasticNetPredictAggResultType() {
    child_list_t<LogicalType> row_children;
    row_children.push_back(make_pair("y", LogicalType::DOUBLE));
    row_children.push_back(make_pair("x", LogicalType::LIST(LogicalType::DOUBLE)));
    row_children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    row_children.push_back(make_pair("is_training", LogicalType::BOOLEAN));

    auto row_struct = LogicalType::STRUCT(std::move(row_children));
    return LogicalType::LIST(row_struct);
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void ElasticNetPredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ElasticNetPredictAggState();
}

static void ElasticNetPredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ElasticNetPredictAggState();
    }
}

static void ElasticNetPredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                        Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<ElasticNetPredictAggBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.alpha = bind_data.alpha;
        state.l1_ratio = bind_data.l1_ratio;
        state.fit_intercept = bind_data.fit_intercept;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;
        state.confidence_level = bind_data.confidence_level;
        state.null_policy = bind_data.null_policy;

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        if (!state.initialized) {
            state.n_features = n_features;
            state.x_train.resize(n_features);
            state.initialized = true;
        }

        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count");
        }

        vector<double> x_row(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            x_row[j] = x_child_data[list_entry.offset + j];
        }

        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        bool row_is_training = y_valid;
        if (row_is_training && state.null_policy == NullPolicy::DROP_Y_ZERO_X) {
            for (idx_t j = 0; j < n_features; j++) {
                if (x_row[j] == 0.0) {
                    row_is_training = false;
                    break;
                }
            }
        }

        state.y_all.push_back(y_val);
        state.y_is_null.push_back(!y_valid);
        state.is_training.push_back(row_is_training);
        state.x_all.push_back(x_row);

        if (row_is_training) {
            state.y_train.push_back(y_val);
            for (idx_t j = 0; j < n_features; j++) {
                state.x_train[j].push_back(x_row[j]);
            }
        }
    }
}

static void ElasticNetPredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ElasticNetPredictAggState **)source_data.data;
    auto targets = (ElasticNetPredictAggState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_train = std::move(source.y_train);
            target.x_train = std::move(source.x_train);
            target.y_all = std::move(source.y_all);
            target.y_is_null = std::move(source.y_is_null);
            target.is_training = std::move(source.is_training);
            target.x_all = std::move(source.x_all);
            target.n_features = source.n_features;
            target.initialized = true;
            target.alpha = source.alpha;
            target.l1_ratio = source.l1_ratio;
            target.fit_intercept = source.fit_intercept;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            target.confidence_level = source.confidence_level;
            target.null_policy = source.null_policy;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts");
        }

        target.y_train.insert(target.y_train.end(), source.y_train.begin(), source.y_train.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_train[j].insert(target.x_train[j].end(), source.x_train[j].begin(), source.x_train[j].end());
        }

        target.y_all.insert(target.y_all.end(), source.y_all.begin(), source.y_all.end());
        target.y_is_null.insert(target.y_is_null.end(), source.y_is_null.begin(), source.y_is_null.end());
        target.is_training.insert(target.is_training.end(), source.is_training.begin(), source.is_training.end());
        target.x_all.insert(target.x_all.end(), source.x_all.begin(), source.x_all.end());
    }
}

static void ElasticNetPredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                          idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Note: Detailed min_obs validation including zero-variance column handling is done in Rust
        AnofoxDataArray y_array;
        y_array.data = state.y_train.data();
        y_array.validity = nullptr;
        y_array.len = state.y_train.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_train) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        AnofoxElasticNetOptions options;
        options.alpha = state.alpha;
        options.l1_ratio = state.l1_ratio;
        options.fit_intercept = state.fit_intercept;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_elasticnet_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t n_rows = state.y_all.size();
        auto list_data = ListVector::GetData(result);
        auto list_offset = ListVector::GetListSize(result);

        list_data[result_idx].offset = list_offset;
        list_data[result_idx].length = n_rows;

        ListVector::Reserve(result, list_offset + n_rows);
        ListVector::SetListSize(result, list_offset + n_rows);

        auto &child_struct = ListVector::GetEntry(result);
        auto &struct_entries = StructVector::GetEntries(child_struct);

        auto &y_vec = *struct_entries[0];
        auto &x_vec = *struct_entries[1];
        auto &yhat_vec = *struct_entries[2];
        auto &yhat_lower_vec = *struct_entries[3];
        auto &yhat_upper_vec = *struct_entries[4];
        auto &is_training_vec = *struct_entries[5];

        for (idx_t row = 0; row < n_rows; row++) {
            idx_t child_idx = list_offset + row;

            if (state.y_is_null[row]) {
                FlatVector::SetNull(y_vec, child_idx, true);
            } else {
                FlatVector::GetData<double>(y_vec)[child_idx] = state.y_all[row];
            }

            auto x_list_data = ListVector::GetData(x_vec);
            auto x_list_offset = ListVector::GetListSize(x_vec);
            x_list_data[child_idx].offset = x_list_offset;
            x_list_data[child_idx].length = state.n_features;

            ListVector::Reserve(x_vec, x_list_offset + state.n_features);
            ListVector::SetListSize(x_vec, x_list_offset + state.n_features);

            auto &x_child = ListVector::GetEntry(x_vec);
            auto x_child_data = FlatVector::GetData<double>(x_child);
            for (idx_t j = 0; j < state.n_features; j++) {
                x_child_data[x_list_offset + j] = state.x_all[row][j];
            }

            AnofoxPredictionResult pred;
            bool pred_success = anofox_predict_with_interval(
                core_result.coefficients, core_result.coefficients_len, core_result.intercept, state.x_all[row].data(),
                state.n_features, core_result.residual_std_error, core_result.n_observations, state.confidence_level,
                &pred);

            if (pred_success) {
                FlatVector::GetData<double>(yhat_vec)[child_idx] = pred.yhat;
                FlatVector::GetData<double>(yhat_lower_vec)[child_idx] = pred.yhat_lower;
                FlatVector::GetData<double>(yhat_upper_vec)[child_idx] = pred.yhat_upper;
            } else {
                FlatVector::SetNull(yhat_vec, child_idx, true);
                FlatVector::SetNull(yhat_lower_vec, child_idx, true);
                FlatVector::SetNull(yhat_upper_vec, child_idx, true);
            }

            FlatVector::GetData<bool>(is_training_vec)[child_idx] = state.is_training[row];
        }

        anofox_free_result_core(&core_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> ElasticNetPredictAggBind(ClientContext &context, AggregateFunction &function,
                                                          vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<ElasticNetPredictAggBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.alpha.has_value()) {
            result->alpha = opts.alpha.value();
        }
        if (opts.l1_ratio.has_value()) {
            result->l1_ratio = opts.l1_ratio.value();
        }
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetElasticNetPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("elasticnet_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterElasticNetFitPredictAggregateFunction(ExtensionLoader &loader) {
    // Primary name (new)
    AggregateFunctionSet func_set("anofox_stats_elasticnet_fit_predict_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_elasticnet_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, AggregateFunction::StateSize<ElasticNetPredictAggState>, ElasticNetPredictAggInitialize,
        ElasticNetPredictAggUpdate, ElasticNetPredictAggCombine, ElasticNetPredictAggFinalize, nullptr,
        ElasticNetPredictAggBind, ElasticNetPredictAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_elasticnet_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<ElasticNetPredictAggState>, ElasticNetPredictAggInitialize,
        ElasticNetPredictAggUpdate, ElasticNetPredictAggCombine, ElasticNetPredictAggFinalize, nullptr,
        ElasticNetPredictAggBind, ElasticNetPredictAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Short alias (new)
    AggregateFunctionSet alias_set("elasticnet_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);

    // Deprecated aliases (old names for backwards compatibility)
    AggregateFunctionSet deprecated_set("elasticnet_predict_agg");
    deprecated_set.AddFunction(basic_func);
    deprecated_set.AddFunction(map_func);
    loader.RegisterFunction(deprecated_set);

    AggregateFunctionSet deprecated_full_set("anofox_stats_elasticnet_predict_agg");
    deprecated_full_set.AddFunction(basic_func);
    deprecated_full_set.AddFunction(map_func);
    loader.RegisterFunction(deprecated_full_set);
}

} // namespace duckdb
