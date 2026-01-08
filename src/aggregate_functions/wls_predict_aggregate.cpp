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
// WLS Predict Aggregate State
//===--------------------------------------------------------------------===//
struct WlsPredictAggState {
    vector<double> y_train;
    vector<vector<double>> x_train;
    vector<double> weights_train;
    vector<double> y_all;
    vector<bool> y_is_null;
    vector<bool> is_training;
    vector<vector<double>> x_all;
    vector<double> weights_all;

    idx_t n_features;
    bool initialized;

    bool fit_intercept;
    double confidence_level;
    NullPolicy null_policy;

    WlsPredictAggState()
        : n_features(0), initialized(false), fit_intercept(true), confidence_level(0.95),
          null_policy(NullPolicy::DROP) {}

    void Reset() {
        y_train.clear();
        x_train.clear();
        weights_train.clear();
        y_all.clear();
        y_is_null.clear();
        is_training.clear();
        x_all.clear();
        weights_all.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data
//===--------------------------------------------------------------------===//
struct WlsPredictAggBindData : public FunctionData {
    bool fit_intercept = true;
    double confidence_level = 0.95;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<WlsPredictAggBindData>();
        result->fit_intercept = fit_intercept;
        result->confidence_level = confidence_level;
        result->null_policy = null_policy;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<WlsPredictAggBindData>();
        return fit_intercept == other.fit_intercept && confidence_level == other.confidence_level &&
               null_policy == other.null_policy;
    }
};

//===--------------------------------------------------------------------===//
// Result type
//===--------------------------------------------------------------------===//
static LogicalType GetWlsPredictAggResultType() {
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

static void WlsPredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) WlsPredictAggState();
}

static void WlsPredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WlsPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~WlsPredictAggState();
    }
}

static void WlsPredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<WlsPredictAggBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    UnifiedVectorFormat w_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);
    inputs[2].ToUnifiedFormat(count, w_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto w_values = UnifiedVectorFormat::GetData<double>(w_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WlsPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.confidence_level = bind_data.confidence_level;
        state.null_policy = bind_data.null_policy;

        auto x_idx = x_data.sel->get_index(i);
        auto w_idx = w_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx) || !w_data.validity.RowIsValid(w_idx)) {
            continue;
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;
        double weight = w_values[w_idx];

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

        bool row_is_training = y_valid && weight > 0;
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
        state.weights_all.push_back(weight);

        if (row_is_training) {
            state.y_train.push_back(y_val);
            state.weights_train.push_back(weight);
            for (idx_t j = 0; j < n_features; j++) {
                state.x_train[j].push_back(x_row[j]);
            }
        }
    }
}

static void WlsPredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (WlsPredictAggState **)source_data.data;
    auto targets = (WlsPredictAggState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_train = std::move(source.y_train);
            target.x_train = std::move(source.x_train);
            target.weights_train = std::move(source.weights_train);
            target.y_all = std::move(source.y_all);
            target.y_is_null = std::move(source.y_is_null);
            target.is_training = std::move(source.is_training);
            target.x_all = std::move(source.x_all);
            target.weights_all = std::move(source.weights_all);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.confidence_level = source.confidence_level;
            target.null_policy = source.null_policy;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts");
        }

        target.y_train.insert(target.y_train.end(), source.y_train.begin(), source.y_train.end());
        target.weights_train.insert(target.weights_train.end(), source.weights_train.begin(), source.weights_train.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_train[j].insert(target.x_train[j].end(), source.x_train[j].begin(), source.x_train[j].end());
        }

        target.y_all.insert(target.y_all.end(), source.y_all.begin(), source.y_all.end());
        target.y_is_null.insert(target.y_is_null.end(), source.y_is_null.begin(), source.y_is_null.end());
        target.is_training.insert(target.is_training.end(), source.is_training.begin(), source.is_training.end());
        target.x_all.insert(target.x_all.end(), source.x_all.begin(), source.x_all.end());
        target.weights_all.insert(target.weights_all.end(), source.weights_all.begin(), source.weights_all.end());
    }
}

static void WlsPredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WlsPredictAggState **)sdata.data;

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

        AnofoxDataArray w_array;
        w_array.data = state.weights_train.data();
        w_array.validity = nullptr;
        w_array.len = state.weights_train.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_train) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        AnofoxWlsOptions options;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = state.confidence_level;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_wls_fit(y_array, x_arrays.data(), x_arrays.size(), w_array, options, &core_result, nullptr, &error);

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
static unique_ptr<FunctionData> WlsPredictAggBind(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<WlsPredictAggBindData>();

    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[3]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetWlsPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("wls_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterWlsFitPredictAggregateFunction(ExtensionLoader &loader) {
    // Primary name (new)
    AggregateFunctionSet func_set("anofox_stats_wls_fit_predict_agg");

    // wls_fit_predict_agg(y, x, weights)
    auto basic_func = AggregateFunction(
        "anofox_stats_wls_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
        LogicalType::ANY, AggregateFunction::StateSize<WlsPredictAggState>, WlsPredictAggInitialize,
        WlsPredictAggUpdate, WlsPredictAggCombine, WlsPredictAggFinalize, nullptr, WlsPredictAggBind,
        WlsPredictAggDestroy);
    func_set.AddFunction(basic_func);

    // wls_fit_predict_agg(y, x, weights, options)
    auto map_func = AggregateFunction(
        "anofox_stats_wls_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<WlsPredictAggState>, WlsPredictAggInitialize,
        WlsPredictAggUpdate, WlsPredictAggCombine, WlsPredictAggFinalize, nullptr, WlsPredictAggBind,
        WlsPredictAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Short alias (new)
    AggregateFunctionSet alias_set("wls_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);

    // Deprecated aliases (old names for backwards compatibility)
    AggregateFunctionSet deprecated_set("wls_predict_agg");
    deprecated_set.AddFunction(basic_func);
    deprecated_set.AddFunction(map_func);
    loader.RegisterFunction(deprecated_set);

    AggregateFunctionSet deprecated_full_set("anofox_stats_wls_predict_agg");
    deprecated_full_set.AddFunction(basic_func);
    deprecated_full_set.AddFunction(map_func);
    loader.RegisterFunction(deprecated_full_set);
}

} // namespace duckdb
