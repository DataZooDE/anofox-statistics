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
// Quantile Predict Aggregate State
//===--------------------------------------------------------------------===//
struct QuantilePredictAggState {
    // Training data (rows where y is NOT NULL)
    vector<double> y_train;
    vector<vector<double>> x_train;

    // ALL rows' data for output
    vector<double> y_all;
    vector<bool> y_is_null;
    vector<bool> is_training;
    vector<vector<double>> x_all;

    idx_t n_features;
    bool initialized;

    // Options
    double tau;
    bool fit_intercept;

    QuantilePredictAggState() : n_features(0), initialized(false), tau(0.5), fit_intercept(true) {}

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
struct QuantilePredictAggBindData : public FunctionData {
    double tau = 0.5;
    bool fit_intercept = true;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<QuantilePredictAggBindData>();
        result->tau = tau;
        result->fit_intercept = fit_intercept;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<QuantilePredictAggBindData>();
        return tau == other.tau && fit_intercept == other.fit_intercept;
    }
};

//===--------------------------------------------------------------------===//
// Result type: LIST(STRUCT(y, x, yhat, is_training))
//===--------------------------------------------------------------------===//
static LogicalType GetQuantilePredictAggResultType() {
    child_list_t<LogicalType> row_children;
    row_children.push_back(make_pair("y", LogicalType::DOUBLE));
    row_children.push_back(make_pair("x", LogicalType::LIST(LogicalType::DOUBLE)));
    row_children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    row_children.push_back(make_pair("is_training", LogicalType::BOOLEAN));

    auto row_struct = LogicalType::STRUCT(std::move(row_children));
    return LogicalType::LIST(row_struct);
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void QuantilePredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) QuantilePredictAggState();
}

static void QuantilePredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (QuantilePredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~QuantilePredictAggState();
    }
}

static void QuantilePredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                      Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<QuantilePredictAggBindData>();

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
    auto states = (QuantilePredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.tau = bind_data.tau;
        state.fit_intercept = bind_data.fit_intercept;

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
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features, n_features);
        }

        vector<double> x_row(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            x_row[j] = x_child_data[list_entry.offset + j];
        }

        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        bool row_is_training = y_valid;

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

static void QuantilePredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (QuantilePredictAggState **)source_data.data;
    auto targets = (QuantilePredictAggState **)target_data.data;

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
            target.tau = source.tau;
            target.fit_intercept = source.fit_intercept;
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

static void QuantilePredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                        idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (QuantilePredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data for fitting
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

        AnofoxQuantileOptions options;
        options.tau = state.tau;
        options.fit_intercept = state.fit_intercept;
        options.max_iterations = 1000;
        options.tolerance = 1e-6;

        AnofoxQuantileFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_quantile_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build LIST result with predictions for ALL rows
        idx_t n_rows = state.y_all.size();
        auto *list_data = ListVector::GetData(result);
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
        auto &is_training_vec = *struct_entries[3];

        for (idx_t row = 0; row < n_rows; row++) {
            idx_t child_idx = list_offset + row;

            if (state.y_is_null[row]) {
                FlatVector::SetNull(y_vec, child_idx, true);
            } else {
                FlatVector::GetData<double>(y_vec)[child_idx] = state.y_all[row];
            }

            auto *x_list_data = ListVector::GetData(x_vec);
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

            // Compute prediction: yhat = intercept + sum(coef[j] * x[j])
            double yhat = std::isnan(core_result.intercept) ? 0.0 : core_result.intercept;
            for (idx_t j = 0; j < core_result.coefficients_len; j++) {
                yhat += core_result.coefficients[j] * state.x_all[row][j];
            }
            FlatVector::GetData<double>(yhat_vec)[child_idx] = yhat;

            FlatVector::GetData<bool>(is_training_vec)[child_idx] = state.is_training[row];
        }

        anofox_free_quantile_result(&core_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> QuantilePredictAggBind(ClientContext &context, AggregateFunction &function,
                                                        vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<QuantilePredictAggBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.tau.has_value()) {
            result->tau = opts.tau.value();
        }
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
    }

    function.return_type = GetQuantilePredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("quantile_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterQuantileFitPredictAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_quantile_fit_predict_agg");

    auto basic_func =
        AggregateFunction("anofox_stats_quantile_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
                          LogicalType::ANY, AggregateFunction::StateSize<QuantilePredictAggState>, QuantilePredictAggInitialize,
                          QuantilePredictAggUpdate, QuantilePredictAggCombine, QuantilePredictAggFinalize, nullptr, QuantilePredictAggBind,
                          QuantilePredictAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_quantile_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<QuantilePredictAggState>, QuantilePredictAggInitialize, QuantilePredictAggUpdate,
        QuantilePredictAggCombine, QuantilePredictAggFinalize, nullptr, QuantilePredictAggBind, QuantilePredictAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    AggregateFunctionSet alias_set("quantile_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
