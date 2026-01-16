#include <cmath>
#include <vector>
#include <algorithm>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Isotonic Predict Aggregate State
// Note: Isotonic regression is 1D: x -> y (not matrix X)
//===--------------------------------------------------------------------===//
struct IsotonicPredictAggState {
    // Training data (rows where y is NOT NULL)
    vector<double> x_train;
    vector<double> y_train;

    // ALL rows' data for output
    vector<double> x_all;
    vector<double> y_all;
    vector<bool> y_is_null;
    vector<bool> is_training;

    bool initialized;

    // Options
    bool increasing;

    // Stored model for prediction
    vector<double> sorted_x;
    vector<double> fitted_y;
    bool model_fitted;

    IsotonicPredictAggState() : initialized(false), increasing(true), model_fitted(false) {}

    void Reset() {
        x_train.clear();
        y_train.clear();
        x_all.clear();
        y_all.clear();
        y_is_null.clear();
        is_training.clear();
        sorted_x.clear();
        fitted_y.clear();
        initialized = false;
        model_fitted = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data
//===--------------------------------------------------------------------===//
struct IsotonicPredictAggBindData : public FunctionData {
    bool increasing = true;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<IsotonicPredictAggBindData>();
        result->increasing = increasing;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<IsotonicPredictAggBindData>();
        return increasing == other.increasing;
    }
};

//===--------------------------------------------------------------------===//
// Result type: LIST(STRUCT(y, x, yhat, is_training))
//===--------------------------------------------------------------------===//
static LogicalType GetIsotonicPredictAggResultType() {
    child_list_t<LogicalType> row_children;
    row_children.push_back(make_pair("y", LogicalType::DOUBLE));
    row_children.push_back(make_pair("x", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    row_children.push_back(make_pair("is_training", LogicalType::BOOLEAN));

    auto row_struct = LogicalType::STRUCT(std::move(row_children));
    return LogicalType::LIST(row_struct);
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void IsotonicPredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) IsotonicPredictAggState();
}

static void IsotonicPredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (IsotonicPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~IsotonicPredictAggState();
    }
}

static void IsotonicPredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                      Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<IsotonicPredictAggBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);  // y
    inputs[1].ToUnifiedFormat(count, x_data);  // x (scalar)

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_values = UnifiedVectorFormat::GetData<double>(x_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (IsotonicPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.increasing = bind_data.increasing;

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;
        }

        double x_val = x_values[x_idx];
        state.initialized = true;

        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        bool row_is_training = y_valid;

        state.x_all.push_back(x_val);
        state.y_all.push_back(y_val);
        state.y_is_null.push_back(!y_valid);
        state.is_training.push_back(row_is_training);

        if (row_is_training) {
            state.x_train.push_back(x_val);
            state.y_train.push_back(y_val);
        }
    }
}

static void IsotonicPredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (IsotonicPredictAggState **)source_data.data;
    auto targets = (IsotonicPredictAggState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.x_train = std::move(source.x_train);
            target.y_train = std::move(source.y_train);
            target.x_all = std::move(source.x_all);
            target.y_all = std::move(source.y_all);
            target.y_is_null = std::move(source.y_is_null);
            target.is_training = std::move(source.is_training);
            target.initialized = true;
            target.increasing = source.increasing;
            continue;
        }

        target.x_train.insert(target.x_train.end(), source.x_train.begin(), source.x_train.end());
        target.y_train.insert(target.y_train.end(), source.y_train.begin(), source.y_train.end());

        target.x_all.insert(target.x_all.end(), source.x_all.begin(), source.x_all.end());
        target.y_all.insert(target.y_all.end(), source.y_all.begin(), source.y_all.end());
        target.y_is_null.insert(target.y_is_null.end(), source.y_is_null.begin(), source.y_is_null.end());
        target.is_training.insert(target.is_training.end(), source.is_training.begin(), source.is_training.end());
    }
}

// Helper: predict using isotonic model (piecewise constant interpolation)
static double IsotonicPredict(const vector<double> &sorted_x, const vector<double> &fitted_y, double x_new) {
    if (sorted_x.empty()) {
        return std::nan("");
    }

    // Binary search to find position
    auto it = std::lower_bound(sorted_x.begin(), sorted_x.end(), x_new);

    if (it == sorted_x.begin()) {
        return fitted_y.front();
    }
    if (it == sorted_x.end()) {
        return fitted_y.back();
    }

    // Linear interpolation between two nearest points
    idx_t upper_idx = it - sorted_x.begin();
    idx_t lower_idx = upper_idx - 1;

    double x_lower = sorted_x[lower_idx];
    double x_upper = sorted_x[upper_idx];
    double y_lower = fitted_y[lower_idx];
    double y_upper = fitted_y[upper_idx];

    if (x_upper == x_lower) {
        return y_lower;
    }

    double t = (x_new - x_lower) / (x_upper - x_lower);
    return y_lower + t * (y_upper - y_lower);
}

static void IsotonicPredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                        idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (IsotonicPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data for fitting
        AnofoxDataArray x_array;
        x_array.data = state.x_train.data();
        x_array.validity = nullptr;
        x_array.len = state.x_train.size();

        AnofoxDataArray y_array;
        y_array.data = state.y_train.data();
        y_array.validity = nullptr;
        y_array.len = state.y_train.size();

        AnofoxIsotonicOptions options;
        options.increasing = state.increasing;

        AnofoxIsotonicFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_isotonic_fit(x_array, y_array, options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build sorted x and fitted_y for prediction
        // The fitted values are in the same order as input, so we need to sort by x
        vector<pair<double, double>> xy_pairs;
        for (idx_t j = 0; j < core_result.fitted_values_len; j++) {
            xy_pairs.push_back({state.x_train[j], core_result.fitted_values[j]});
        }
        std::sort(xy_pairs.begin(), xy_pairs.end());

        state.sorted_x.clear();
        state.fitted_y.clear();
        for (auto &p : xy_pairs) {
            state.sorted_x.push_back(p.first);
            state.fitted_y.push_back(p.second);
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

            FlatVector::GetData<double>(x_vec)[child_idx] = state.x_all[row];

            // Predict using isotonic model
            double yhat = IsotonicPredict(state.sorted_x, state.fitted_y, state.x_all[row]);
            FlatVector::GetData<double>(yhat_vec)[child_idx] = yhat;

            FlatVector::GetData<bool>(is_training_vec)[child_idx] = state.is_training[row];
        }

        anofox_free_isotonic_result(&core_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> IsotonicPredictAggBind(ClientContext &context, AggregateFunction &function,
                                                        vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<IsotonicPredictAggBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.increasing.has_value()) {
            result->increasing = opts.increasing.value();
        }
    }

    function.return_type = GetIsotonicPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("isotonic_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterIsotonicFitPredictAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_isotonic_fit_predict_agg");

    // isotonic_fit_predict_agg(y, x) - y is response, x is 1D input
    auto basic_func =
        AggregateFunction("anofox_stats_isotonic_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE},
                          LogicalType::ANY, AggregateFunction::StateSize<IsotonicPredictAggState>, IsotonicPredictAggInitialize,
                          IsotonicPredictAggUpdate, IsotonicPredictAggCombine, IsotonicPredictAggFinalize, nullptr, IsotonicPredictAggBind,
                          IsotonicPredictAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_isotonic_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<IsotonicPredictAggState>, IsotonicPredictAggInitialize, IsotonicPredictAggUpdate,
        IsotonicPredictAggCombine, IsotonicPredictAggFinalize, nullptr, IsotonicPredictAggBind, IsotonicPredictAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    AggregateFunctionSet alias_set("isotonic_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
