#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

static LogicalType GetRansacResultType(bool compute_inference) {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));
    children.push_back(make_pair("residual_threshold", LogicalType::DOUBLE));
    children.push_back(make_pair("n_inliers", LogicalType::BIGINT));
    children.push_back(make_pair("n_trials", LogicalType::BIGINT));

    if (compute_inference) {
        children.push_back(make_pair("std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("t_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("p_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
        children.push_back(make_pair("f_pvalue", LogicalType::DOUBLE));
    }

    return LogicalType::STRUCT(std::move(children));
}

struct RansacFitBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;
    uint32_t max_trials = 100;
    double stop_probability = 0.99;
    uint64_t random_state = 0;
    bool min_samples_set = false;
    uint32_t min_samples_value = 0;
    bool residual_threshold_set = false;
    double residual_threshold_value = 0.0;
    bool stop_n_inliers_set = false;
    uint32_t stop_n_inliers_value = 0;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<RansacFitBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        result->max_trials = max_trials;
        result->stop_probability = stop_probability;
        result->random_state = random_state;
        result->min_samples_set = min_samples_set;
        result->min_samples_value = min_samples_value;
        result->residual_threshold_set = residual_threshold_set;
        result->residual_threshold_value = residual_threshold_value;
        result->stop_n_inliers_set = stop_n_inliers_set;
        result->stop_n_inliers_value = stop_n_inliers_value;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<RansacFitBindData>();
        return fit_intercept == other.fit_intercept && compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level && max_trials == other.max_trials &&
               stop_probability == other.stop_probability && random_state == other.random_state &&
               min_samples_set == other.min_samples_set && min_samples_value == other.min_samples_value &&
               residual_threshold_set == other.residual_threshold_set &&
               residual_threshold_value == other.residual_threshold_value &&
               stop_n_inliers_set == other.stop_n_inliers_set &&
               stop_n_inliers_value == other.stop_n_inliers_value;
    }
};

static unique_ptr<FunctionData> RansacFitBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RansacFitBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.compute_inference.has_value()) {
            result->compute_inference = opts.compute_inference.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.max_trials.has_value()) {
            result->max_trials = opts.max_trials.value();
        }
        if (opts.stop_probability.has_value()) {
            result->stop_probability = opts.stop_probability.value();
        }
        if (opts.random_state.has_value()) {
            result->random_state = opts.random_state.value();
        }
        if (opts.min_samples.has_value()) {
            result->min_samples_set = true;
            result->min_samples_value = opts.min_samples.value();
        }
        if (opts.residual_threshold.has_value()) {
            result->residual_threshold_set = true;
            result->residual_threshold_value = opts.residual_threshold.value();
        }
        if (opts.stop_n_inliers.has_value()) {
            result->stop_n_inliers_set = true;
            result->stop_n_inliers_value = opts.stop_n_inliers.value();
        }
    }

    bound_function.return_type = GetRansacResultType(result->compute_inference);

    PostHogTelemetry::Instance().CaptureFunctionExecution("ransac_fit");
    return std::move(result);
}

static vector<double> ExtractDoubleList(Vector &vec, idx_t row_idx) {
    auto list_data = ListVector::GetData(vec);
    auto &child = ListVector::GetEntry(vec);
    auto child_data = FlatVector::GetData<double>(child);

    vector<double> result;
    auto offset = list_data[row_idx].offset;
    auto length = list_data[row_idx].length;

    for (idx_t i = 0; i < length; i++) {
        result.push_back(child_data[offset + i]);
    }

    return result;
}

static void RansacFitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<RansacFitBindData>();

    auto &y_vec = args.data[0];
    auto &x_vec = args.data[1];

    idx_t count = args.size();

    for (idx_t row = 0; row < count; row++) {
        vector<double> y_data = ExtractDoubleList(y_vec, row);

        auto x_list_data = ListVector::GetData(x_vec);
        auto &x_child = ListVector::GetEntry(x_vec);

        vector<vector<double>> x_cols;
        auto x_offset = x_list_data[row].offset;
        auto x_length = x_list_data[row].length;

        for (idx_t col = 0; col < x_length; col++) {
            x_cols.push_back(ExtractDoubleList(x_child, x_offset + col));
        }

        AnofoxDataArray y_array;
        y_array.data = y_data.data();
        y_array.validity = nullptr;
        y_array.len = y_data.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        AnofoxRansacOptions options;
        options.fit_intercept = bind_data.fit_intercept;
        options.compute_inference = bind_data.compute_inference;
        options.confidence_level = bind_data.confidence_level;
        options.max_trials = bind_data.max_trials;
        options.stop_probability = bind_data.stop_probability;
        options.random_state = bind_data.random_state;
        options.min_samples_set = bind_data.min_samples_set;
        options.min_samples_value = bind_data.min_samples_value;
        options.residual_threshold_set = bind_data.residual_threshold_set;
        options.residual_threshold_value = bind_data.residual_threshold_value;
        options.stop_n_inliers_set = bind_data.stop_n_inliers_set;
        options.stop_n_inliers_value = bind_data.stop_n_inliers_value;

        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxRansacFitExtras extras_result;
        AnofoxError error;

        bool success = anofox_ransac_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                         bind_data.compute_inference ? &inference_result : nullptr, &extras_result,
                                         &error);

        if (!success) {
            throw InvalidInputException("RANSAC fit failed: " + string(error.message));
        }

        auto &struct_vec = StructVector::GetEntries(result);
        idx_t struct_idx = 0;

        auto &coef_list = *struct_vec[struct_idx++];
        auto &coef_child = ListVector::GetEntry(coef_list);
        auto coef_offset = ListVector::GetListSize(coef_list);
        ListVector::SetListSize(coef_list, coef_offset + core_result.coefficients_len);
        auto coef_data = FlatVector::GetData<double>(coef_child);
        for (size_t i = 0; i < core_result.coefficients_len; i++) {
            coef_data[coef_offset + i] = core_result.coefficients[i];
        }
        ListVector::GetData(coef_list)[row] = {coef_offset, core_result.coefficients_len};

        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.intercept;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = core_result.n_features;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = extras_result.residual_threshold;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = (int64_t)extras_result.n_inliers;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = (int64_t)extras_result.n_trials;

        if (bind_data.compute_inference) {
            auto set_list = [&](Vector &list_vec, double *data, size_t len) {
                auto &child = ListVector::GetEntry(list_vec);
                auto offset = ListVector::GetListSize(list_vec);
                ListVector::SetListSize(list_vec, offset + len);
                auto vec_data = FlatVector::GetData<double>(child);
                for (size_t i = 0; i < len; i++) {
                    vec_data[offset + i] = data[i];
                }
                ListVector::GetData(list_vec)[row] = {offset, len};
            };

            set_list(*struct_vec[struct_idx++], inference_result.std_errors, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.t_values, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.p_values, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.ci_lower, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.ci_upper, inference_result.len);

            FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = inference_result.f_statistic;
            FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = inference_result.f_pvalue;

            anofox_free_result_inference(&inference_result);
        }

        anofox_free_ransac_extras(&extras_result);
        anofox_free_result_core(&core_result);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

void RegisterRansacFitFunction(ExtensionLoader &loader) {
    ScalarFunction basic_func(
        {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
        LogicalType::ANY, RansacFitFunction, RansacFitBind);

    ScalarFunction map_func({LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), LogicalType::ANY},
                            LogicalType::ANY, RansacFitFunction, RansacFitBind);

    {
        ScalarFunctionSet func_set("anofox_stats_ransac_fit");
        func_set.AddFunction(basic_func);
        func_set.AddFunction(map_func);
        CreateScalarFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

        FunctionDescription d1;
        d1.description = "Fits a RANSAC robust regression model. Returns coefficients, fit statistics, the residual "
                         "threshold used, and the inlier / trial counts as a struct.";
        d1.examples = {"anofox_stats_ransac_fit(y, x)"};
        d1.categories = {"regression"};
        d1.parameter_names = {"y", "x"};
        d1.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))};
        info.descriptions.push_back(std::move(d1));

        FunctionDescription d2;
        d2.description = "Fits a RANSAC regression model with optional MAP of settings (residual_threshold, "
                         "max_trials, min_samples, stop_probability, stop_n_inliers, random_state, fit_intercept, "
                         "compute_inference, confidence_level).";
        d2.examples = {"anofox_stats_ransac_fit(y, x, {'residual_threshold': 0.5, 'random_state': 42})"};
        d2.categories = {"regression"};
        d2.parameter_names = {"y", "x", "options"};
        d2.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), LogicalType::ANY};
        info.descriptions.push_back(std::move(d2));

        loader.RegisterFunction(std::move(info));
    }
    {
        ScalarFunctionSet alias_set("ransac_fit");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateScalarFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_ransac_fit";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
