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

static LogicalType GetHuberResultType(bool compute_inference) {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));
    children.push_back(make_pair("scale", LogicalType::DOUBLE));
    children.push_back(make_pair("n_outliers", LogicalType::BIGINT));

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

struct HuberFitBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;
    double epsilon = 1.35;
    double alpha = 0.0001;
    uint32_t max_iterations = 100;
    double tolerance = 1e-5;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<HuberFitBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        result->epsilon = epsilon;
        result->alpha = alpha;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<HuberFitBindData>();
        return fit_intercept == other.fit_intercept && compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level && epsilon == other.epsilon && alpha == other.alpha &&
               max_iterations == other.max_iterations && tolerance == other.tolerance;
    }
};

static unique_ptr<FunctionData> HuberFitBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<HuberFitBindData>();

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
        if (opts.epsilon.has_value()) {
            result->epsilon = opts.epsilon.value();
        }
        if (opts.alpha.has_value()) {
            result->alpha = opts.alpha.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
    }

    bound_function.return_type = GetHuberResultType(result->compute_inference);

    PostHogTelemetry::Instance().CaptureFunctionExecution("huber_fit");
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

static void HuberFitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<HuberFitBindData>();

    auto &y_vec = args.data[0]; // LIST(DOUBLE)
    auto &x_vec = args.data[1]; // LIST(LIST(DOUBLE))

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

        AnofoxHuberOptions options;
        options.epsilon = bind_data.epsilon;
        options.alpha = bind_data.alpha;
        options.fit_intercept = bind_data.fit_intercept;
        options.compute_inference = bind_data.compute_inference;
        options.confidence_level = bind_data.confidence_level;
        options.max_iterations = bind_data.max_iterations;
        options.tolerance = bind_data.tolerance;

        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxHuberFitExtras extras_result;
        AnofoxError error;

        bool success = anofox_huber_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                        bind_data.compute_inference ? &inference_result : nullptr, &extras_result,
                                        &error);

        if (!success) {
            throw InvalidInputException("Huber fit failed: " + string(error.message));
        }

        auto &struct_vec = StructVector::GetEntries(result);
        idx_t struct_idx = 0;

        // Coefficients list
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
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = extras_result.scale;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = (int64_t)extras_result.n_outliers;

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

        anofox_free_huber_extras(&extras_result);
        anofox_free_result_core(&core_result);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

void RegisterHuberFitFunction(ExtensionLoader &loader) {
    ScalarFunction basic_func(
        {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
        LogicalType::ANY, HuberFitFunction, HuberFitBind);

    ScalarFunction map_func({LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), LogicalType::ANY},
                            LogicalType::ANY, HuberFitFunction, HuberFitBind);

    {
        ScalarFunctionSet func_set("anofox_stats_huber_fit");
        func_set.AddFunction(basic_func);
        func_set.AddFunction(map_func);
        CreateScalarFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

        FunctionDescription d1;
        d1.description = "Fits a Huber M-estimator robust regression model. Returns coefficients, fit statistics, the "
                         "MAD-based scale, and the outlier count as a struct.";
        d1.examples = {"anofox_stats_huber_fit(y, x)"};
        d1.categories = {"regression"};
        d1.parameter_names = {"y", "x"};
        d1.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))};
        info.descriptions.push_back(std::move(d1));

        FunctionDescription d2;
        d2.description = "Fits a Huber M-estimator regression model with optional MAP of settings (epsilon, alpha, "
                         "fit_intercept, compute_inference, confidence_level, max_iterations, tolerance).";
        d2.examples = {"anofox_stats_huber_fit(y, x, {'epsilon': 1.35, 'alpha': 0.01})"};
        d2.categories = {"regression"};
        d2.parameter_names = {"y", "x", "options"};
        d2.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), LogicalType::ANY};
        info.descriptions.push_back(std::move(d2));

        loader.RegisterFunction(std::move(info));
    }
    {
        ScalarFunctionSet alias_set("huber_fit");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateScalarFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_huber_fit";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
