#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "../include/null_rows.hpp"
#include "telemetry.hpp"
#include "anofox_statistics_banner.hpp"

namespace duckdb {

// Result struct type for WLS
static LogicalType GetWlsResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));

    // Stable shape (#152): the inference fields are always present, NULL
    // without compute_inference.
    AppendCoefficientInferenceFields(children);
    AppendModelSummaryFields(children);
    AppendLooFields(children);

    return LogicalType::STRUCT(std::move(children));
}

// Bind data for WLS function
struct WlsFitBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;
    SolverType solver = SolverType::SVD;
    HcType hc_type = HcType::NONE;
    double alias_tolerance = 0.0; // <= 0: upstream default (1e-7, R lm tol)
    bool compute_loo = false;     // press, press_rmse, loo_r_squared (#220): one more QR

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<WlsFitBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        result->solver = solver;
        result->hc_type = hc_type;
        result->alias_tolerance = alias_tolerance;
        result->compute_loo = compute_loo;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<WlsFitBindData>();
        return fit_intercept == other.fit_intercept && compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level && solver == other.solver && hc_type == other.hc_type && alias_tolerance == other.alias_tolerance &&
               compute_loo == other.compute_loo;
    }
};

// Bind function
static unique_ptr<FunctionData> WlsFitBind(ClientContext &context, ScalarBindFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<WlsFitBindData>();

    // Parse MAP options if provided as 4th argument (after y, x, weights)
    if (arguments.size() >= 4) {
        auto opts = RegressionMapOptions::ParseFromExpression(
            context, *arguments[3], "wls_fit",
            {"fit_intercept", "compute_inference", "confidence_level", "solver", "hc_type", "alias_tolerance",
             "compute_loo"});
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.compute_inference.has_value()) {
            result->compute_inference = opts.compute_inference.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.solver.has_value()) {
            result->solver = opts.solver.value();
        }
        if (opts.hc_type.has_value()) {
            result->hc_type = opts.hc_type.value();
        }
        if (opts.alias_tolerance.has_value()) {
            result->alias_tolerance = opts.alias_tolerance.value();
        }
        if (opts.compute_loo.has_value()) {
            result->compute_loo = opts.compute_loo.value();
        }
    }

    // Set return type
    SetReturnTypeOf(bound_function, GetWlsResultType());

    PostHogTelemetry::Instance().RecordFunctionCall("wls_fit");
    return std::move(result);
}

// Extract list of doubles from a DuckDB list value
static vector<double> ExtractDoubleList(Vector &vec, idx_t row_idx) {
    auto list_data = ListVector::GetData(vec);
    auto &child = ListVector::GetEntry(vec);
    auto child_data = FlatDataMut<double>(child);

    vector<double> result;
    auto offset = list_data[row_idx].offset;
    auto length = list_data[row_idx].length;

    for (idx_t i = 0; i < length; i++) {
        result.push_back(child_data[offset + i]);
    }

    return result;
}

// Main WLS fit function
static void WlsFitFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
static void WlsFitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	// The implementation reads list arguments by row index; flatten constant and
	// dictionary inputs first so every row has its own entry (#210).
	args.Flatten();
	// A NULL data argument gives a NULL row; a NULL options argument means no options (#276).
	ExecuteOnNonNullRows(args, state, result, 3, WlsFitFunctionImpl);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void WlsFitFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &bind_data = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<WlsFitBindData>();

    auto &y_vec = args.data[0];       // LIST(DOUBLE)
    auto &x_vec = args.data[1];       // LIST(LIST(DOUBLE)) - list of feature columns
    auto &weights_vec = args.data[2]; // LIST(DOUBLE) - weights

    idx_t count = args.size();

    // Process each row
    for (idx_t row = 0; row < count; row++) {
        // Extract y values
        vector<double> y_data = ExtractDoubleList(y_vec, row);

        // Extract weights
        vector<double> weights_data = ExtractDoubleList(weights_vec, row);

        // Extract x values (list of columns)
        auto x_list_data = ListVector::GetData(x_vec);
        auto &x_child = ListVector::GetEntry(x_vec);

        vector<vector<double>> x_cols;
        auto x_offset = x_list_data[row].offset;
        auto x_length = x_list_data[row].length;

        for (idx_t col = 0; col < x_length; col++) {
            x_cols.push_back(ExtractDoubleList(x_child, x_offset + col));
        }

        // Prepare FFI data
        AnofoxDataArray y_array;
        y_array.data = y_data.data();
        y_array.validity = nullptr;
        y_array.len = y_data.size();

        AnofoxDataArray weights_array;
        weights_array.data = weights_data.data();
        weights_array.validity = nullptr;
        weights_array.len = weights_data.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        // Set options
        AnofoxWlsOptions options {};
        options.fit_intercept = bind_data.fit_intercept;
        options.compute_inference = bind_data.compute_inference;
        options.confidence_level = bind_data.confidence_level;
        options.solver = ConvertSolverType(bind_data.solver);
        options.hc_type = ConvertHcType(bind_data.hc_type);
        options.alias_tolerance = bind_data.alias_tolerance;
        options.compute_loo = bind_data.compute_loo;

        // Call Rust FFI
        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_wls_fit(y_array, x_arrays.data(), x_arrays.size(), weights_array, options, &core_result,
                                      bind_data.compute_inference ? &inference_result : nullptr, &error);

        if (!success) {
            ThrowUnlessDegenerate("wls_fit", error);
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Build result struct
        auto &struct_vec = StructVector::GetEntries(result);
        ModelStructWriter writer(result);
        idx_t struct_idx = 0;

        // Coefficients
        auto &coef_list = StructEntry(struct_vec[struct_idx++]);
        auto &coef_child = ListVector::GetEntry(coef_list);
        auto coef_offset = ListVector::GetListSize(coef_list);
        ListVector::Reserve(coef_list, coef_offset + core_result.coefficients_len);
        ListVector::SetListSize(coef_list, coef_offset + core_result.coefficients_len);
        auto coef_data = FlatDataMut<double>(coef_child);
        for (size_t i = 0; i < core_result.coefficients_len; i++) {
            coef_data[coef_offset + i] = core_result.coefficients[i];
        }
        ListVector::GetData(coef_list)[row] = {coef_offset, core_result.coefficients_len};

        // Scalars
        FlatDataMut<double>(StructEntry(struct_vec[struct_idx++]))[row] = core_result.intercept;
        FlatDataMut<double>(StructEntry(struct_vec[struct_idx++]))[row] = core_result.r_squared;
        FlatDataMut<double>(StructEntry(struct_vec[struct_idx++]))[row] = core_result.adj_r_squared;
        FlatDataMut<double>(StructEntry(struct_vec[struct_idx++]))[row] = core_result.residual_std_error;
        FlatDataMut<int64_t>(StructEntry(struct_vec[struct_idx++]))[row] = core_result.n_observations;
        FlatDataMut<int64_t>(StructEntry(struct_vec[struct_idx++]))[row] = core_result.n_features;

        // Inference (NULL unless requested), CI fields and the model summary
        writer.WriteInference(row, bind_data.compute_inference ? &inference_result : nullptr);
        writer.WriteSummary(row, core_result.summary);
        if (bind_data.compute_inference) {
            anofox_free_result_inference(&inference_result);
        }

        // Free core result
        anofox_free_result_core(&core_result);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the function
void RegisterWlsFitFunction(ExtensionLoader &loader) {
    ScalarFunction basic_func({LogicalType::LIST(LogicalType::DOUBLE),
                               LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                               LogicalType::LIST(LogicalType::DOUBLE)},
                              LogicalType::ANY,
                              DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, WlsFitFunction),
                        ANOFOX_SCALAR_BIND(DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, WlsFitBind)));
    MarkFallible(basic_func);
    ScalarFunction map_func({LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                             LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::ANY},
                            LogicalType::ANY, DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, WlsFitFunction),
                        ANOFOX_SCALAR_BIND(DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, WlsFitBind)));
    MarkFallible(map_func);
    SetNullHandlingOf(map_func, FunctionNullHandling::SPECIAL_HANDLING);

    // Primary
    {
        ScalarFunctionSet func_set("wls_fit");
        func_set.AddFunction(basic_func);
        func_set.AddFunction(map_func);
        CreateScalarFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

        FunctionDescription d1;
        d1.description     = "Fits a Weighted Least Squares (WLS) regression model using per-observation weights.";
        d1.examples        = {"wls_fit(y, x, weights)"};
        d1.categories      = {"regression"};
        d1.parameter_names = {"y", "x", "weights"};
        d1.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                              LogicalType::LIST(LogicalType::DOUBLE)};
        info.descriptions.push_back(std::move(d1));

        FunctionDescription d2;
        d2.description     = "Fits a WLS regression model with optional MAP of settings (fit_intercept, compute_inference, confidence_level, solver, hc_type, alias_tolerance, compute_loo).";
        d2.examples        = {"wls_fit(y, x, weights, {'compute_inference': true})"};
        d2.categories      = {"regression"};
        d2.parameter_names = {"y", "x", "weights", "options"};
        d2.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                              LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::ANY};
        info.descriptions.push_back(std::move(d2));

        loader.RegisterFunction(std::move(info));
    }
    // Alias
}

} // namespace duckdb
