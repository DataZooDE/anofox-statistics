#!/usr/bin/env python3
"""
Generate remaining inference and predict_interval functions for Anofox Statistics extension.

This script creates the .hpp and .cpp files for:
- Ridge predict_interval (cpp only, hpp already exists)
- WLS predict_interval
- RLS predict_interval
- Elastic Net predict_interval

Based on the templates established by the existing inference functions.
"""

import os
from pathlib import Path

# Base directory (script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src" / "functions" / "inference"

# Ensure output directory exists
SRC_DIR.mkdir(parents=True, exist_ok=True)


def generate_predict_interval_hpp(method_name, method_title, extra_params="", extra_notes=""):
    """Generate header file for predict_interval function."""

    template = f'''#pragma once

#include "duckdb.hpp"

namespace duckdb {{
namespace anofox_statistics {{

/**
 * @brief {method_title} Prediction Intervals
 *
 * Fits {method_title} regression on training data and returns predictions with
 * confidence or prediction intervals for new observations.
 *{extra_notes}
 * Usage:
 *   SELECT * FROM anofox_statistics_{method_name}_predict_interval(
 *       y_train := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x_train := [[1.0], [2.0], [3.0]]::DOUBLE[][],{extra_params}
 *       x_new := [[4.0], [5.0]]::DOUBLE[][],
 *       options := MAP{{'confidence_level': 0.95, 'interval_type': 'prediction'}}
 *   );
 *
 * Returns one row per new observation with:
 * - observation_id: Row number (1-indexed)
 * - predicted: Point prediction
 * - ci_lower: Lower interval bound
 * - ci_upper: Upper interval bound
 * - se: Standard error of prediction
 */
class {method_title.replace(" ", "")}PredictIntervalFunction {{
public:
	static void Register(ExtensionLoader &loader);
}};

}} // namespace anofox_statistics
}} // namespace duckdb
'''
    return template


def generate_predict_interval_cpp(method_name, method_title, class_name, fit_function, extra_params_sig="", extra_params_extract="", extra_validation=""):
    """Generate implementation file for predict_interval function."""

    template = f'''#include "{method_name}_prediction_intervals.hpp"
#include "../utils/tracing.hpp"
#include "../bridge/libanostat_wrapper.hpp"
#include "../bridge/type_converters.hpp"
#include "../utils/options_parser.hpp"
#include "../utils/validation.hpp"
#include "../utils/statistical_distributions.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {{
namespace anofox_statistics {{

struct {class_name}PredictIntervalBindData : public FunctionData {{
	vector<double> predictions;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<double> std_errors;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {{
		auto result = make_uniq<{class_name}PredictIntervalBindData>();
		result->predictions = predictions;
		result->ci_lowers = ci_lowers;
		result->ci_uppers = ci_uppers;
		result->std_errors = std_errors;
		result->current_row = current_row;
		return std::move(result);
	}}

	bool Equals(const FunctionData &other) const override {{
		return false;
	}}
}};

static unique_ptr<FunctionData> {class_name}PredictIntervalBind(ClientContext &context, TableFunctionBindInput &input,
                                                                 vector<LogicalType> &return_types, vector<string> &names) {{

	auto bind_data = make_uniq<{class_name}PredictIntervalBindData>();

	// Get parameters
	auto &y_train_value = input.inputs[0];
	auto &x_train_value = input.inputs[1];
{extra_params_extract}	auto &x_new_value = input.inputs[{2 + extra_params_extract.count("input.inputs")}];
	auto &options_value = input.inputs[{3 + extra_params_extract.count("input.inputs")}];

	// Parse options
	RegressionOptions options = RegressionOptions::ParseFromMap(options_value);
{extra_validation}
	double confidence_level = options.confidence_level;
	bool add_intercept = options.intercept;
	bool is_prediction_interval = true;

	if (options_value.type().id() == LogicalTypeId::MAP) {{
		// Check interval_type in options
		auto &map_value = options_value;
		auto &map_keys = StructValue::GetChildren(map_value);
		for (size_t i = 0; i < map_keys.size(); i += 2) {{
			if (map_keys[i].ToString() == "interval_type") {{
				string interval_type = map_keys[i + 1].ToString();
				is_prediction_interval = (interval_type == "prediction");
				break;
			}}
		}}
	}}

	// Extract y_train
	vector<double> y_train;
	auto &y_list = ListValue::GetChildren(y_train_value);
	for (auto &val : y_list) {{
		y_train.push_back(val.GetValue<double>());
	}}

	idx_t n_train = y_train.size();

	// Extract X_train matrix
	vector<vector<double>> x_train_matrix;
	auto &x_train_outer = ListValue::GetChildren(x_train_value);

	for (auto &row_val : x_train_outer) {{
		auto &row_list = ListValue::GetChildren(row_val);
		vector<double> row;
		for (auto &val : row_list) {{
			row.push_back(val.GetValue<double>());
		}}
		x_train_matrix.push_back(row);
	}}

	if (x_train_matrix.empty() || x_train_matrix[0].empty()) {{
		throw InvalidInputException("X_train cannot be empty");
	}}

	idx_t p = x_train_matrix[0].size();

	// Extract X_new matrix
	vector<vector<double>> x_new_matrix;
	auto &x_new_outer = ListValue::GetChildren(x_new_value);

	for (auto &row_val : x_new_outer) {{
		auto &row_list = ListValue::GetChildren(row_val);
		vector<double> row;
		for (auto &val : row_list) {{
			row.push_back(val.GetValue<double>());
		}}
		x_new_matrix.push_back(row);
	}}

	idx_t n_new = x_new_matrix.size();

	// Fit {method_title} model with full output
	auto result = bridge::LibanostatWrapper::{fit_function}(y_train, x_train_matrix, {extra_params_sig}options, true);

	Eigen::VectorXd coefficients = result.coefficients;
	double intercept = result.intercept;
	idx_t rank = result.rank;
	idx_t df = n_train - rank;

	if (df == 0) {{
		throw InvalidInputException("Not enough observations for effective parameters: n=%llu, rank=%llu", n_train, rank);
	}}

	double mse = result.mse;
	double alpha = 1.0 - confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, static_cast<int>(df));

	// Build X matrix for training data
	Eigen::MatrixXd X_train(n_train, p);
	for (idx_t i = 0; i < n_train; i++) {{
		for (idx_t j = 0; j < p; j++) {{
			X_train(i, j) = x_train_matrix[i][j];
		}}
	}}

	// Compute (X'X)^(-1) for leverage calculation (using non-aliased features only)
	Eigen::MatrixXd XtX_inv;
	idx_t n_valid = 0;
	for (idx_t j = 0; j < p; j++) {{
		if (!std::isnan(coefficients(j)) && !result.is_aliased[j]) {{
			n_valid++;
		}}
	}}

	if (n_valid > 0) {{
		Eigen::MatrixXd X_valid(n_train, n_valid);
		idx_t valid_idx = 0;
		for (idx_t j = 0; j < p; j++) {{
			if (!std::isnan(coefficients(j)) && !result.is_aliased[j]) {{
				X_valid.col(valid_idx) = X_train.col(j);
				valid_idx++;
			}}
		}}
		Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
		XtX_inv = XtX.inverse();
	}} else {{
		XtX_inv = Eigen::MatrixXd::Identity(1, 1);
	}}

	// Make predictions for each new observation
	for (idx_t i = 0; i < n_new; i++) {{
		// Build x_new vector
		Eigen::VectorXd x_new(p);
		for (idx_t j = 0; j < p; j++) {{
			x_new(j) = x_new_matrix[i][j];
		}}

		// Point prediction: y_pred = intercept + beta' * x_new
		double y_pred = intercept;
		for (idx_t j = 0; j < p; j++) {{
			if (!std::isnan(coefficients(j))) {{
				y_pred += coefficients(j) * x_new(j);
			}}
		}}

		// Compute leverage for interval calculation
		double leverage = 0.0;
		if (n_valid > 0) {{
			Eigen::VectorXd x_valid(n_valid);
			idx_t valid_idx = 0;
			for (idx_t j = 0; j < p; j++) {{
				if (!std::isnan(coefficients(j)) && !result.is_aliased[j]) {{
					x_valid(valid_idx) = x_new(j);
					valid_idx++;
				}}
			}}
			leverage = x_valid.transpose() * XtX_inv * x_valid;
		}}

		// Standard error and interval
		double se;
		if (is_prediction_interval) {{
			// Prediction interval: SE = sqrt(MSE * (1 + leverage))
			se = std::sqrt(mse * (1.0 + leverage));
		}} else {{
			// Confidence interval: SE = sqrt(MSE * leverage)
			se = std::sqrt(mse * leverage);
		}}

		double ci_lower = y_pred - t_crit * se;
		double ci_upper = y_pred + t_crit * se;

		bind_data->predictions.push_back(y_pred);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->std_errors.push_back(se);

		ANOFOX_DEBUG("{method_title} Prediction " << i << ": " << y_pred << " [" << ci_lower << ", " << ci_upper << "]");
	}}

	// Define return types
	names = {{"observation_id", "predicted", "ci_lower", "ci_upper", "se"}};
	return_types = {{LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE}};

	return std::move(bind_data);
}}

static void {class_name}PredictIntervalTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {{
	auto &bind_data = data_p.bind_data->CastNoConst<{class_name}PredictIntervalBindData>();

	idx_t n_results = bind_data.predictions.size();
	idx_t rows_to_output = std::min((idx_t)STANDARD_VECTOR_SIZE, n_results - bind_data.current_row);

	if (rows_to_output == 0) {{
		return;
	}}

	output.SetCardinality(rows_to_output);

	auto id_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto pred_data = FlatVector::GetData<double>(output.data[1]);
	auto ci_lower_data = FlatVector::GetData<double>(output.data[2]);
	auto ci_upper_data = FlatVector::GetData<double>(output.data[3]);
	auto se_data = FlatVector::GetData<double>(output.data[4]);

	for (idx_t i = 0; i < rows_to_output; i++) {{
		idx_t idx = bind_data.current_row + i;

		id_data[i] = idx + 1; // 1-indexed
		pred_data[i] = bind_data.predictions[idx];
		ci_lower_data[i] = bind_data.ci_lowers[idx];
		ci_upper_data[i] = bind_data.ci_uppers[idx];
		se_data[i] = bind_data.std_errors[idx];
	}}

	bind_data.current_row += rows_to_output;
}}

void {class_name}PredictIntervalFunction::Register(ExtensionLoader &loader) {{
	ANOFOX_DEBUG("Registering {method_title} predict_interval function");

	TableFunction func("anofox_statistics_{method_name}_predict_interval",
	                   {{LogicalType::LIST(LogicalType::DOUBLE),                    // y_train
	                    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x_train{extra_params_sig.replace("weights, ", "\n\t                    LogicalType::LIST(LogicalType::DOUBLE),                    // weights\n\t                    ")}
	                    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x_new
	                    LogicalType::MAP(LogicalType::VARCHAR, LogicalType::ANY)}},// options
	                   {class_name}PredictIntervalTableFunc, {class_name}PredictIntervalBind);

	loader.RegisterFunction(func);

	ANOFOX_DEBUG("{method_title} predict_interval function registered successfully");
}}

}} // namespace anofox_statistics
}} // namespace duckdb
'''
    return template


# Function definitions
PREDICT_INTERVAL_FUNCTIONS = [
    {
        "method_name": "ridge",
        "method_title": "Ridge",
        "class_name": "Ridge",
        "fit_function": "FitRidge",
        "extra_params": "",
        "extra_params_sig": "",
        "extra_params_extract": "",
        "extra_validation": "\n\t// Ridge requires lambda\n\tif (options.lambda <= 0.0) {\n\t\tthrow InvalidInputException(\"Ridge prediction requires lambda > 0 (got lambda=%f)\", options.lambda);\n\t}",
        "extra_notes": "\n * Note: Uses Ridge regression with L2 regularization."
    },
    {
        "method_name": "wls",
        "method_title": "WLS",
        "class_name": "Wls",
        "fit_function": "FitWLS",
        "extra_params": "\n *       weights := [1.0, 2.0, 1.5]::DOUBLE[],",
        "extra_params_sig": "weights, ",
        "extra_params_extract": "\tauto &weights_value = input.inputs[2];\n\t// Extract weights\n\tvector<double> weights;\n\tauto &weights_list = ListValue::GetChildren(weights_value);\n\tfor (auto &val : weights_list) {\n\t\tdouble w = val.GetValue<double>();\n\t\tif (w <= 0.0) {\n\t\t\tthrow InvalidInputException(\"All weights must be positive, got %f\", w);\n\t\t}\n\t\tweights.push_back(w);\n\t}\n\n\tif (weights.size() != y_train.size()) {\n\t\tthrow InvalidInputException(\"Length mismatch: y_train has %llu observations, weights has %llu\", y_train.size(), weights.size());\n\t}\n\n",
        "extra_validation": "",
        "extra_notes": "\n * Note: Accounts for heteroscedasticity via observation weights."
    },
    {
        "method_name": "rls",
        "method_title": "RLS",
        "class_name": "Rls",
        "fit_function": "FitRLS",
        "extra_params": "",
        "extra_params_sig": "",
        "extra_params_extract": "",
        "extra_validation": "",
        "extra_notes": "\n * Note: Uses Recursive Least Squares with optional forgetting factor."
    },
    {
        "method_name": "elastic_net",
        "method_title": "Elastic Net",
        "class_name": "ElasticNet",
        "fit_function": "FitElasticNet",
        "extra_params": "",
        "extra_params_sig": "",
        "extra_params_extract": "",
        "extra_validation": "\n\t// Elastic Net requires alpha and lambda\n\tif (options.alpha < 0.0 || options.alpha > 1.0) {\n\t\tthrow InvalidInputException(\"Elastic Net requires 0 <= alpha <= 1 (got alpha=%f)\", options.alpha);\n\t}\n\tif (options.lambda <= 0.0) {\n\t\tthrow InvalidInputException(\"Elastic Net requires lambda > 0 (got lambda=%f)\", options.lambda);\n\t}",
        "extra_notes": "\n * Note: Combines L1 and L2 regularization for sparse solutions."
    }
]


def main():
    print("Generating predict_interval functions...")

    generated_files = []

    for func in PREDICT_INTERVAL_FUNCTIONS:
        # Generate .hpp file
        if func["method_name"] != "ridge":  # Ridge hpp already exists
            hpp_path = SRC_DIR / f"{func['method_name']}_prediction_intervals.hpp"
            hpp_content = generate_predict_interval_hpp(
                func["method_name"],
                func["method_title"],
                func["extra_params"],
                func["extra_notes"]
            )

            with open(hpp_path, 'w') as f:
                f.write(hpp_content)

            generated_files.append(str(hpp_path))
            print(f"✓ Created {hpp_path.name}")

        # Generate .cpp file
        cpp_path = SRC_DIR / f"{func['method_name']}_prediction_intervals.cpp"
        cpp_content = generate_predict_interval_cpp(
            func["method_name"],
            func["method_title"],
            func["class_name"],
            func["fit_function"],
            func["extra_params_sig"],
            func["extra_params_extract"],
            func["extra_validation"]
        )

        with open(cpp_path, 'w') as f:
            f.write(cpp_content)

        generated_files.append(str(cpp_path))
        print(f"✓ Created {cpp_path.name}")

    print(f"\n✅ Successfully generated {len(generated_files)} files:")
    for f in generated_files:
        print(f"   - {f}")

    print("\n⚠️  Next steps:")
    print("1. Run: python3 scripts/generate_predict_aggregates.py")
    print("2. Update CMakeLists.txt with new source files")
    print("3. Update src/anofox_statistics_extension.cpp with new registrations")
    print("4. Build: make release")


if __name__ == "__main__":
    main()
