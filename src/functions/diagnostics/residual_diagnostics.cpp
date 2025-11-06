#include "residual_diagnostics.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <algorithm>

namespace duckdb {
namespace anofox_statistics {

/**
 * Bind data for residual diagnostics
 * Simplified version that works with (y_actual, y_predicted)
 */
struct ResidualDiagnosticsBindData : public FunctionData {
	vector<double> residuals;
	vector<double> std_residuals;  // Standardized using sample SD
	vector<bool> is_outlier;

	double outlier_threshold;
	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ResidualDiagnosticsBindData>();
		result->residuals = residuals;
		result->std_residuals = std_residuals;
		result->is_outlier = is_outlier;
		result->outlier_threshold = outlier_threshold;
		result->current_row = current_row;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Bind function - Compute residual diagnostics from y_actual and y_predicted
 */
static unique_ptr<FunctionData> ResidualDiagnosticsBind(ClientContext &context, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<ResidualDiagnosticsBindData>();

	// Get parameters: y_actual, y_predicted, outlier_threshold
	auto &y_actual_value = input.inputs[0];
	auto &y_predicted_value = input.inputs[1];

	bind_data->outlier_threshold = 2.5;  // Default
	if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
		bind_data->outlier_threshold = input.inputs[2].GetValue<double>();
	}

	// Extract y_actual array
	vector<double> y_actual;
	auto &y_actual_list = ListValue::GetChildren(y_actual_value);
	for (auto &val : y_actual_list) {
		y_actual.push_back(val.GetValue<double>());
	}

	// Extract y_predicted array
	vector<double> y_predicted;
	auto &y_predicted_list = ListValue::GetChildren(y_predicted_value);
	for (auto &val : y_predicted_list) {
		y_predicted.push_back(val.GetValue<double>());
	}

	idx_t n = y_actual.size();
	if (y_predicted.size() != n) {
		throw InvalidInputException("y_actual and y_predicted must have the same length: got %d and %d", n,
		                            y_predicted.size());
	}

	if (n < 3) {
		throw InvalidInputException("Need at least 3 observations for residual diagnostics, got %d", n);
	}

	// Compute residuals
	bind_data->residuals.resize(n);
	for (idx_t i = 0; i < n; i++) {
		bind_data->residuals[i] = y_actual[i] - y_predicted[i];
	}

	// Compute standardized residuals (z-scores)
	// Mean should be ~0 for residuals, so we can use sample SD directly
	double mean_residual = 0.0;
	for (idx_t i = 0; i < n; i++) {
		mean_residual += bind_data->residuals[i];
	}
	mean_residual /= static_cast<double>(n);

	double variance = 0.0;
	for (idx_t i = 0; i < n; i++) {
		double diff = bind_data->residuals[i] - mean_residual;
		variance += diff * diff;
	}
	double sd = std::sqrt(variance / static_cast<double>(n - 1));

	bind_data->std_residuals.resize(n);
	bind_data->is_outlier.resize(n);
	for (idx_t i = 0; i < n; i++) {
		if (sd > 1e-10) {
			bind_data->std_residuals[i] = (bind_data->residuals[i] - mean_residual) / sd;
			bind_data->is_outlier[i] = std::abs(bind_data->std_residuals[i]) > bind_data->outlier_threshold;
		} else {
			bind_data->std_residuals[i] = 0.0;
			bind_data->is_outlier[i] = false;
		}
	}

	// Set return schema
	names = {"obs_id", "residual", "std_residual", "is_outlier"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN};

	return std::move(bind_data);
}

/**
 * Execute function - Stream results
 */
static void ResidualDiagnosticsTableFunc(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->CastNoConst<ResidualDiagnosticsBindData>();

	idx_t n = bind_data.residuals.size();
	idx_t start_row = bind_data.current_row;
	idx_t rows_to_output = std::min<idx_t>(STANDARD_VECTOR_SIZE, n - start_row);

	if (rows_to_output == 0) {
		return;
	}

	output.SetCardinality(rows_to_output);

	// Column 0: obs_id (1-indexed)
	auto obs_id_data = FlatVector::GetData<int64_t>(output.data[0]);
	for (idx_t i = 0; i < rows_to_output; i++) {
		obs_id_data[i] = static_cast<int64_t>(start_row + i + 1);
	}

	// Column 1: residual
	auto residual_data = FlatVector::GetData<double>(output.data[1]);
	for (idx_t i = 0; i < rows_to_output; i++) {
		residual_data[i] = bind_data.residuals[start_row + i];
	}

	// Column 2: std_residual
	auto std_residual_data = FlatVector::GetData<double>(output.data[2]);
	for (idx_t i = 0; i < rows_to_output; i++) {
		std_residual_data[i] = bind_data.std_residuals[start_row + i];
	}

	// Column 3: is_outlier
	auto is_outlier_data = FlatVector::GetData<bool>(output.data[3]);
	for (idx_t i = 0; i < rows_to_output; i++) {
		is_outlier_data[i] = bind_data.is_outlier[start_row + i];
	}

	bind_data.current_row += rows_to_output;
}

void ResidualDiagnosticsFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering residual diagnostics function");

	TableFunction residual_diagnostics_func("anofox_statistics_residual_diagnostics",
	                                        {LogicalType::LIST(LogicalType::DOUBLE), // y_actual
	                                         LogicalType::LIST(LogicalType::DOUBLE), // y_predicted
	                                         LogicalType::DOUBLE},                   // outlier_threshold
	                                        ResidualDiagnosticsTableFunc, ResidualDiagnosticsBind);

	// Set named parameters
	residual_diagnostics_func.named_parameters["y_actual"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_func.named_parameters["y_predicted"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_func.named_parameters["outlier_threshold"] = LogicalType::DOUBLE;

	loader.RegisterFunction(residual_diagnostics_func);

	ANOFOX_DEBUG("Residual diagnostics function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
