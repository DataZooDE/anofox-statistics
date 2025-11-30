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
	vector<double> std_residuals; // Standardized using sample SD
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

//===--------------------------------------------------------------------===//
// Lateral Join Support Structures (must be declared before bind function)
//===--------------------------------------------------------------------===//

/**
 * Bind data for in-out mode (lateral join support)
 * Stores only options, not data (data comes from input rows)
 */
struct ResidualDiagnosticsInOutBindData : public FunctionData {
	double outlier_threshold = 2.5;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ResidualDiagnosticsInOutBindData>();
		result->outlier_threshold = outlier_threshold;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Local state for in-out mode
 * Tracks which input rows have been processed
 */
struct ResidualDiagnosticsInOutLocalState : public LocalTableFunctionState {
	idx_t current_input_row = 0;
};

static unique_ptr<LocalTableFunctionState> ResidualDiagnosticsInOutLocalInit(ExecutionContext &context,
                                                                             TableFunctionInitInput &input,
                                                                             GlobalTableFunctionState *global_state) {
	return make_uniq<ResidualDiagnosticsInOutLocalState>();
}

/**
 * Helper function to compute residual diagnostics
 * Extracted for reuse in both literal and lateral join modes
 */
static void ComputeResidualDiagnostics(const vector<double> &y_actual, const vector<double> &y_predicted,
                                       double outlier_threshold, ResidualDiagnosticsBindData &result) {
	idx_t n = y_actual.size();

	if (y_predicted.size() != n) {
		throw InvalidInputException("y_actual and y_predicted must have the same length: got %d and %d", n,
		                            y_predicted.size());
	}

	if (n < 3) {
		throw InvalidInputException("Need at least 3 observations for residual diagnostics, got %d", n);
	}

	result.outlier_threshold = outlier_threshold;

	// Compute residuals
	result.residuals.resize(n);
	for (idx_t i = 0; i < n; i++) {
		result.residuals[i] = y_actual[i] - y_predicted[i];
	}

	// Compute standardized residuals (z-scores)
	double mean_residual = 0.0;
	for (idx_t i = 0; i < n; i++) {
		mean_residual += result.residuals[i];
	}
	mean_residual /= static_cast<double>(n);

	double variance = 0.0;
	for (idx_t i = 0; i < n; i++) {
		double diff = result.residuals[i] - mean_residual;
		variance += diff * diff;
	}
	double sd = std::sqrt(variance / static_cast<double>(n - 1));

	result.std_residuals.resize(n);
	result.is_outlier.resize(n);
	for (idx_t i = 0; i < n; i++) {
		if (sd > 1e-10) {
			result.std_residuals[i] = (result.residuals[i] - mean_residual) / sd;
			result.is_outlier[i] = std::abs(result.std_residuals[i]) > outlier_threshold;
		} else {
			result.std_residuals[i] = 0.0;
			result.is_outlier[i] = false;
		}
	}
}

/**
 * Bind function - Compute residual diagnostics from y_actual and y_predicted
 */
static unique_ptr<FunctionData> ResidualDiagnosticsBind(ClientContext &context, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {

	// Set return schema first (needed for both literal and lateral join modes)
	names = {"obs_id", "residual", "std_residual", "is_outlier"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN};

	// Check if this is being called for lateral joins (in-out function mode)
	// In that case, we don't have literal values to process
	if (input.inputs.size() >= 2 && !input.inputs[0].IsNull()) {
		// Check if the first input is actually a constant value
		try {
			auto y_actual_list = ListValue::GetChildren(input.inputs[0]);
			// If we can get children, it's a literal value - process it normally

			auto bind_data = make_uniq<ResidualDiagnosticsBindData>();

			double outlier_threshold = 2.5; // Default
			if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
				outlier_threshold = input.inputs[2].GetValue<double>();
			}

			// Extract y_actual array
			vector<double> y_actual;
			for (auto &val : y_actual_list) {
				y_actual.push_back(val.GetValue<double>());
			}

			// Extract y_predicted array
			auto y_predicted_list = ListValue::GetChildren(input.inputs[1]);
			vector<double> y_predicted;
			for (auto &val : y_predicted_list) {
				y_predicted.push_back(val.GetValue<double>());
			}

			// Compute diagnostics using helper function
			ComputeResidualDiagnostics(y_actual, y_predicted, outlier_threshold, *bind_data);

			return std::move(bind_data);

		} catch (...) {
			// If we can't get children, it's probably a column reference (lateral join mode)
			// Return minimal bind data for in-out function mode
			auto result = make_uniq<ResidualDiagnosticsInOutBindData>();

			// Extract outlier_threshold if provided as literal
			if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
				try {
					result->outlier_threshold = input.inputs[2].GetValue<double>();
				} catch (...) {
					// Ignore if we can't parse threshold
				}
			}

			return std::move(result);
		}
	}

	// Fallback: return minimal bind data
	auto result = make_uniq<ResidualDiagnosticsInOutBindData>();
	return std::move(result);
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

/**
 * In-out function for lateral join support
 * Processes rows from input table, computes diagnostics for each row
 */
static OperatorResultType ResidualDiagnosticsInOut(ExecutionContext &context, TableFunctionInput &data_p,
                                                   DataChunk &input, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<ResidualDiagnosticsInOutBindData>();
	auto &state = data_p.local_state->Cast<ResidualDiagnosticsInOutLocalState>();

	// Process all input rows
	if (state.current_input_row >= input.size()) {
		// Finished processing all rows in this chunk
		state.current_input_row = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Flatten input vectors for easier access
	input.Flatten();

	idx_t output_idx = 0;
	while (state.current_input_row < input.size() && output_idx < STANDARD_VECTOR_SIZE) {
		idx_t row_idx = state.current_input_row;

		// Check for NULL inputs
		if (FlatVector::IsNull(input.data[0], row_idx) || FlatVector::IsNull(input.data[1], row_idx)) {
			// Skip NULL rows
			state.current_input_row++;
			continue;
		}

		// Extract y_actual from column 0 (LIST of DOUBLE)
		auto y_actual_list = FlatVector::GetValue<list_entry_t>(input.data[0], row_idx);
		auto &y_actual_child = ListVector::GetEntry(input.data[0]);
		vector<double> y_actual;
		for (idx_t i = y_actual_list.offset; i < y_actual_list.offset + y_actual_list.length; i++) {
			if (!FlatVector::IsNull(y_actual_child, i)) {
				y_actual.push_back(FlatVector::GetValue<double>(y_actual_child, i));
			}
		}

		// Extract y_predicted from column 1 (LIST of DOUBLE)
		auto y_predicted_list = FlatVector::GetValue<list_entry_t>(input.data[1], row_idx);
		auto &y_predicted_child = ListVector::GetEntry(input.data[1]);
		vector<double> y_predicted;
		for (idx_t i = y_predicted_list.offset; i < y_predicted_list.offset + y_predicted_list.length; i++) {
			if (!FlatVector::IsNull(y_predicted_child, i)) {
				y_predicted.push_back(FlatVector::GetValue<double>(y_predicted_child, i));
			}
		}

		// Compute diagnostics
		ResidualDiagnosticsBindData temp_data;
		try {
			ComputeResidualDiagnostics(y_actual, y_predicted, bind_data.outlier_threshold, temp_data);
		} catch (const Exception &e) {
			// If computation fails for this row, skip it
			state.current_input_row++;
			continue;
		}

		// Output all diagnostics for this input row
		idx_t n = temp_data.residuals.size();
		for (idx_t i = 0; i < n && output_idx < STANDARD_VECTOR_SIZE; i++, output_idx++) {
			// Column 0: obs_id (1-indexed)
			FlatVector::GetData<int64_t>(output.data[0])[output_idx] = static_cast<int64_t>(i + 1);

			// Column 1: residual
			FlatVector::GetData<double>(output.data[1])[output_idx] = temp_data.residuals[i];

			// Column 2: std_residual
			FlatVector::GetData<double>(output.data[2])[output_idx] = temp_data.std_residuals[i];

			// Column 3: is_outlier
			FlatVector::GetData<bool>(output.data[3])[output_idx] = temp_data.is_outlier[i];
		}

		state.current_input_row++;

		// If we filled the output buffer mid-row, break and continue later
		if (output_idx >= STANDARD_VECTOR_SIZE) {
			break;
		}
	}

	output.SetCardinality(output_idx);

	if (output_idx > 0) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	} else {
		return OperatorResultType::NEED_MORE_INPUT;
	}
}

void ResidualDiagnosticsFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_stats_residual_diagnostics (with alias residual_diagnostics, dual mode: literals "
	             "+ lateral joins)");

	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE), // y_actual
	    LogicalType::LIST(LogicalType::DOUBLE)  // y_predicted
	};

	// Register with literal mode (bind + execute)
	TableFunction residual_diagnostics_func("anofox_stats_residual_diagnostics", arguments,
	                                        ResidualDiagnosticsTableFunc, ResidualDiagnosticsBind, nullptr,
	                                        ResidualDiagnosticsInOutLocalInit);

	// Add lateral join support (in_out_function)
	residual_diagnostics_func.in_out_function = ResidualDiagnosticsInOut;
	residual_diagnostics_func.varargs = LogicalType::ANY;

	// Set named parameters
	residual_diagnostics_func.named_parameters["y_actual"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_func.named_parameters["y_predicted"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_func.named_parameters["outlier_threshold"] = LogicalType::DOUBLE;

	loader.RegisterFunction(residual_diagnostics_func);

	// Register alias
	TableFunction residual_diagnostics_alias("residual_diagnostics", arguments, ResidualDiagnosticsTableFunc,
	                                         ResidualDiagnosticsBind, nullptr, ResidualDiagnosticsInOutLocalInit);

	// Add lateral join support (in_out_function)
	residual_diagnostics_alias.in_out_function = ResidualDiagnosticsInOut;
	residual_diagnostics_alias.varargs = LogicalType::ANY;

	// Set named parameters
	residual_diagnostics_alias.named_parameters["y_actual"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_alias.named_parameters["y_predicted"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_alias.named_parameters["outlier_threshold"] = LogicalType::DOUBLE;

	loader.RegisterFunction(residual_diagnostics_alias);

	ANOFOX_DEBUG(
	    "anofox_stats_residual_diagnostics registered successfully with alias residual_diagnostics (both modes)");
}

} // namespace anofox_statistics
} // namespace duckdb
