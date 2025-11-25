#include "vif.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"
#include "../bridge/type_converters.hpp"
#include "libanostat/diagnostics/vif.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * Bind data for VIF
 */
struct VifBindData : public FunctionData {
	vector<string> variable_names;
	vector<double> vif_values;
	vector<string> severities;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<VifBindData>();
		result->variable_names = variable_names;
		result->vif_values = vif_values;
		result->severities = severities;
		result->current_row = current_row;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Bind function - Compute VIF for each variable
 */
static unique_ptr<FunctionData> VifBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<VifBindData>();

	// Get X matrix parameter
	auto &x_value = input.inputs[0];

	// Extract X matrix
	vector<vector<double>> x_matrix;
	auto &x_outer_list = ListValue::GetChildren(x_value);

	for (auto &row_val : x_outer_list) {
		auto &row_list = ListValue::GetChildren(row_val);
		vector<double> row;
		for (auto &val : row_list) {
			row.push_back(val.GetValue<double>());
		}
		x_matrix.push_back(row);
	}

	if (x_matrix.empty() || x_matrix[0].empty()) {
		throw InvalidInputException("X matrix cannot be empty");
	}

	idx_t n = x_matrix.size();    // Number of observations
	idx_t p = x_matrix[0].size(); // Number of variables

	if (p < 2) {
		throw InvalidInputException("Need at least 2 variables to compute VIF, got %llu", p);
	}

	if (n <= p) {
		throw InvalidInputException("Need more observations than variables: n=%llu, p=%llu", n, p);
	}

	// Build X matrix
	Eigen::MatrixXd X(n, p);
	for (idx_t i = 0; i < n; i++) {
		for (idx_t j = 0; j < p; j++) {
			X(i, j) = x_matrix[i][j];
		}
	}

	// Use libanostat VIFCalculator
	// VIF is computed WITH intercept (standard definition)
	libanostat::core::RegressionOptions opts = libanostat::core::RegressionOptions::OLS();
	opts.intercept = true;

	try {
		auto vif_results = libanostat::diagnostics::VIFCalculator::ComputeVIF(X, opts);

		// Extract VIF values and determine severities
		for (const auto &vif_result : vif_results) {
			std::string var_name = "x" + std::to_string(vif_result.variable_index + 1);

			if (!vif_result.is_defined) {
				// VIF is undefined (constant feature or perfect collinearity)
				if (vif_result.status == libanostat::diagnostics::VIFCalculator::VIFResult::Status::CONSTANT_FEATURE) {
					bind_data->variable_names.push_back(var_name);
					bind_data->vif_values.push_back(std::numeric_limits<double>::quiet_NaN());
					bind_data->severities.push_back("undefined");
					ANOFOX_DEBUG("VIF[" << vif_result.variable_index << "]: constant column (undefined)");
				} else {
					// Perfect collinearity
					bind_data->variable_names.push_back(var_name);
					bind_data->vif_values.push_back(std::numeric_limits<double>::infinity());
					bind_data->severities.push_back("perfect");
					ANOFOX_DEBUG("VIF[" << vif_result.variable_index << "]: perfect collinearity");
				}
			} else {
				// VIF is defined - determine severity
				double vif = vif_result.vif;
				string severity;
				if (vif < 5.0) {
					severity = "low";
				} else if (vif < 10.0) {
					severity = "moderate";
				} else {
					severity = "high";
				}

				bind_data->variable_names.push_back(var_name);
				bind_data->vif_values.push_back(vif);
				bind_data->severities.push_back(severity);

				ANOFOX_DEBUG("VIF[" << vif_result.variable_index << "]: " << vif << " (" << severity << ")");
			}
		}
	} catch (const std::exception &e) {
		throw InvalidInputException("VIF calculation failed: %s", e.what());
	}

	// Define return types
	names = {"variable_id", "variable_name", "vif", "severity"};
	return_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::VARCHAR};

	return std::move(bind_data);
}

/**
 * Table function implementation
 */
static void VifTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<VifBindData>();

	idx_t n_results = bind_data.variable_names.size();
	idx_t rows_to_output = std::min((idx_t)STANDARD_VECTOR_SIZE, n_results - bind_data.current_row);

	if (rows_to_output == 0) {
		return;
	}

	output.SetCardinality(rows_to_output);

	auto var_id_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto var_name_data = FlatVector::GetData<string_t>(output.data[1]);
	auto vif_data = FlatVector::GetData<double>(output.data[2]);
	auto severity_data = FlatVector::GetData<string_t>(output.data[3]);

	for (idx_t i = 0; i < rows_to_output; i++) {
		idx_t idx = bind_data.current_row + i;

		var_id_data[i] = idx + 1; // 1-indexed
		var_name_data[i] = StringVector::AddString(output.data[1], bind_data.variable_names[idx]);
		vif_data[i] = bind_data.vif_values[idx];
		severity_data[i] = StringVector::AddString(output.data[3], bind_data.severities[idx]);
	}

	bind_data.current_row += rows_to_output;
}

void VifFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering VIF function");

	TableFunction vif_func("anofox_statistics_vif",
	                       {LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))}, // X matrix
	                       VifTableFunc, VifBind);

	// Set named parameters
	vif_func.named_parameters["x"] = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));

	loader.RegisterFunction(vif_func);

	ANOFOX_DEBUG("VIF function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
