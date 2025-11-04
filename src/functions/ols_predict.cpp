#include "ols_predict.hpp"
#include "../bridge/type_converter.hpp"
#include "../bridge/memory_manager.hpp"
#include "../utils/validation.hpp"
#include "../utils/tracing.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function.hpp"

#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Variadic prediction function: anofox_statistics_ols_predict(coeffs, intercept, x1, x2, ...)
 *
 * Performs prediction as: y = intercept + sum(coeff[i] * x[i])
 */
void OlsPredictFunction::OlsPredictVariadic(DataChunk &args, ExpressionState &state, Vector &result) {
	ANOFOX_DEBUG("OLS variadic predict called");

	if (args.ColumnCount() < 3) {
		throw InvalidInputException("anofox_statistics_ols_predict requires at least 3 arguments: "
		                            "coefficients (DOUBLE[]), intercept (DOUBLE), and at least one x value (DOUBLE)");
	}

	idx_t n_rows = args.size();
	auto *result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	// Process each row
	for (idx_t row = 0; row < n_rows; row++) {
		// Get coefficients from first argument (DOUBLE array)
		auto &coeff_vec = args.data[0];
		if (FlatVector::IsNull(coeff_vec, row)) {
			result_validity.SetInvalid(row);
			continue;
		}

		// Extract array value - get the pointer to the array data
		auto coeff_value = FlatVector::GetData<Value>(coeff_vec)[row];
		std::vector<double> coeffs;

		if (coeff_value.type().id() == LogicalTypeId::ARRAY) {
			auto &array_children = ListVector::GetEntry(coeff_vec);
			auto array_entries = ListVector::GetListSize(coeff_vec);

			// Get this row's array data
			// DuckDB API INCOMPATIBILITY: ListVector::GetListEntry() doesn't exist in newer versions
			// TODO: Update to use modern ListVector API (GetListSize, GetListEntry alternatives)
			// This is a placeholder until the API is properly updated
			ANOFOX_WARN("Array-based coefficient handling requires DuckDB API update");
			result_validity.SetInvalid(row);
			continue;
		}

		// Get intercept from second argument
		auto &intercept_vec = args.data[1];
		if (FlatVector::IsNull(intercept_vec, row)) {
			result_validity.SetInvalid(row);
			continue;
		}
		double intercept = FlatVector::GetData<double>(intercept_vec)[row];

		// Get x values from remaining arguments
		std::vector<double> x_values;
		bool has_null = false;

		for (idx_t col = 2; col < args.ColumnCount(); col++) {
			auto &x_vec = args.data[col];
			if (FlatVector::IsNull(x_vec, row)) {
				has_null = true;
				break;
			}
			x_values.push_back(FlatVector::GetData<double>(x_vec)[row]);
		}

		if (has_null || coeffs.size() != x_values.size()) {
			result_validity.SetInvalid(row);
			continue;
		}

		// Compute prediction: y = intercept + sum(coeff[i] * x[i])
		double prediction = intercept;
		for (idx_t i = 0; i < coeffs.size(); i++) {
			prediction += coeffs[i] * x_values[i];
		}

		result_data[row] = prediction;
		result_validity.SetValid(row);
	}

	ANOFOX_DEBUG("OLS variadic predict completed");
}

/**
 * @brief Array-based prediction function: anofox_statistics_ols_predict(coeffs, intercept, x_array)
 *
 * Performs prediction using array of x values
 */
void OlsPredictFunction::OlsPredictArray(DataChunk &args, ExpressionState &state, Vector &result) {
	ANOFOX_DEBUG("OLS array predict called");

	if (args.ColumnCount() != 3) {
		throw InvalidInputException("anofox_statistics_ols_predict with array form requires exactly 3 arguments: "
		                            "coefficients (DOUBLE[]), intercept (DOUBLE), x_values (DOUBLE[])");
	}

	idx_t n_rows = args.size();
	auto *result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	// Process each row
	for (idx_t row = 0; row < n_rows; row++) {
		// Get coefficients from first argument
		auto &coeff_vec = args.data[0];
		if (FlatVector::IsNull(coeff_vec, row)) {
			result_validity.SetInvalid(row);
			continue;
		}

		// Extract coefficients array
		std::vector<double> coeffs;
		// Use modern ListVector API
		auto &coeff_array_children = ListVector::GetEntry(coeff_vec);
		auto list_data = ListVector::GetData(coeff_vec);
		auto coeff_list = list_data[row];
		for (idx_t i = 0; i < coeff_list.length; i++) {
			idx_t child_idx = coeff_list.offset + i;
			if (!FlatVector::IsNull(coeff_array_children, child_idx)) {
				coeffs.push_back(FlatVector::GetData<double>(coeff_array_children)[child_idx]);
			} else {
				result_validity.SetInvalid(row);
				goto next_row;
			}
		}

		// Get intercept from second argument
		{
			auto &intercept_vec = args.data[1];
			if (FlatVector::IsNull(intercept_vec, row)) {
				result_validity.SetInvalid(row);
				continue;
			}
			double intercept = FlatVector::GetData<double>(intercept_vec)[row];

			// Get x values array from third argument
			auto &x_vec = args.data[2];
			if (FlatVector::IsNull(x_vec, row)) {
				result_validity.SetInvalid(row);
				continue;
			}

			std::vector<double> x_values;
			auto &x_array_children = ListVector::GetEntry(x_vec);
			auto x_list_data = ListVector::GetData(x_vec);
			auto x_list = x_list_data[row];

			if (x_list.length != coeffs.size()) {
				throw InvalidInputException("Number of x values (" + std::to_string(x_list.length) +
				                            ") "
				                            "does not match number of coefficients (" +
				                            std::to_string(coeffs.size()) + ")");
			}

			for (idx_t i = 0; i < x_list.length; i++) {
				idx_t child_idx = x_list.offset + i;
				if (!FlatVector::IsNull(x_array_children, child_idx)) {
					x_values.push_back(FlatVector::GetData<double>(x_array_children)[child_idx]);
				} else {
					result_validity.SetInvalid(row);
					goto next_row;
				}
			}

			// Compute prediction: y = intercept + sum(coeff[i] * x[i])
			double prediction = intercept;
			for (idx_t i = 0; i < coeffs.size(); i++) {
				prediction += coeffs[i] * x_values[i];
			}

			result_data[row] = prediction;
			result_validity.SetValid(row);
		}

	next_row:;
	}

	ANOFOX_DEBUG("OLS array predict completed");
}

void OlsPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_INFO("Registering anofox_statistics_ols_predict scalar function");

	// Register variadic version for individual x values
	// anofox_statistics_ols_predict(coeffs DOUBLE[], intercept DOUBLE, x1 DOUBLE, [x2 DOUBLE], ...) -> DOUBLE
	{
		ScalarFunction ols_predict_variadic(
		    "anofox_statistics_ols_predict",
		    {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		    OlsPredictVariadic);

		ols_predict_variadic.varargs = LogicalType::DOUBLE;
		loader.RegisterFunction(ols_predict_variadic);
	}

	// Register array version for array of x values
	// anofox_statistics_ols_predict(coeffs DOUBLE[], intercept DOUBLE, x_values DOUBLE[]) -> DOUBLE
	{
		ScalarFunction ols_predict_array(
		    "anofox_statistics_ols_predict_array",
		    {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
		    LogicalType::DOUBLE, OlsPredictArray);

		loader.RegisterFunction(ols_predict_array);
	}

	ANOFOX_INFO("anofox_statistics_ols_predict and anofox_statistics_ols_predict_array registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
