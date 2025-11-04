#include "ols_metrics.hpp"
#include "../utils/tracing.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function.hpp"

#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Compute R² (coefficient of determination)
 * R² = 1 - (SS_res / SS_tot)
 * Where:
 *   SS_res = sum((y_actual - y_predicted)²)
 *   SS_tot = sum((y_actual - mean(y_actual))²)
 */
void OlsMetricsFunction::ComputeRSquared(DataChunk &args, ExpressionState &state, Vector &result) {
	ANOFOX_DEBUG("Computing R² metric");

	if (args.ColumnCount() != 2) {
		throw InvalidInputException(
		    "anofox_statistics_ols_r_squared requires exactly 2 arguments: actual (DOUBLE), predicted (DOUBLE)");
	}

	idx_t n_rows = args.size();
	auto &actual_vec = args.data[0];
	auto &predicted_vec = args.data[1];
	auto *result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	// Collect valid pairs
	std::vector<double> actual_vals, predicted_vals;
	for (idx_t i = 0; i < n_rows; i++) {
		if (!FlatVector::IsNull(actual_vec, i) && !FlatVector::IsNull(predicted_vec, i)) {
			actual_vals.push_back(FlatVector::GetData<double>(actual_vec)[i]);
			predicted_vals.push_back(FlatVector::GetData<double>(predicted_vec)[i]);
		}
	}

	if (actual_vals.empty()) {
		result_validity.SetInvalid(0);
		return;
	}

	// Compute mean of actual values
	double mean_actual = 0.0;
	for (double val : actual_vals) {
		mean_actual += val;
	}
	mean_actual /= static_cast<double>(actual_vals.size());

	// Compute SS_res and SS_tot
	double ss_res = 0.0, ss_tot = 0.0;
	for (size_t i = 0; i < actual_vals.size(); i++) {
		double residual = actual_vals[i] - predicted_vals[i];
		double total_dev = actual_vals[i] - mean_actual;
		ss_res += residual * residual;
		ss_tot += total_dev * total_dev;
	}

	// Compute R²
	double r_squared = (ss_tot != 0.0) ? 1.0 - (ss_res / ss_tot) : 0.0;
	result_data[0] = r_squared;
	result_validity.SetValid(0);

	ANOFOX_DEBUG("R² computed: " + std::to_string(r_squared));
}

/**
 * @brief Compute Mean Squared Error
 * MSE = mean((y_actual - y_predicted)²)
 */
void OlsMetricsFunction::ComputeMSE(DataChunk &args, ExpressionState &state, Vector &result) {
	ANOFOX_DEBUG("Computing MSE metric");

	if (args.ColumnCount() != 2) {
		throw InvalidInputException(
		    "anofox_statistics_ols_mse requires exactly 2 arguments: actual (DOUBLE), predicted (DOUBLE)");
	}

	idx_t n_rows = args.size();
	auto &actual_vec = args.data[0];
	auto &predicted_vec = args.data[1];
	auto *result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	// Collect valid pairs and compute MSE
	double sum_squared_error = 0.0;
	idx_t valid_count = 0;

	for (idx_t i = 0; i < n_rows; i++) {
		if (!FlatVector::IsNull(actual_vec, i) && !FlatVector::IsNull(predicted_vec, i)) {
			double actual = FlatVector::GetData<double>(actual_vec)[i];
			double predicted = FlatVector::GetData<double>(predicted_vec)[i];
			double error = actual - predicted;
			sum_squared_error += error * error;
			valid_count++;
		}
	}

	if (valid_count == 0) {
		result_validity.SetInvalid(0);
		return;
	}

	double mse = sum_squared_error / static_cast<double>(valid_count);
	result_data[0] = mse;
	result_validity.SetValid(0);

	ANOFOX_DEBUG("MSE computed: " + std::to_string(mse));
}

/**
 * @brief Compute Root Mean Squared Error
 * RMSE = sqrt(MSE)
 */
void OlsMetricsFunction::ComputeRMSE(DataChunk &args, ExpressionState &state, Vector &result) {
	ANOFOX_DEBUG("Computing RMSE metric");

	if (args.ColumnCount() != 2) {
		throw InvalidInputException(
		    "anofox_statistics_ols_rmse requires exactly 2 arguments: actual (DOUBLE), predicted (DOUBLE)");
	}

	idx_t n_rows = args.size();
	auto &actual_vec = args.data[0];
	auto &predicted_vec = args.data[1];
	auto *result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	// Collect valid pairs and compute RMSE
	double sum_squared_error = 0.0;
	idx_t valid_count = 0;

	for (idx_t i = 0; i < n_rows; i++) {
		if (!FlatVector::IsNull(actual_vec, i) && !FlatVector::IsNull(predicted_vec, i)) {
			double actual = FlatVector::GetData<double>(actual_vec)[i];
			double predicted = FlatVector::GetData<double>(predicted_vec)[i];
			double error = actual - predicted;
			sum_squared_error += error * error;
			valid_count++;
		}
	}

	if (valid_count == 0) {
		result_validity.SetInvalid(0);
		return;
	}

	double mse = sum_squared_error / static_cast<double>(valid_count);
	double rmse = std::sqrt(mse);
	result_data[0] = rmse;
	result_validity.SetValid(0);

	ANOFOX_DEBUG("RMSE computed: " + std::to_string(rmse));
}

/**
 * @brief Compute Mean Absolute Error
 * MAE = mean(|y_actual - y_predicted|)
 */
void OlsMetricsFunction::ComputeMAE(DataChunk &args, ExpressionState &state, Vector &result) {
	ANOFOX_DEBUG("Computing MAE metric");

	if (args.ColumnCount() != 2) {
		throw InvalidInputException(
		    "anofox_statistics_ols_mae requires exactly 2 arguments: actual (DOUBLE), predicted (DOUBLE)");
	}

	idx_t n_rows = args.size();
	auto &actual_vec = args.data[0];
	auto &predicted_vec = args.data[1];
	auto *result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	// Collect valid pairs and compute MAE
	double sum_absolute_error = 0.0;
	idx_t valid_count = 0;

	for (idx_t i = 0; i < n_rows; i++) {
		if (!FlatVector::IsNull(actual_vec, i) && !FlatVector::IsNull(predicted_vec, i)) {
			double actual = FlatVector::GetData<double>(actual_vec)[i];
			double predicted = FlatVector::GetData<double>(predicted_vec)[i];
			double error = actual - predicted;
			sum_absolute_error += std::abs(error);
			valid_count++;
		}
	}

	if (valid_count == 0) {
		result_validity.SetInvalid(0);
		return;
	}

	double mae = sum_absolute_error / static_cast<double>(valid_count);
	result_data[0] = mae;
	result_validity.SetValid(0);

	ANOFOX_DEBUG("MAE computed: " + std::to_string(mae));
}

void OlsMetricsFunction::Register(ExtensionLoader &loader) {
	// R² metric
	ScalarFunction r_squared_func("anofox_statistics_ols_r_squared", {LogicalType::DOUBLE, LogicalType::DOUBLE},
	                              LogicalType::DOUBLE, [](DataChunk &args, ExpressionState &state, Vector &result) {
		                              OlsMetricsFunction::ComputeRSquared(args, state, result);
	                              });
	loader.RegisterFunction(r_squared_func);

	// MSE metric
	ScalarFunction mse_func("anofox_statistics_ols_mse", {LogicalType::DOUBLE, LogicalType::DOUBLE},
	                        LogicalType::DOUBLE, [](DataChunk &args, ExpressionState &state, Vector &result) {
		                        OlsMetricsFunction::ComputeMSE(args, state, result);
	                        });
	loader.RegisterFunction(mse_func);

	// RMSE metric
	ScalarFunction rmse_func("anofox_statistics_ols_rmse", {LogicalType::DOUBLE, LogicalType::DOUBLE},
	                         LogicalType::DOUBLE, [](DataChunk &args, ExpressionState &state, Vector &result) {
		                         OlsMetricsFunction::ComputeRMSE(args, state, result);
	                         });
	loader.RegisterFunction(rmse_func);

	// MAE metric
	ScalarFunction mae_func("anofox_statistics_ols_mae", {LogicalType::DOUBLE, LogicalType::DOUBLE},
	                        LogicalType::DOUBLE, [](DataChunk &args, ExpressionState &state, Vector &result) {
		                        OlsMetricsFunction::ComputeMAE(args, state, result);
	                        });
	loader.RegisterFunction(mae_func);
}

} // namespace anofox_statistics
} // namespace duckdb
