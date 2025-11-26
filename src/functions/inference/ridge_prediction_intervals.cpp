#include "ridge_prediction_intervals.hpp"
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

namespace duckdb {
namespace anofox_statistics {

struct RidgePredictIntervalBindData : public FunctionData {
	vector<double> predictions;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<double> std_errors;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<RidgePredictIntervalBindData>();
		result->predictions = predictions;
		result->ci_lowers = ci_lowers;
		result->ci_uppers = ci_uppers;
		result->std_errors = std_errors;
		result->current_row = current_row;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

static unique_ptr<FunctionData> RidgePredictIntervalBind(ClientContext &context, TableFunctionBindInput &input,
                                                         vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<RidgePredictIntervalBindData>();

	// Get parameters
	auto &y_train_value = input.inputs[0];
	auto &x_train_value = input.inputs[1];
	auto &x_new_value = input.inputs[2];
	auto &options_value = input.inputs[3];

	// Parse options
	RegressionOptions options = RegressionOptions::ParseFromMap(options_value);

	// Ridge requires lambda
	if (options.lambda <= 0.0) {
		throw InvalidInputException("Ridge prediction requires lambda > 0 (got lambda=%f)", options.lambda);
	}
	double confidence_level = options.confidence_level;
	bool add_intercept = options.intercept;
	bool is_prediction_interval = true;

	if (options_value.type().id() == LogicalTypeId::MAP) {
		// Check interval_type in options
		auto &map_value = options_value;
		auto &map_keys = StructValue::GetChildren(map_value);
		for (size_t i = 0; i < map_keys.size(); i += 2) {
			if (map_keys[i].ToString() == "interval_type") {
				string interval_type = map_keys[i + 1].ToString();
				is_prediction_interval = (interval_type == "prediction");
				break;
			}
		}
	}

	// Extract y_train
	vector<double> y_train;
	auto &y_list = ListValue::GetChildren(y_train_value);
	for (auto &val : y_list) {
		y_train.push_back(val.GetValue<double>());
	}

	idx_t n_train = y_train.size();

	// Extract X_train matrix
	vector<vector<double>> x_train_matrix;
	auto &x_train_outer = ListValue::GetChildren(x_train_value);

	for (auto &row_val : x_train_outer) {
		auto &row_list = ListValue::GetChildren(row_val);
		vector<double> row;
		for (auto &val : row_list) {
			row.push_back(val.GetValue<double>());
		}
		x_train_matrix.push_back(row);
	}

	if (x_train_matrix.empty() || x_train_matrix[0].empty()) {
		throw InvalidInputException("X_train cannot be empty");
	}

	idx_t p = x_train_matrix[0].size();

	// Extract X_new matrix
	vector<vector<double>> x_new_matrix;
	auto &x_new_outer = ListValue::GetChildren(x_new_value);

	for (auto &row_val : x_new_outer) {
		auto &row_list = ListValue::GetChildren(row_val);
		vector<double> row;
		for (auto &val : row_list) {
			row.push_back(val.GetValue<double>());
		}
		x_new_matrix.push_back(row);
	}

	idx_t n_new = x_new_matrix.size();

	// Fit Ridge model with full output
	auto result = bridge::LibanostatWrapper::FitRidge(y_train, x_train_matrix, options, true);

	Eigen::VectorXd coefficients = result.coefficients;
	double intercept = result.intercept;
	idx_t rank = result.rank;
	idx_t df = n_train - rank;

	if (df == 0) {
		throw InvalidInputException("Not enough observations for effective parameters: n=%llu, rank=%llu", n_train,
		                            rank);
	}

	double mse = result.mse;
	double alpha = 1.0 - confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, static_cast<int>(df));

	// Build X matrix for training data
	Eigen::MatrixXd X_train(n_train, p);
	for (idx_t i = 0; i < n_train; i++) {
		for (idx_t j = 0; j < p; j++) {
			X_train(i, j) = x_train_matrix[i][j];
		}
	}

	// Compute (X'X)^(-1) for leverage calculation (using non-aliased features only)
	Eigen::MatrixXd XtX_inv;
	idx_t n_valid = 0;
	for (idx_t j = 0; j < p; j++) {
		if (!std::isnan(coefficients(j)) && !result.is_aliased[j]) {
			n_valid++;
		}
	}

	if (n_valid > 0) {
		Eigen::MatrixXd X_valid(n_train, n_valid);
		idx_t valid_idx = 0;
		for (idx_t j = 0; j < p; j++) {
			if (!std::isnan(coefficients(j)) && !result.is_aliased[j]) {
				X_valid.col(valid_idx) = X_train.col(j);
				valid_idx++;
			}
		}
		Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
		XtX_inv = XtX.inverse();
	} else {
		XtX_inv = Eigen::MatrixXd::Identity(1, 1);
	}

	// Make predictions for each new observation
	for (idx_t i = 0; i < n_new; i++) {
		// Build x_new vector
		Eigen::VectorXd x_new(p);
		for (idx_t j = 0; j < p; j++) {
			x_new(j) = x_new_matrix[i][j];
		}

		// Point prediction: y_pred = intercept + beta' * x_new
		double y_pred = intercept;
		for (idx_t j = 0; j < p; j++) {
			if (!std::isnan(coefficients(j))) {
				y_pred += coefficients(j) * x_new(j);
			}
		}

		// Compute leverage for interval calculation
		double leverage = 0.0;
		if (n_valid > 0) {
			Eigen::VectorXd x_valid(n_valid);
			idx_t valid_idx = 0;
			for (idx_t j = 0; j < p; j++) {
				if (!std::isnan(coefficients(j)) && !result.is_aliased[j]) {
					x_valid(valid_idx) = x_new(j);
					valid_idx++;
				}
			}
			leverage = x_valid.transpose() * XtX_inv * x_valid;
		}

		// Standard error and interval
		double se;
		if (is_prediction_interval) {
			// Prediction interval: SE = sqrt(MSE * (1 + leverage))
			se = std::sqrt(mse * (1.0 + leverage));
		} else {
			// Confidence interval: SE = sqrt(MSE * leverage)
			se = std::sqrt(mse * leverage);
		}

		double ci_lower = y_pred - t_crit * se;
		double ci_upper = y_pred + t_crit * se;

		bind_data->predictions.push_back(y_pred);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->std_errors.push_back(se);

		ANOFOX_DEBUG("Ridge Prediction " << i << ": " << y_pred << " [" << ci_lower << ", " << ci_upper << "]");
	}

	// Define return types
	names = {"observation_id", "predicted", "ci_lower", "ci_upper", "se"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE};

	return std::move(bind_data);
}

static void RidgePredictIntervalTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<RidgePredictIntervalBindData>();

	idx_t n_results = bind_data.predictions.size();
	idx_t rows_to_output = std::min((idx_t)STANDARD_VECTOR_SIZE, n_results - bind_data.current_row);

	if (rows_to_output == 0) {
		return;
	}

	output.SetCardinality(rows_to_output);

	auto id_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto pred_data = FlatVector::GetData<double>(output.data[1]);
	auto ci_lower_data = FlatVector::GetData<double>(output.data[2]);
	auto ci_upper_data = FlatVector::GetData<double>(output.data[3]);
	auto se_data = FlatVector::GetData<double>(output.data[4]);

	for (idx_t i = 0; i < rows_to_output; i++) {
		idx_t idx = bind_data.current_row + i;

		id_data[i] = idx + 1; // 1-indexed
		pred_data[i] = bind_data.predictions[idx];
		ci_lower_data[i] = bind_data.ci_lowers[idx];
		ci_upper_data[i] = bind_data.ci_uppers[idx];
		se_data[i] = bind_data.std_errors[idx];
	}

	bind_data.current_row += rows_to_output;
}

void RidgePredictIntervalFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering Ridge predict_interval function");

	TableFunction func("anofox_statistics_predict_ridge",
	                   {LogicalType::LIST(LogicalType::DOUBLE),                     // y_train
	                    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),  // x_train
	                    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))}, // x_new
	                   RidgePredictIntervalTableFunc, RidgePredictIntervalBind);

	// Add support for optional MAP parameter via varargs
	func.varargs = LogicalType::ANY;

	loader.RegisterFunction(func);

	ANOFOX_DEBUG("Ridge predict_interval function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
