#include "prediction_intervals.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"
#include "../utils/statistical_distributions.hpp"
#include "../utils/rank_deficient_ols.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * Bind data for prediction intervals
 */
struct OlsPredictIntervalBindData : public FunctionData {
	vector<double> predictions;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<double> std_errors;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<OlsPredictIntervalBindData>();
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

/**
 * Bind function - Fit model and compute predictions with intervals
 */
static unique_ptr<FunctionData> OlsPredictIntervalBind(ClientContext &context, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<OlsPredictIntervalBindData>();

	// Get parameters
	auto &y_train_value = input.inputs[0];
	auto &x_train_value = input.inputs[1];
	auto &x_new_value = input.inputs[2];

	double confidence_level = 0.95;
	string interval_type = "prediction";
	bool add_intercept = true;

	if (input.inputs.size() > 3 && !input.inputs[3].IsNull()) {
		confidence_level = input.inputs[3].GetValue<double>();
	}
	if (input.inputs.size() > 4 && !input.inputs[4].IsNull()) {
		interval_type = input.inputs[4].GetValue<string>();
	}
	if (input.inputs.size() > 5 && !input.inputs[5].IsNull()) {
		add_intercept = input.inputs[5].GetValue<bool>();
	}

	bool is_prediction_interval = (interval_type == "prediction");

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

	// Build training matrices WITHOUT intercept column (will use data centering)
	Eigen::MatrixXd X_train(n_train, p);
	Eigen::VectorXd y(n_train);

	for (idx_t i = 0; i < n_train; i++) {
		y(i) = y_train[i];
		for (idx_t j = 0; j < p; j++) {
			X_train(i, j) = x_train_matrix[i][j];
		}
	}

	// Center data if fitting with intercept (same approach as ols_inference)
	Eigen::MatrixXd X_work = X_train;
	Eigen::VectorXd y_work = y;
	double y_mean = 0.0;
	Eigen::VectorXd x_means = Eigen::VectorXd::Zero(p);

	if (add_intercept) {
		// Compute means
		y_mean = y.mean();
		x_means = X_train.colwise().mean();

		// Center the data
		y_work = y.array() - y_mean;
		for (idx_t j = 0; j < p; j++) {
			X_work.col(j) = X_train.col(j).array() - x_means(j);
		}
	}

	// Use rank-deficient OLS solver on CENTERED data
	auto ols_result = RankDeficientOls::FitWithStdErrors(y_work, X_work);

	// Compute intercept from centered coefficients
	double intercept = 0.0;
	if (add_intercept) {
		intercept = y_mean;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				intercept -= ols_result.coefficients[j] * x_means(j);
			}
		}
	}

	// Recompute MSE on ORIGINAL scale (not centered scale)
	Eigen::VectorXd y_pred_train = Eigen::VectorXd::Zero(n_train);
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			y_pred_train += ols_result.coefficients[j] * X_train.col(j);
		}
	}
	if (add_intercept) {
		y_pred_train.array() += intercept;
	}

	Eigen::VectorXd residuals_train = y - y_pred_train;
	double ss_res = residuals_train.squaredNorm();

	// Degrees of freedom: After ba2334b fix, rank now includes intercept if present
	idx_t n_params_fitted = ols_result.rank;
	idx_t df = n_train - n_params_fitted;

	if (df == 0) {
		throw InvalidInputException("Not enough observations for effective parameters: n=%llu, params=%llu", n_train,
		                            n_params_fitted);
	}

	double mse = ss_res / static_cast<double>(df);

	// Get critical value
	double alpha = 1.0 - confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, static_cast<int>(df));

	// Compute (X'X)^-1 for leverage calculation (using centered X for non-aliased features)
	Eigen::MatrixXd XtX_inv;
	idx_t n_valid = 0;
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			n_valid++;
		}
	}

	if (n_valid > 0) {
		Eigen::MatrixXd X_valid(n_train, n_valid);
		idx_t valid_idx = 0;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				X_valid.col(valid_idx) = X_work.col(j); // Use centered X
				valid_idx++;
			}
		}
		Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
		XtX_inv = XtX.inverse();
	} else {
		// No valid features - use identity
		XtX_inv = Eigen::MatrixXd::Identity(1, 1);
	}

	// Make predictions for each new observation
	for (idx_t i = 0; i < n_new; i++) {
		// Build x_new vector (original scale)
		Eigen::VectorXd x_new(p);
		for (idx_t j = 0; j < p; j++) {
			x_new(j) = x_new_matrix[i][j];
		}

		// Point prediction on original scale: y_pred = intercept + beta' * x_new
		double y_pred = intercept;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				y_pred += ols_result.coefficients[j] * x_new(j);
			}
		}

		// Compute leverage: x_new_centered' * (X'X)^-1 * x_new_centered
		// where x_new_centered = x_new - x_means
		double leverage = 0.0;
		if (n_valid > 0) {
			Eigen::VectorXd x_new_centered(n_valid);
			idx_t valid_idx = 0;
			for (idx_t j = 0; j < p; j++) {
				if (!ols_result.is_aliased[j]) {
					x_new_centered(valid_idx) = x_new(j) - x_means(j);
					valid_idx++;
				}
			}
			leverage = x_new_centered.transpose() * XtX_inv * x_new_centered;
		}

		// Standard error of prediction
		// Confidence interval: SE = sqrt(MSE * (1/n + leverage))
		// Prediction interval: SE = sqrt(MSE * (1 + 1/n + leverage))
		double variance_factor = is_prediction_interval ? 1.0 : 0.0;
		double se = std::sqrt(mse * (variance_factor + (1.0 / static_cast<double>(n_train)) + leverage));

		// Confidence/Prediction interval
		double ci_lower = y_pred - t_crit * se;
		double ci_upper = y_pred + t_crit * se;

		bind_data->predictions.push_back(y_pred);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->std_errors.push_back(se);

		ANOFOX_DEBUG("Prediction " << i << ": " << y_pred << " [" << ci_lower << ", " << ci_upper << "]"
		                           << " (SE=" << se << ")");
	}

	// Define return types
	names = {"observation_id", "predicted", "ci_lower", "ci_upper", "se"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE};

	return std::move(bind_data);
}

/**
 * Table function implementation
 */
static void OlsPredictIntervalTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<OlsPredictIntervalBindData>();

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

void OlsPredictIntervalFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering OLS predict interval function");

	TableFunction ols_predict_interval_func("ols_predict_interval",
	                                        {LogicalType::LIST(LogicalType::DOUBLE),                    // y_train
	                                         LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // X_train
	                                         LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // X_new
	                                         LogicalType::DOUBLE,   // confidence_level
	                                         LogicalType::VARCHAR,  // interval_type
	                                         LogicalType::BOOLEAN}, // add_intercept
	                                        OlsPredictIntervalTableFunc, OlsPredictIntervalBind);

	// Set named parameters
	ols_predict_interval_func.named_parameters["y_train"] = LogicalType::LIST(LogicalType::DOUBLE);
	ols_predict_interval_func.named_parameters["x_train"] = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
	ols_predict_interval_func.named_parameters["x_new"] = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
	ols_predict_interval_func.named_parameters["confidence_level"] = LogicalType::DOUBLE;
	ols_predict_interval_func.named_parameters["interval_type"] = LogicalType::VARCHAR;
	ols_predict_interval_func.named_parameters["add_intercept"] = LogicalType::BOOLEAN;

	loader.RegisterFunction(ols_predict_interval_func);

	ANOFOX_DEBUG("OLS predict interval function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
