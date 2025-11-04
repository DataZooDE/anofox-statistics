#include "residual_diagnostics.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"
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
 * Bind data for residual diagnostics
 */
struct ResidualDiagnosticsBindData : public FunctionData {
	vector<double> residuals;
	vector<double> std_residuals;
	vector<double> studentized_residuals;
	vector<double> leverages;
	vector<double> cooks_distances;
	vector<double> dffits_values;
	vector<bool> is_outlier;
	vector<bool> is_influential;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ResidualDiagnosticsBindData>();
		result->residuals = residuals;
		result->std_residuals = std_residuals;
		result->studentized_residuals = studentized_residuals;
		result->leverages = leverages;
		result->cooks_distances = cooks_distances;
		result->dffits_values = dffits_values;
		result->is_outlier = is_outlier;
		result->is_influential = is_influential;
		result->current_row = current_row;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Bind function - Fit model and compute diagnostics
 */
static unique_ptr<FunctionData> ResidualDiagnosticsBind(ClientContext &context, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<ResidualDiagnosticsBindData>();

	// Get parameters
	auto &y_value = input.inputs[0];
	auto &x_value = input.inputs[1];

	bool add_intercept = true;
	double outlier_threshold = 2.5;
	double influence_threshold = 0.5;

	if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
		add_intercept = input.inputs[2].GetValue<bool>();
	}
	if (input.inputs.size() > 3 && !input.inputs[3].IsNull()) {
		outlier_threshold = input.inputs[3].GetValue<double>();
	}
	if (input.inputs.size() > 4 && !input.inputs[4].IsNull()) {
		influence_threshold = input.inputs[4].GetValue<double>();
	}

	// Extract y array
	vector<double> y_values;
	auto &y_list = ListValue::GetChildren(y_value);
	for (auto &val : y_list) {
		y_values.push_back(val.GetValue<double>());
	}

	idx_t n = y_values.size();

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

	idx_t p = x_matrix[0].size();

	// Build matrices WITHOUT intercept column (will use data centering)
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	for (idx_t i = 0; i < n; i++) {
		y(i) = y_values[i];
		for (idx_t j = 0; j < p; j++) {
			X(i, j) = x_matrix[i][j];
		}
	}

	// Center data if fitting with intercept (same approach as ols_inference)
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;
	double y_mean = 0.0;
	Eigen::VectorXd x_means = Eigen::VectorXd::Zero(p);

	if (add_intercept) {
		// Compute means
		y_mean = y.mean();
		x_means = X.colwise().mean();

		// Center the data
		y_work = y.array() - y_mean;
		for (idx_t j = 0; j < p; j++) {
			X_work.col(j) = X.col(j).array() - x_means(j);
		}
	}

	// Fit OLS using rank-deficient solver on CENTERED data
	auto ols_result = RankDeficientOls::Fit(y_work, X_work);

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

	// Predictions and residuals on ORIGINAL scale (using only non-aliased features)
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			y_pred += ols_result.coefficients[j] * X.col(j);
		}
	}
	if (add_intercept) {
		y_pred.array() += intercept;
	}
	Eigen::VectorXd residuals_vec = y - y_pred;

	// MSE using effective rank + intercept for degrees of freedom
	double rss = residuals_vec.squaredNorm();
	idx_t n_params_fitted = ols_result.rank + (add_intercept ? 1 : 0);
	idx_t df = n - n_params_fitted;
	if (df == 0) {
		throw InvalidInputException("Not enough observations for effective parameters: n=%llu, params=%llu", n,
		                            n_params_fitted);
	}
	double mse = rss / df;
	double rmse = std::sqrt(mse);

	// Build design matrix for leverage calculation
	// For intercept model: augment with column of 1s (this is OK for leverage, not for fitting)
	Eigen::MatrixXd X_hat;
	if (add_intercept) {
		X_hat = Eigen::MatrixXd(n, p + 1);
		X_hat.col(0) = Eigen::VectorXd::Ones(n);
		for (idx_t j = 0; j < p; j++) {
			X_hat.col(j + 1) = X.col(j);
		}
	} else {
		X_hat = X;
	}

	// Hat matrix diagonal (leverage values): H = X(X'X)^+X'
	// h_i = X_i' (X'X)^+ X_i where ^+ is pseudo-inverse
	Eigen::MatrixXd XtX = X_hat.transpose() * X_hat;
	Eigen::MatrixXd XtX_pinv = XtX.completeOrthogonalDecomposition().pseudoInverse();

	for (idx_t i = 0; i < n; i++) {
		Eigen::VectorXd x_i = X_hat.row(i); // Use X_hat which includes intercept column

		// Residual
		double residual = residuals_vec(i);
		bind_data->residuals.push_back(residual);

		// Leverage (hat value) using pseudo-inverse
		double h_i = (x_i.transpose() * XtX_pinv * x_i)(0, 0);
		bind_data->leverages.push_back(h_i);

		// Standardized residual: r_i / sqrt(MSE)
		double std_resid = residual / rmse;
		bind_data->std_residuals.push_back(std_resid);

		// Studentized residual: r_i / sqrt(MSE * (1 - h_i))
		double stud_resid = residual / std::sqrt(mse * (1.0 - h_i));
		bind_data->studentized_residuals.push_back(stud_resid);

		// Cook's Distance: D_i = (r_i^2 / (k * MSE)) * (h_i / (1 - h_i)^2)
		// Simplified: D_i = (stud_resid^2 / rank) * (h_i / (1 - h_i))
		// Use effective rank for rank-deficient models
		double cooks_d = (stud_resid * stud_resid / ols_result.rank) * (h_i / ((1.0 - h_i) * (1.0 - h_i)));
		bind_data->cooks_distances.push_back(cooks_d);

		// DFFITS: measures change in fitted value when point i is removed
		// DFFITS_i = stud_resid * sqrt(h_i / (1 - h_i))
		double dffits = stud_resid * std::sqrt(h_i / (1.0 - h_i));
		bind_data->dffits_values.push_back(dffits);

		// Outlier flag: |studentized residual| > threshold
		bool outlier = std::abs(stud_resid) > outlier_threshold;
		bind_data->is_outlier.push_back(outlier);

		// Influential point flag: Cook's D > threshold
		bool influential = cooks_d > influence_threshold;
		bind_data->is_influential.push_back(influential);

		ANOFOX_DEBUG("Obs " << i << ": resid=" << residual << ", leverage=" << h_i << ", Cook's D=" << cooks_d);
	}

	// Define return types
	names = {"obs_id",         "residual", "std_residual", "studentized_residual", "leverage",
	         "cooks_distance", "dffits",   "is_outlier",   "is_influential"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::BOOLEAN, LogicalType::BOOLEAN};

	return std::move(bind_data);
}

/**
 * Table function implementation
 */
static void ResidualDiagnosticsTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<ResidualDiagnosticsBindData>();

	idx_t n_results = bind_data.residuals.size();
	idx_t rows_to_output = std::min((idx_t)STANDARD_VECTOR_SIZE, n_results - bind_data.current_row);

	if (rows_to_output == 0) {
		return;
	}

	output.SetCardinality(rows_to_output);

	auto obs_id_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto resid_data = FlatVector::GetData<double>(output.data[1]);
	auto std_resid_data = FlatVector::GetData<double>(output.data[2]);
	auto stud_resid_data = FlatVector::GetData<double>(output.data[3]);
	auto leverage_data = FlatVector::GetData<double>(output.data[4]);
	auto cooks_d_data = FlatVector::GetData<double>(output.data[5]);
	auto dffits_data = FlatVector::GetData<double>(output.data[6]);
	auto outlier_data = FlatVector::GetData<bool>(output.data[7]);
	auto influential_data = FlatVector::GetData<bool>(output.data[8]);

	for (idx_t i = 0; i < rows_to_output; i++) {
		idx_t idx = bind_data.current_row + i;

		obs_id_data[i] = idx + 1; // 1-indexed
		resid_data[i] = bind_data.residuals[idx];
		std_resid_data[i] = bind_data.std_residuals[idx];
		stud_resid_data[i] = bind_data.studentized_residuals[idx];
		leverage_data[i] = bind_data.leverages[idx];
		cooks_d_data[i] = bind_data.cooks_distances[idx];
		dffits_data[i] = bind_data.dffits_values[idx];
		outlier_data[i] = bind_data.is_outlier[idx];
		influential_data[i] = bind_data.is_influential[idx];
	}

	bind_data.current_row += rows_to_output;
}

void ResidualDiagnosticsFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering residual diagnostics function");

	TableFunction residual_diagnostics_func("residual_diagnostics",
	                                        {LogicalType::LIST(LogicalType::DOUBLE),                    // y
	                                         LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // X
	                                         LogicalType::BOOLEAN,                                      // add_intercept
	                                         LogicalType::DOUBLE,  // outlier_threshold
	                                         LogicalType::DOUBLE}, // influence_threshold
	                                        ResidualDiagnosticsTableFunc, ResidualDiagnosticsBind);

	// Set named parameters
	residual_diagnostics_func.named_parameters["y"] = LogicalType::LIST(LogicalType::DOUBLE);
	residual_diagnostics_func.named_parameters["x"] = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
	residual_diagnostics_func.named_parameters["add_intercept"] = LogicalType::BOOLEAN;
	residual_diagnostics_func.named_parameters["outlier_threshold"] = LogicalType::DOUBLE;
	residual_diagnostics_func.named_parameters["influence_threshold"] = LogicalType::DOUBLE;

	loader.RegisterFunction(residual_diagnostics_func);

	ANOFOX_DEBUG("Residual diagnostics function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
