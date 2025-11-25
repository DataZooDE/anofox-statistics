#include "ols_inference.hpp"
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

/**
 * Bind data for OLS inference table function
 */
struct OlsInferenceBindData : public FunctionData {
	vector<string> variable_names;
	vector<double> estimates;
	vector<double> std_errors;
	vector<double> t_statistics;
	vector<double> p_values;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<bool> significant;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<OlsInferenceBindData>();
		result->variable_names = variable_names;
		result->estimates = estimates;
		result->std_errors = std_errors;
		result->t_statistics = t_statistics;
		result->p_values = p_values;
		result->ci_lowers = ci_lowers;
		result->ci_uppers = ci_uppers;
		result->significant = significant;
		result->current_row = current_row;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Bind function - Extract arrays and compute inference
 */
static unique_ptr<FunctionData> OlsInferenceBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<OlsInferenceBindData>();

	// Get parameters
	auto &y_value = input.inputs[0];
	auto &x_value = input.inputs[1];

	double confidence_level = 0.95;
	bool add_intercept = true;

	if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
		confidence_level = input.inputs[2].GetValue<double>();
	}
	if (input.inputs.size() > 3 && !input.inputs[3].IsNull()) {
		add_intercept = input.inputs[3].GetValue<bool>();
	}

	// Extract y array
	vector<double> y_values;
	auto &y_list = ListValue::GetChildren(y_value);
	for (auto &val : y_list) {
		y_values.push_back(val.GetValue<double>());
	}

	idx_t n = y_values.size();

	// Extract X matrix (list of lists)
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

	idx_t p = x_matrix[0].size(); // Number of features

	if (x_matrix.size() != n) {
		throw InvalidInputException("Length mismatch: y has %llu observations, X has %llu", n, x_matrix.size());
	}

	// Build Eigen matrices WITHOUT intercept column (will center data instead)
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	for (idx_t i = 0; i < n; i++) {
		y(i) = y_values[i];
		for (idx_t j = 0; j < p; j++) {
			X(i, j) = x_matrix[i][j];
		}
	}

	// Center data if fitting with intercept (standard approach for OLS with intercept)
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

	idx_t n_params = p + (add_intercept ? 1 : 0);

	if (n <= n_params) {
		throw InvalidInputException(
		    "Insufficient observations: need at least %llu observations for %llu parameters, got %llu", n_params + 1,
		    n_params, n);
	}

	// Use rank-deficient OLS solver with standard errors on CENTERED data
	// Convert Eigen to DuckDB vectors
	vector<double> y_vec(n);
	vector<vector<double>> x_vec(n, vector<double>(n_params));
	for (size_t i = 0; i < n; i++) {
		y_vec[i] = y_work(i);
		for (size_t j = 0; j < n_params; j++) {
			x_vec[i][j] = X_work(i, j);
		}
	}

	// Use libanostat OLSSolver on centered data
	RegressionOptions centered_opts;
	centered_opts.intercept = false; // Data already centered
	auto ols_result = bridge::LibanostatWrapper::FitOLS(y_vec, x_vec, centered_opts, true);

	// Compute intercept first (needed for MSE calculation)
	double intercept = 0.0;
	if (add_intercept) {
		intercept = y_mean;
		for (idx_t j = 0; j < p; j++) {
			if (!std::isnan(ols_result.coefficients(j))) {
				intercept -= ols_result.coefficients(j) * x_means(j);
			}
		}
	}

	// Recompute MSE on ORIGINAL scale (not centered scale)
	// Predictions: y_pred = intercept + X * beta
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
	for (idx_t j = 0; j < p; j++) {
		if (!std::isnan(ols_result.coefficients(j))) {
			y_pred += ols_result.coefficients(j) * X.col(j);
		}
	}
	if (add_intercept) {
		y_pred.array() += intercept;
	}

	Eigen::VectorXd residuals = y - y_pred;
	double ss_res = residuals.squaredNorm();

	// Degrees of freedom: After ba2334b fix, rank now includes intercept if present
	idx_t n_params_fitted = ols_result.rank;
	idx_t df = n - n_params_fitted;

	if (df == 0) {
		throw InvalidInputException("Not enough observations for effective parameters: n=%llu, params=%llu", n,
		                            n_params_fitted);
	}

	double mse = ss_res / df;

	// Compute critical value for confidence intervals
	double alpha = 1.0 - confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, df);

	// Recompute standard errors for slopes using correct MSE
	// The slope SEs from ols_result are based on centered MSE, we need original-scale MSE
	// SE(β_j) = sqrt(MSE * (X'X)^-1_jj) where X is centered data
	Eigen::VectorXd slope_std_errors = Eigen::VectorXd::Constant(p, std::numeric_limits<double>::quiet_NaN());

	try {
		// Build (X'X)^-1 for non-aliased features
		idx_t n_valid = 0;
		for (idx_t j = 0; j < p; j++) {
			if (!std::isnan(ols_result.coefficients(j))) {
				n_valid++;
			}
		}

		if (n_valid > 0) {
			Eigen::MatrixXd X_valid(n, n_valid);
			idx_t valid_idx = 0;
			for (idx_t j = 0; j < p; j++) {
				if (!std::isnan(ols_result.coefficients(j))) {
					X_valid.col(valid_idx) = X_work.col(j);
					valid_idx++;
				}
			}

			// Compute (X'X)^-1 for centered data
			Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
			Eigen::MatrixXd XtX_inv = XtX.inverse();

			// Recompute standard errors using correct MSE
			valid_idx = 0;
			for (idx_t j = 0; j < p; j++) {
				if (!std::isnan(ols_result.coefficients(j))) {
					slope_std_errors[j] = std::sqrt(mse * XtX_inv(valid_idx, valid_idx));
					valid_idx++;
				}
			}
		}
	} catch (...) {
		// If computation fails, leave as NaN
	}

	// If add_intercept, compute intercept standard error
	if (add_intercept) {
		// Intercept standard error: SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^-1 * x_mean))
		// Need to compute (X'X)^-1 for non-aliased features
		double intercept_se = std::numeric_limits<double>::quiet_NaN();

		try {
			// Build (X'X) for non-aliased features only
			idx_t n_valid = 0;
			for (idx_t j = 0; j < p; j++) {
				if (!std::isnan(ols_result.coefficients(j))) {
					n_valid++;
				}
			}

			if (n_valid > 0) {
				Eigen::MatrixXd X_valid(n, n_valid);
				Eigen::VectorXd x_means_valid(n_valid);
				idx_t valid_idx = 0;
				for (idx_t j = 0; j < p; j++) {
					if (!std::isnan(ols_result.coefficients(j))) {
						X_valid.col(valid_idx) = X_work.col(j);
						x_means_valid(valid_idx) = x_means(j);
						valid_idx++;
					}
				}

				// Compute (X'X)^-1 for centered data
				Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
				Eigen::MatrixXd XtX_inv = XtX.inverse();

				// SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^-1 * x_mean))
				double variance_component = x_means_valid.transpose() * XtX_inv * x_means_valid;
				intercept_se = std::sqrt(mse * (1.0 / n + variance_component));
			} else {
				// No valid features -> just use MSE/n
				intercept_se = std::sqrt(mse / n);
			}
		} catch (...) {
			// If computation fails, set to NaN
			intercept_se = std::numeric_limits<double>::quiet_NaN();
		}

		// Compute t-statistic and p-value for intercept
		double t_stat = intercept / intercept_se;
		double p_value = student_t_pvalue(t_stat, df);
		double ci_lower = intercept - t_crit * intercept_se;
		double ci_upper = intercept + t_crit * intercept_se;
		bool is_sig = p_value < 0.05;

		bind_data->variable_names.push_back("intercept");
		bind_data->estimates.push_back(intercept);
		bind_data->std_errors.push_back(intercept_se);
		bind_data->t_statistics.push_back(t_stat);
		bind_data->p_values.push_back(p_value);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->significant.push_back(is_sig);

		ANOFOX_DEBUG("Inference: intercept = " << intercept << " (SE=" << intercept_se << ", t=" << t_stat
		                                       << ", p=" << p_value << ")");
	}

	// Store results for each slope coefficient (from centered fit)
	for (idx_t j = 0; j < p; j++) {
		// Variable name
		string var_name = "x" + std::to_string(j + 1);

		// Check if coefficient is aliased (NaN)
		double estimate = ols_result.coefficients(j);
		bool is_aliased = ols_result.is_aliased[j];

		double std_error, t_stat, p_value, ci_lower, ci_upper;
		bool is_sig;

		if (is_aliased || std::isnan(estimate)) {
			// Aliased coefficient -> all statistics are NaN (will be NULL in output)
			estimate = std::numeric_limits<double>::quiet_NaN();
			std_error = std::numeric_limits<double>::quiet_NaN();
			t_stat = std::numeric_limits<double>::quiet_NaN();
			p_value = std::numeric_limits<double>::quiet_NaN();
			ci_lower = std::numeric_limits<double>::quiet_NaN();
			ci_upper = std::numeric_limits<double>::quiet_NaN();
			is_sig = false;

			ANOFOX_DEBUG("Inference: " << var_name << " = ALIASED (NULL)");
		} else {
			// Valid coefficient - compute statistics using recomputed SE
			std_error = slope_std_errors[j];
			t_stat = estimate / std_error;
			p_value = student_t_pvalue(t_stat, df);
			ci_lower = estimate - t_crit * std_error;
			ci_upper = estimate + t_crit * std_error;
			is_sig = p_value < 0.05;

			ANOFOX_DEBUG("Inference: " << var_name << " = " << estimate << " (SE=" << std_error << ", t=" << t_stat
			                           << ", p=" << p_value << ")");
		}

		bind_data->variable_names.push_back(var_name);
		bind_data->estimates.push_back(estimate);
		bind_data->std_errors.push_back(std_error);
		bind_data->t_statistics.push_back(t_stat);
		bind_data->p_values.push_back(p_value);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->significant.push_back(is_sig);
	}

	// Define return types
	names = {"variable", "estimate", "std_error", "t_statistic", "p_value", "ci_lower", "ci_upper", "significant"};
	return_types = {LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN};

	return std::move(bind_data);
}

/**
 * Table function implementation
 */
static void OlsInferenceTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<OlsInferenceBindData>();

	idx_t n_results = bind_data.variable_names.size();
	idx_t rows_to_output = std::min((idx_t)STANDARD_VECTOR_SIZE, n_results - bind_data.current_row);

	if (rows_to_output == 0) {
		return; // Done
	}

	output.SetCardinality(rows_to_output);

	auto var_data = FlatVector::GetData<string_t>(output.data[0]);
	auto est_data = FlatVector::GetData<double>(output.data[1]);
	auto se_data = FlatVector::GetData<double>(output.data[2]);
	auto t_data = FlatVector::GetData<double>(output.data[3]);
	auto p_data = FlatVector::GetData<double>(output.data[4]);
	auto ci_lower_data = FlatVector::GetData<double>(output.data[5]);
	auto ci_upper_data = FlatVector::GetData<double>(output.data[6]);
	auto sig_data = FlatVector::GetData<bool>(output.data[7]);

	auto &est_validity = FlatVector::Validity(output.data[1]);
	auto &se_validity = FlatVector::Validity(output.data[2]);
	auto &t_validity = FlatVector::Validity(output.data[3]);
	auto &p_validity = FlatVector::Validity(output.data[4]);
	auto &ci_lower_validity = FlatVector::Validity(output.data[5]);
	auto &ci_upper_validity = FlatVector::Validity(output.data[6]);

	for (idx_t i = 0; i < rows_to_output; i++) {
		idx_t idx = bind_data.current_row + i;

		var_data[i] = StringVector::AddString(output.data[0], bind_data.variable_names[idx]);

		// Check if values are NaN (aliased) and set validity accordingly
		double est = bind_data.estimates[idx];
		if (std::isnan(est)) {
			est_validity.SetInvalid(i);
			se_validity.SetInvalid(i);
			t_validity.SetInvalid(i);
			p_validity.SetInvalid(i);
			ci_lower_validity.SetInvalid(i);
			ci_upper_validity.SetInvalid(i);
			est_data[i] = 0.0; // Placeholder
			se_data[i] = 0.0;
			t_data[i] = 0.0;
			p_data[i] = 0.0;
			ci_lower_data[i] = 0.0;
			ci_upper_data[i] = 0.0;
		} else {
			est_data[i] = est;
			se_data[i] = bind_data.std_errors[idx];
			t_data[i] = bind_data.t_statistics[idx];
			p_data[i] = bind_data.p_values[idx];
			ci_lower_data[i] = bind_data.ci_lowers[idx];
			ci_upper_data[i] = bind_data.ci_uppers[idx];
		}

		sig_data[i] = bind_data.significant[idx];
	}

	bind_data.current_row += rows_to_output;
}

void OlsInferenceFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering OLS inference function");

	TableFunction ols_inference_func("ols_inference",
	                                 {LogicalType::LIST(LogicalType::DOUBLE),                    // y
	                                  LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // X (matrix)
	                                  LogicalType::DOUBLE,                                       // confidence_level
	                                  LogicalType::BOOLEAN},                                     // add_intercept
	                                 OlsInferenceTableFunc, OlsInferenceBind);

	// Set named parameters with defaults
	ols_inference_func.named_parameters["y"] = LogicalType::LIST(LogicalType::DOUBLE);
	ols_inference_func.named_parameters["x"] = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
	ols_inference_func.named_parameters["confidence_level"] = LogicalType::DOUBLE;
	ols_inference_func.named_parameters["add_intercept"] = LogicalType::BOOLEAN;

	loader.RegisterFunction(ols_inference_func);

	ANOFOX_DEBUG("OLS inference function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
