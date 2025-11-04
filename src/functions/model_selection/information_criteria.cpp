// Define _USE_MATH_DEFINES before including cmath to make M_PI available on Windows/MSVC
#define _USE_MATH_DEFINES

#include "information_criteria.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * Bind data for information criteria
 */
struct InformationCriteriaBindData : public FunctionData {
	idx_t n_obs;
	idx_t n_params;
	double rss;
	double r_squared;
	double adj_r_squared;
	double aic;
	double bic;
	double aicc;
	double log_likelihood;

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<InformationCriteriaBindData>();
		result->n_obs = n_obs;
		result->n_params = n_params;
		result->rss = rss;
		result->r_squared = r_squared;
		result->adj_r_squared = adj_r_squared;
		result->aic = aic;
		result->bic = bic;
		result->aicc = aicc;
		result->log_likelihood = log_likelihood;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Bind function - Fit model and compute information criteria
 */
static unique_ptr<FunctionData> InformationCriteriaBind(ClientContext &context, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<InformationCriteriaBindData>();

	// Get parameters
	auto &y_value = input.inputs[0];
	auto &x_value = input.inputs[1];

	bool add_intercept = true;
	if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
		add_intercept = input.inputs[2].GetValue<bool>();
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
		y(static_cast<Eigen::Index>(i)) = y_values[i];
		for (idx_t j = 0; j < p; j++) {
			X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = x_matrix[i][j];
		}
	}

	// Center data if fitting with intercept (standard approach for numerical stability)
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;
	double y_mean = 0.0;
	Eigen::VectorXd x_means = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));

	if (add_intercept) {
		// Compute means
		y_mean = y.mean();
		x_means = X.colwise().mean();

		// Center the data
		y_work = y.array() - y_mean;
		for (idx_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			X_work.col(j_idx) = X.col(j_idx).array() - x_means(j_idx);
		}
	}

	// Fit OLS on centered data (if intercept) or original data (if no intercept)
	Eigen::MatrixXd XtX = X_work.transpose() * X_work;
	Eigen::VectorXd Xty = X_work.transpose() * y_work;
	Eigen::VectorXd beta = XtX.ldlt().solve(Xty);

	// Compute intercept from centered coefficients
	double intercept = 0.0;
	if (add_intercept) {
		intercept = y_mean;
		for (idx_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			intercept -= beta(j_idx) * x_means(j_idx);
		}
	}

	// Compute residuals on ORIGINAL scale (using original X and y)
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
	for (idx_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		y_pred += beta(j_idx) * X.col(j_idx);
	}
	if (add_intercept) {
		y_pred.array() += intercept;
	}
	Eigen::VectorXd residuals = y - y_pred;

	// Number of parameters (slopes + intercept if present)
	idx_t k = p + (add_intercept ? 1 : 0);

	// Sum of squares
	double rss = residuals.squaredNorm();
	double mean_y = y.mean();
	double tss = (y.array() - mean_y).square().sum();

	// R² and Adjusted R²
	double r2 = 1.0 - rss / tss;
	double n_dbl = static_cast<double>(n);
	double k_dbl = static_cast<double>(k);
	double adj_r2 = 1.0 - (1.0 - r2) * (n_dbl - 1.0) / (n_dbl - k_dbl - 1.0);

	// Log-likelihood (assuming normal errors)
	// LL = -n/2 * ln(2π) - n/2 * ln(σ²) - RSS/(2σ²)
	// where σ² = RSS/n (MLE estimate)
	double sigma2 = rss / n_dbl;
	double log_lik = -0.5 * n_dbl * std::log(2.0 * M_PI) - 0.5 * n_dbl * std::log(sigma2) - 0.5 * n_dbl;

	// Information Criteria
	// NOTE: Uses simplified AIC/BIC formulas that differ from R by a constant offset.
	// Both formulations are mathematically valid and give IDENTICAL model rankings.
	// See TECHNICAL_NOTE_AIC_BIC_ALIGNMENT.md for alignment instructions if needed.
	//
	// Current (simplified): AIC = n*ln(RSS/n) + 2k
	// R's formula:          AIC = -2*log(L) + 2k  where log(L) uses REML variance
	// Difference:           Constant ≈ n*(ln(2π) + 1 + ln(n/(n-k)))
	//
	// For model comparison (primary use case), the constant cancels out.

	// AIC = 2k - 2*ln(L) = n*ln(RSS/n) + 2k
	double aic = n_dbl * std::log(rss / n_dbl) + 2.0 * k_dbl;

	// BIC = k*ln(n) - 2*ln(L) = n*ln(RSS/n) + k*ln(n)
	double bic = n_dbl * std::log(rss / n_dbl) + k_dbl * std::log(n_dbl);

	// AICc = AIC + 2k(k+1)/(n-k-1) (small sample correction)
	double aicc = aic + (2.0 * k_dbl * (k_dbl + 1.0)) / (n_dbl - k_dbl - 1.0);

	// Store results
	bind_data->n_obs = n;
	bind_data->n_params = k;
	bind_data->rss = rss;
	bind_data->r_squared = r2;
	bind_data->adj_r_squared = adj_r2;
	bind_data->aic = aic;
	bind_data->bic = bic;
	bind_data->aicc = aicc;
	bind_data->log_likelihood = log_lik;

	ANOFOX_DEBUG("Information Criteria: n=" << n << ", k=" << k << ", R²=" << r2 << ", AIC=" << aic << ", BIC=" << bic);

	// Define return types
	names = {"n_obs", "n_params", "rss", "r_squared", "adj_r_squared", "aic", "bic", "aicc", "log_likelihood"};
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE};

	return std::move(bind_data);
}

/**
 * Table function implementation
 */
static void InformationCriteriaTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<InformationCriteriaBindData>();

	if (bind_data.result_returned) {
		return;
	}

	output.SetCardinality(1);

	auto n_obs_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto n_params_data = FlatVector::GetData<int64_t>(output.data[1]);
	auto rss_data = FlatVector::GetData<double>(output.data[2]);
	auto r2_data = FlatVector::GetData<double>(output.data[3]);
	auto adj_r2_data = FlatVector::GetData<double>(output.data[4]);
	auto aic_data = FlatVector::GetData<double>(output.data[5]);
	auto bic_data = FlatVector::GetData<double>(output.data[6]);
	auto aicc_data = FlatVector::GetData<double>(output.data[7]);
	auto ll_data = FlatVector::GetData<double>(output.data[8]);

	n_obs_data[0] = static_cast<int64_t>(bind_data.n_obs);
	n_params_data[0] = static_cast<int64_t>(bind_data.n_params);
	rss_data[0] = bind_data.rss;
	r2_data[0] = bind_data.r_squared;
	adj_r2_data[0] = bind_data.adj_r_squared;
	aic_data[0] = bind_data.aic;
	bic_data[0] = bind_data.bic;
	aicc_data[0] = bind_data.aicc;
	ll_data[0] = bind_data.log_likelihood;

	bind_data.result_returned = true;
}

void InformationCriteriaFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering information criteria function");

	TableFunction info_criteria_func("information_criteria",
	                                 {LogicalType::LIST(LogicalType::DOUBLE),                    // y
	                                  LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // X
	                                  LogicalType::BOOLEAN},                                     // add_intercept
	                                 InformationCriteriaTableFunc, InformationCriteriaBind);

	// Set named parameters
	info_criteria_func.named_parameters["y"] = LogicalType::LIST(LogicalType::DOUBLE);
	info_criteria_func.named_parameters["x"] = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
	info_criteria_func.named_parameters["add_intercept"] = LogicalType::BOOLEAN;

	loader.RegisterFunction(info_criteria_func);

	ANOFOX_DEBUG("Information criteria function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
