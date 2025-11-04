#include "ridge_fit.hpp"
#include "../utils/tracing.hpp"
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
 * Ridge Regression with L2 Regularization
 *
 * Formula: β = (X'X + λI)^(-1) X'y
 *
 * Where:
 * - λ (lambda) is the regularization parameter
 * - I is the identity matrix
 * - Higher λ means more regularization (coefficients shrink towards zero)
 * - When λ=0, reduces to standard OLS
 */

struct RidgeFitBindData : public FunctionData {
	vector<double> y_values;
	vector<vector<double>> x_values;
	bool add_intercept = true;
	double lambda = 1.0; // Regularization parameter

	// Results
	vector<double> coefficients;
	double intercept = 0.0;
	double r_squared = 0.0;
	double adj_r_squared = 0.0;
	double mse = 0.0;
	double rmse = 0.0;
	idx_t n_obs = 0;
	idx_t n_features = 0;

	// Rank-deficiency tracking
	vector<bool> is_aliased;
	idx_t rank = 0;

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<RidgeFitBindData>();
		result->y_values = y_values;
		result->x_values = x_values;
		result->add_intercept = add_intercept;
		result->lambda = lambda;
		result->coefficients = coefficients;
		result->intercept = intercept;
		result->r_squared = r_squared;
		result->adj_r_squared = adj_r_squared;
		result->mse = mse;
		result->rmse = rmse;
		result->n_obs = n_obs;
		result->n_features = n_features;
		result->is_aliased = is_aliased;
		result->rank = rank;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Ridge regression with rank-deficiency handling
 * β = (X'X + λI)^(-1) X'y
 *
 * Note: When λ=0, reduces to OLS with full rank-deficiency handling.
 * When λ>0, regularization typically makes matrix full rank, but we still
 * detect constant features for consistency.
 */
static void ComputeRidge(RidgeFitBindData &data) {
	idx_t n = data.y_values.size();
	idx_t p = data.x_values.size();

	if (n == 0 || p == 0) {
		throw InvalidInputException("Cannot fit Ridge with empty data");
	}

	if (n < p + 1) {
		throw InvalidInputException("Insufficient observations: need at least %d observations for %d features, got %d",
		                            p + 1, p, n);
	}

	if (data.lambda < 0.0) {
		throw InvalidInputException("Lambda must be non-negative, got %f", data.lambda);
	}

	data.n_obs = n;
	data.n_features = p;

	ANOFOX_DEBUG("Computing Ridge regression with " << n << " observations, " << p << " features, λ=" << data.lambda);

	// Build design matrix X (n x p) and response vector y (n x 1)
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	// Fill X matrix with features
	for (idx_t j = 0; j < p; j++) {
		if (data.x_values[j].size() != n) {
			throw InvalidInputException("Feature %d has %d values, expected %d", j, data.x_values[j].size(), n);
		}
		for (idx_t i = 0; i < n; i++) {
			X(i, j) = data.x_values[j][i];
		}
	}

	// Fill y vector
	for (idx_t i = 0; i < n; i++) {
		y(i) = data.y_values[i];
	}

	// Special case: λ=0 reduces to OLS, use rank-deficient solver
	if (data.lambda == 0.0) {
		auto result = RankDeficientOls::Fit(y, X);

		data.rank = result.rank;
		data.is_aliased.resize(p);
		for (idx_t i = 0; i < p; i++) {
			data.is_aliased[i] = result.is_aliased[i];
		}

		data.coefficients.resize(p);
		for (idx_t i = 0; i < p; i++) {
			data.coefficients[i] = result.coefficients[i];
		}

		data.r_squared = result.r_squared;
		data.adj_r_squared = result.adj_r_squared;
		data.mse = result.mse;
		data.rmse = result.rmse;

		// Compute intercept
		if (data.add_intercept) {
			double y_mean = y.mean();
			Eigen::VectorXd x_means = X.colwise().mean();
			double beta_dot_xmean = 0.0;
			for (idx_t j = 0; j < p; j++) {
				if (!result.is_aliased[j]) {
					beta_dot_xmean += result.coefficients[j] * x_means[j];
				}
			}
			data.intercept = y_mean - beta_dot_xmean;
		} else {
			data.intercept = 0.0;
		}

		ANOFOX_DEBUG("Ridge (λ=0, OLS mode): R² = " << data.r_squared << ", rank = " << data.rank << "/" << p);
		return;
	}

	// λ > 0: Use ridge regression with rank-deficiency detection
	// First, detect constant features
	auto constant_features = RankDeficientOls::DetectConstantColumns(X);

	// Center data if fitting intercept (standard Ridge regression practice)
	// Ridge should only penalize slopes, not the intercept
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;
	Eigen::VectorXd x_means = Eigen::VectorXd::Zero(p);
	double y_mean = 0.0;

	if (data.add_intercept) {
		// Compute means
		y_mean = y.mean();
		x_means = X.colwise().mean();

		// Center the data
		for (idx_t i = 0; i < n; i++) {
			y_work(i) = y(i) - y_mean;
			for (idx_t j = 0; j < p; j++) {
				X_work(i, j) = X(i, j) - x_means(j);
			}
		}
	}

	// Ridge regression on (centered) data: β = (X'X + λI)^(-1) X'y
	Eigen::MatrixXd XtX = X_work.transpose() * X_work;
	Eigen::VectorXd Xty = X_work.transpose() * y_work;

	// Add regularization: X'X + λI
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
	Eigen::MatrixXd XtX_regularized = XtX + data.lambda * identity;

	// Use ColPivHouseholderQR for rank-revealing solve
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
	data.rank = qr.rank();

	// Solve for coefficients (these are coefficients for centered data if add_intercept=true)
	Eigen::VectorXd beta = qr.solve(Xty);

	// Initialize aliasing info
	data.is_aliased.resize(p);
	for (idx_t i = 0; i < p; i++) {
		data.is_aliased[i] = constant_features[i]; // Mark constant features as aliased
	}

	// Store coefficients (set NaN for constant features)
	data.coefficients.resize(p);
	for (idx_t i = 0; i < p; i++) {
		if (constant_features[i]) {
			data.coefficients[i] = std::numeric_limits<double>::quiet_NaN();
		} else {
			data.coefficients[i] = beta(i);
		}
	}

	// Compute intercept (for centered data: intercept = y_mean - beta·x_mean)
	if (data.add_intercept) {
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!constant_features[j]) {
				beta_dot_xmean += data.coefficients[j] * x_means[j];
			}
		}
		data.intercept = y_mean - beta_dot_xmean;
	} else {
		data.intercept = 0.0;
	}

	// Compute predictions on ORIGINAL scale
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
	for (idx_t j = 0; j < p; j++) {
		if (!constant_features[j]) {
			y_pred += data.coefficients[j] * X.col(j);
		}
	}
	if (data.add_intercept) {
		y_pred.array() += data.intercept;
	}

	// Compute residuals
	Eigen::VectorXd residuals = y - y_pred;

	// Compute statistics
	double ss_res = residuals.squaredNorm();
	double ss_tot = (y.array() - y.mean()).square().sum();

	data.r_squared = (ss_tot > 0) ? 1.0 - (ss_res / ss_tot) : 0.0;

	// Adjusted R² using effective rank
	idx_t effective_params = data.rank;
	if (n > effective_params + 1) {
		data.adj_r_squared = 1.0 - ((1.0 - data.r_squared) * (n - 1.0) / (n - effective_params - 1.0));
	} else {
		data.adj_r_squared = data.r_squared;
	}

	data.mse = ss_res / n;
	data.rmse = std::sqrt(data.mse);

	ANOFOX_DEBUG("Ridge complete: R² = " << data.r_squared << ", λ=" << data.lambda << ", rank = " << data.rank << "/"
	                                     << p);
}

static unique_ptr<FunctionData> RidgeFitBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("Ridge regression bind phase");

	auto result = make_uniq<RidgeFitBindData>();

	// Expected parameters: y (DOUBLE[]), x1 (DOUBLE[]), [x2 (DOUBLE[]), ...], [lambda (DOUBLE)], [add_intercept
	// (BOOLEAN)]

	if (input.inputs.size() < 2) {
		throw InvalidInputException(
		    "anofox_statistics_ridge_fit requires at least 2 parameters: "
		    "y (DOUBLE[]), x1 (DOUBLE[]), [x2 (DOUBLE[]), ...], [lambda (DOUBLE)], [add_intercept (BOOLEAN)]");
	}

	// Extract y values (first parameter)
	if (input.inputs[0].type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("First parameter (y) must be an array of DOUBLE");
	}
	auto y_list = ListValue::GetChildren(input.inputs[0]);
	for (const auto &val : y_list) {
		result->y_values.push_back(val.GetValue<double>());
	}

	idx_t n = result->y_values.size();
	ANOFOX_DEBUG("y has " << n << " observations");

	// Determine where x features end
	idx_t last_param_idx = input.inputs.size() - 1;
	bool has_add_intercept = false;
	bool has_lambda = false;

	// Check last parameter: might be add_intercept (BOOLEAN)
	if (input.inputs[last_param_idx].type().id() == LogicalTypeId::BOOLEAN) {
		result->add_intercept = input.inputs[last_param_idx].GetValue<bool>();
		has_add_intercept = true;
		last_param_idx--;
	}

	// Check second-to-last parameter: might be lambda (DOUBLE)
	if (last_param_idx > 0 && input.inputs[last_param_idx].type().id() == LogicalTypeId::DOUBLE) {
		result->lambda = input.inputs[last_param_idx].GetValue<double>();
		has_lambda = true;
		last_param_idx--;
	}

	// Extract all x feature arrays (from index 1 to last_param_idx)
	for (idx_t param_idx = 1; param_idx <= last_param_idx; param_idx++) {
		if (input.inputs[param_idx].type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("Parameter %d (x%d) must be an array of DOUBLE", param_idx + 1, param_idx);
		}

		auto x_list = ListValue::GetChildren(input.inputs[param_idx]);
		vector<double> x_feature;
		for (const auto &val : x_list) {
			x_feature.push_back(val.GetValue<double>());
		}

		// Validate dimensions
		if (x_feature.size() != n) {
			throw InvalidInputException("Array dimensions mismatch: y has %d elements, x%d has %d elements", n,
			                            param_idx, x_feature.size());
		}

		result->x_values.push_back(x_feature);
	}

	ANOFOX_INFO("Fitting Ridge with " << n << " observations, " << result->x_values.size()
	                                  << " features, λ=" << result->lambda);

	// Perform Ridge fitting
	ComputeRidge(*result);

	ANOFOX_INFO("Ridge fit completed: R² = " << result->r_squared);

	// Set return schema
	names = {"coefficients", "intercept", "r_squared", "adj_r_squared", "mse", "rmse", "n_obs", "n_features", "lambda"};

	return_types = {
	    LogicalType::LIST(LogicalType::DOUBLE), // coefficients
	    LogicalType::DOUBLE,                    // intercept
	    LogicalType::DOUBLE,                    // r_squared
	    LogicalType::DOUBLE,                    // adj_r_squared
	    LogicalType::DOUBLE,                    // mse
	    LogicalType::DOUBLE,                    // rmse
	    LogicalType::BIGINT,                    // n_obs
	    LogicalType::BIGINT,                    // n_features
	    LogicalType::DOUBLE                     // lambda
	};

	return std::move(result);
}

static void RidgeFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {

	auto &bind_data = const_cast<RidgeFitBindData &>(dynamic_cast<const RidgeFitBindData &>(*data.bind_data));

	if (bind_data.result_returned) {
		return;
	}

	bind_data.result_returned = true;
	output.SetCardinality(1);

	// Return results - convert NaN to NULL for aliased coefficients
	vector<Value> coeffs_values;
	for (idx_t i = 0; i < bind_data.coefficients.size(); i++) {
		double coef = bind_data.coefficients[i];
		if (std::isnan(coef)) {
			// Aliased coefficient -> NULL
			coeffs_values.push_back(Value(LogicalType::DOUBLE));
		} else {
			coeffs_values.push_back(Value(coef));
		}
	}
	output.data[0].SetValue(0, Value::LIST(LogicalType::DOUBLE, coeffs_values));
	output.data[1].SetValue(0, Value(bind_data.intercept));
	output.data[2].SetValue(0, Value(bind_data.r_squared));
	output.data[3].SetValue(0, Value(bind_data.adj_r_squared));
	output.data[4].SetValue(0, Value(bind_data.mse));
	output.data[5].SetValue(0, Value(bind_data.rmse));
	output.data[6].SetValue(0, Value::BIGINT(bind_data.n_obs));
	output.data[7].SetValue(0, Value::BIGINT(bind_data.n_features));
	output.data[8].SetValue(0, Value(bind_data.lambda));
}

void RidgeFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_ridge_fit with L2 regularization");

	// Required arguments: y (DOUBLE[]), x1 (DOUBLE[])
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE), // y
	    LogicalType::LIST(LogicalType::DOUBLE)  // x1
	};

	TableFunction function("anofox_statistics_ridge_fit", arguments, RidgeFitExecute, RidgeFitBind);

	// Optional arguments: x2-xN (DOUBLE[]), lambda (DOUBLE), add_intercept (BOOLEAN)
	function.varargs = LogicalType::ANY;

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_ridge_fit registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
