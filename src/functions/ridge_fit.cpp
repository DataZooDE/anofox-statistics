#include "ridge_fit.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * Ridge Regression with L2 Regularization and MAP-based options
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_ridge(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       options := MAP{'intercept': true, 'lambda': 1.0}
 *   )
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
	RegressionOptions options;

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
		result->options = options;
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

	if (data.options.lambda < 0.0) {
		throw InvalidInputException("Lambda must be non-negative, got %f", data.options.lambda);
	}

	data.n_obs = n;
	data.n_features = p;

	ANOFOX_DEBUG("Computing Ridge regression with " << n << " observations, " << p << " features, λ=" << data.options.lambda);

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
	if (data.options.lambda == 0.0) {
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
		if (data.options.intercept) {
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

	if (data.options.intercept) {
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
	Eigen::MatrixXd XtX_regularized = XtX + data.options.lambda * identity;

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
	if (data.options.intercept) {
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
	if (data.options.intercept) {
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
		data.adj_r_squared = 1.0 - ((1.0 - data.r_squared) * (static_cast<double>(n) - 1.0) / (static_cast<double>(n) - static_cast<double>(effective_params) - 1.0));
	} else {
		data.adj_r_squared = data.r_squared;
	}

	data.mse = ss_res / static_cast<double>(n);
	data.rmse = std::sqrt(data.mse);

	ANOFOX_DEBUG("Ridge complete: R² = " << data.r_squared << ", λ=" << data.options.lambda << ", rank = " << data.rank << "/"
	                                     << p);
}

static unique_ptr<FunctionData> RidgeFitBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("Ridge regression bind phase");

	auto result = make_uniq<RidgeFitBindData>();

	// Expected parameters: y (DOUBLE[]), x (DOUBLE[][]), [options (MAP)]

	if (input.inputs.size() < 2) {
		throw InvalidInputException("anofox_statistics_ridge requires at least 2 parameters: "
		                            "y (DOUBLE[]), x (DOUBLE[][]), [options (MAP)]");
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

	// Extract x values (second parameter - 2D array)
	if (input.inputs[1].type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("Second parameter (x) must be a 2D array (DOUBLE[][])");
	}

	auto x_outer = ListValue::GetChildren(input.inputs[1]);
	for (const auto &x_inner_val : x_outer) {
		if (x_inner_val.type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("Second parameter (x) must be a 2D array where each element is DOUBLE[]");
		}

		auto x_feature_list = ListValue::GetChildren(x_inner_val);
		vector<double> x_feature;
		for (const auto &val : x_feature_list) {
			x_feature.push_back(val.GetValue<double>());
		}

		// Validate dimensions
		if (x_feature.size() != n) {
			throw InvalidInputException("Array dimensions mismatch: y has %d elements, feature %d has %d elements", n,
			                            result->x_values.size() + 1, x_feature.size());
		}

		result->x_values.push_back(x_feature);
	}

	// Extract options (third parameter - MAP, optional)
	if (input.inputs.size() >= 3) {
		if (input.inputs[2].type().id() == LogicalTypeId::MAP) {
			result->options = RegressionOptions::ParseFromMap(input.inputs[2]);
			result->options.Validate();
		} else if (!input.inputs[2].IsNull()) {
			throw InvalidInputException("Third parameter (options) must be a MAP or NULL");
		}
	}

	ANOFOX_INFO("Fitting Ridge with " << n << " observations, " << result->x_values.size()
	                                  << " features, λ=" << result->options.lambda);

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

	auto &bind_data = data.bind_data->CastNoConst<RidgeFitBindData>();

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
	output.data[6].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_obs)));
	output.data[7].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_features)));
	output.data[8].SetValue(0, Value(bind_data.options.lambda));
}

void RidgeFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_ridge with L2 regularization");

	// Signature: anofox_statistics_ridge(y DOUBLE[], x DOUBLE[][], [options MAP])
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                     // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))   // x: DOUBLE[][]
	};

	TableFunction function("anofox_statistics_ridge", arguments, RidgeFitExecute, RidgeFitBind);

	// Optional third parameter: options (MAP)
	function.varargs = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::ANY);

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_ridge registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
