#include "wls_fit.hpp"
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
 * Weighted Least Squares (WLS) with MAP-based options
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_wls(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       weights := [1.0, 1.0, 2.0, 2.0],
 *       options := MAP{'intercept': true}
 *   )
 *
 * Formula: β = (X'WX)^(-1) X'Wy
 *
 * Where:
 * - W is a diagonal matrix of weights
 * - Higher weights give more importance to certain observations
 * - Useful for handling heteroscedasticity (non-constant variance)
 * - When all weights = 1, reduces to standard OLS
 */

struct WlsFitBindData : public FunctionData {
	vector<double> y_values;
	vector<vector<double>> x_values;
	vector<double> weights;
	RegressionOptions options;

	// Results
	vector<double> coefficients;
	double intercept = 0.0;
	double r_squared = 0.0;
	double adj_r_squared = 0.0;
	double mse = 0.0;
	double rmse = 0.0;
	double weighted_mse = 0.0; // MSE weighted by observation weights
	idx_t n_obs = 0;
	idx_t n_features = 0;

	// Rank-deficiency tracking
	vector<bool> is_aliased;
	idx_t rank = 0;

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<WlsFitBindData>();
		result->y_values = y_values;
		result->x_values = x_values;
		result->weights = weights;
		result->options = options;
		result->coefficients = coefficients;
		result->intercept = intercept;
		result->r_squared = r_squared;
		result->adj_r_squared = adj_r_squared;
		result->mse = mse;
		result->rmse = rmse;
		result->weighted_mse = weighted_mse;
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
 * WLS implementation with rank-deficiency handling
 * β = (X'WX)^(-1) X'Wy
 *
 * Approach: Transform to weighted OLS problem
 * - X_weighted = sqrt(W) * X
 * - y_weighted = sqrt(W) * y
 * - Then solve OLS on weighted matrices with rank-deficient solver
 */
static void ComputeWLS(WlsFitBindData &data) {
	idx_t n = data.y_values.size();
	idx_t p = data.x_values.size();

	if (n == 0 || p == 0) {
		throw InvalidInputException("Cannot fit WLS with empty data");
	}

	if (data.weights.size() != n) {
		throw InvalidInputException("Weights array must have same length as y: expected %d, got %d", n,
		                            data.weights.size());
	}

	if (n < p + 1) {
		throw InvalidInputException("Insufficient observations: need at least %d observations for %d features, got %d",
		                            p + 1, p, n);
	}

	// Validate weights (must be positive)
	for (idx_t i = 0; i < n; i++) {
		if (data.weights[i] <= 0.0) {
			throw InvalidInputException("All weights must be positive, got weight[%d] = %f", i, data.weights[i]);
		}
	}

	data.n_obs = n;
	data.n_features = p;

	ANOFOX_DEBUG("Computing WLS with " << n << " observations and " << p << " features");

	// Build design matrix X (n x p) and response vector y (n x 1)
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);
	Eigen::VectorXd w(n);

	// Fill X matrix with features
	for (idx_t j = 0; j < p; j++) {
		if (data.x_values[j].size() != n) {
			throw InvalidInputException("Feature %d has %d values, expected %d", j, data.x_values[j].size(), n);
		}
		for (idx_t i = 0; i < n; i++) {
			X(i, j) = data.x_values[j][i];
		}
	}

	// Fill y vector and weight vector
	for (idx_t i = 0; i < n; i++) {
		y(i) = data.y_values[i];
		w(i) = data.weights[i];
	}

	// Transform to weighted problem:
	// For WLS with intercept, we must center using WEIGHTED means before transformation
	// This is analogous to how OLS centers data before solving
	Eigen::VectorXd sqrt_w = w.array().sqrt();

	// Compute weighted means (needed for R² and for centering if add_intercept=true)
	double sum_weights = w.sum();
	double y_weighted_mean = (w.array() * y.array()).sum() / sum_weights;
	Eigen::VectorXd x_weighted_means = Eigen::VectorXd::Zero(p);
	for (idx_t j = 0; j < p; j++) {
		x_weighted_means(j) = (w.array() * X.col(j).array()).sum() / sum_weights;
	}

	// Work matrices (will be centered if options.intercept=true)
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;

	if (data.options.intercept) {
		// Center the data BEFORE applying sqrt(W)
		// This ensures the regression coefficients are for centered data
		for (idx_t i = 0; i < n; i++) {
			y_work(i) = y(i) - y_weighted_mean;
			for (idx_t j = 0; j < p; j++) {
				X_work(i, j) = X(i, j) - x_weighted_means(j);
			}
		}
	}

	// Now apply sqrt(W) transformation to centered (or uncentered) data
	Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_work;
	Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_work;

	// Use rank-deficient solver on weighted, centered matrices
	auto result = RankDeficientOls::Fit(y_weighted, X_weighted);

	// Store rank and aliasing info
	data.rank = result.rank;
	data.is_aliased.resize(p);
	for (idx_t i = 0; i < p; i++) {
		data.is_aliased[i] = result.is_aliased[i];
	}

	// Store coefficients (NaN for aliased features)
	data.coefficients.resize(p);
	for (idx_t i = 0; i < p; i++) {
		data.coefficients[i] = result.coefficients[i];
	}

	// Compute intercept (coefficients were computed on centered data)
	if (data.options.intercept) {
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				beta_dot_xmean += result.coefficients[j] * x_weighted_means(j);
			}
		}
		data.intercept = y_weighted_mean - beta_dot_xmean;
	} else {
		data.intercept = 0.0;
	}

	// Compute predictions (using only non-aliased features)
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
	for (idx_t j = 0; j < p; j++) {
		if (!result.is_aliased[j]) {
			y_pred += result.coefficients[j] * X.col(j);
		}
	}
	if (data.options.intercept) {
		y_pred.array() += data.intercept;
	}

	// Compute residuals
	Eigen::VectorXd residuals = y - y_pred;

	// Compute weighted sum of squares
	double ss_res_weighted = (w.array() * residuals.array().square()).sum();
	double ss_tot_weighted = (w.array() * (y.array() - y_weighted_mean).square()).sum();

	// Compute unweighted statistics (for comparison)
	double ss_res = residuals.squaredNorm();

	// Weighted R² (uses weighted SS)
	data.r_squared = (ss_tot_weighted > 0) ? 1.0 - (ss_res_weighted / ss_tot_weighted) : 0.0;

	// Adjusted R² using effective rank
	idx_t effective_params = data.rank;
	if (n > effective_params + 1) {
		data.adj_r_squared = 1.0 - ((1.0 - data.r_squared) * (static_cast<double>(n) - 1.0) / (static_cast<double>(n) - static_cast<double>(effective_params) - 1.0));
	} else {
		data.adj_r_squared = data.r_squared;
	}

	// Weighted MSE
	data.weighted_mse = ss_res_weighted / sum_weights;

	// Unweighted MSE (for comparison with OLS)
	data.mse = ss_res / static_cast<double>(n);
	data.rmse = std::sqrt(data.mse);

	ANOFOX_DEBUG("WLS complete: R² = " << data.r_squared << ", weighted MSE = " << data.weighted_mse
	                                   << ", coefficients = [" << data.coefficients[0] << (p > 1 ? ", ...]" : "]"));
}

static unique_ptr<FunctionData> WlsFitBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("WLS regression bind phase");

	auto result = make_uniq<WlsFitBindData>();

	// Expected parameters: y (DOUBLE[]), x (DOUBLE[][]), weights (DOUBLE[]), [options (MAP)]

	if (input.inputs.size() < 3) {
		throw InvalidInputException("anofox_statistics_wls requires at least 3 parameters: "
		                            "y (DOUBLE[]), x (DOUBLE[][]), weights (DOUBLE[]), [options (MAP)]");
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

	// Extract weights (third parameter - DOUBLE[])
	if (input.inputs[2].type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("Third parameter (weights) must be an array of DOUBLE");
	}
	auto weights_list = ListValue::GetChildren(input.inputs[2]);
	for (const auto &val : weights_list) {
		result->weights.push_back(val.GetValue<double>());
	}

	// Extract options (fourth parameter - MAP, optional)
	if (input.inputs.size() >= 4) {
		if (input.inputs[3].type().id() == LogicalTypeId::MAP) {
			result->options = RegressionOptions::ParseFromMap(input.inputs[3]);
			result->options.Validate();
		} else if (!input.inputs[3].IsNull()) {
			throw InvalidInputException("Fourth parameter (options) must be a MAP or NULL");
		}
	}

	ANOFOX_INFO("Fitting WLS with " << n << " observations and " << result->x_values.size() << " features");

	// Perform WLS fitting
	ComputeWLS(*result);

	ANOFOX_INFO("WLS fit completed: R² = " << result->r_squared);

	// Set return schema
	names = {"coefficients", "intercept",    "r_squared", "adj_r_squared", "mse",
	         "rmse",         "weighted_mse", "n_obs",     "n_features"};

	return_types = {
	    LogicalType::LIST(LogicalType::DOUBLE), // coefficients
	    LogicalType::DOUBLE,                    // intercept
	    LogicalType::DOUBLE,                    // r_squared
	    LogicalType::DOUBLE,                    // adj_r_squared
	    LogicalType::DOUBLE,                    // mse
	    LogicalType::DOUBLE,                    // rmse
	    LogicalType::DOUBLE,                    // weighted_mse
	    LogicalType::BIGINT,                    // n_obs
	    LogicalType::BIGINT                     // n_features
	};

	return std::move(result);
}

static void WlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {

	auto &bind_data = data.bind_data->CastNoConst<WlsFitBindData>();

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
	output.data[6].SetValue(0, Value(bind_data.weighted_mse));
	output.data[7].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_obs)));
	output.data[8].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_features)));
}

void WlsFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_wls with weighted least squares");

	// Signature: anofox_statistics_wls(y DOUBLE[], x DOUBLE[][], weights DOUBLE[], [options MAP])
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                     // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x: DOUBLE[][]
	    LogicalType::LIST(LogicalType::DOUBLE)                      // weights: DOUBLE[]
	};

	TableFunction function("anofox_statistics_wls", arguments, WlsFitExecute, WlsFitBind);

	// Optional fourth parameter: options (MAP)
	function.varargs = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::ANY);

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_wls registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
