#include "elastic_net_fit.hpp"
#include "../utils/tracing.hpp"
#include "../utils/elastic_net_solver.hpp"
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
 * @brief Elastic Net fit using array inputs with MAP-based options (v1.4.1 compatible)
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_elastic_net(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       options := MAP{'intercept': true, 'alpha': 0.5, 'lambda': 0.01}
 *   )
 *
 * Parameters:
 *   - y: Response variable (DOUBLE[])
 *   - x: Feature matrix (DOUBLE[][], each inner array is one feature)
 *   - options: Optional MAP with keys:
 *     - 'intercept' (BOOLEAN, default true)
 *     - 'alpha' (DOUBLE, default 0.5) - mixing: 0=Ridge, 1=Lasso, (0,1)=Elastic Net
 *     - 'lambda' (DOUBLE, default 0.01) - regularization strength
 */

struct ElasticNetFitBindData : public FunctionData {
	// Input data extracted from arrays
	vector<double> y_values;
	vector<vector<double>> x_values; // Each inner vector is one feature

	// Fit parameters
	RegressionOptions options;

	// Results from fitting
	vector<double> coefficients;
	double intercept = 0.0;
	double r_squared = 0.0;
	double adj_r_squared = 0.0;
	double mse = 0.0;
	double rmse = 0.0;
	idx_t n_obs = 0;
	idx_t n_features = 0;
	idx_t n_nonzero = 0;

	// Extended metadata (when full_output=true)
	vector<double> coefficient_std_errors;
	double intercept_std_error = std::numeric_limits<double>::quiet_NaN();
	idx_t df_residual = 0;
	vector<bool> is_zero; // Track which coefficients are exactly zero (feature selection)
	vector<double> x_train_means;

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ElasticNetFitBindData>();
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
		result->n_nonzero = n_nonzero;
		result->coefficient_std_errors = coefficient_std_errors;
		result->intercept_std_error = intercept_std_error;
		result->df_residual = df_residual;
		result->is_zero = is_zero;
		result->x_train_means = x_train_means;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Elastic Net implementation using coordinate descent solver
 */
static void ComputeElasticNet(ElasticNetFitBindData &data) {
	idx_t n = data.y_values.size();
	idx_t p = data.x_values.size();

	if (n == 0 || p == 0) {
		throw InvalidInputException("Cannot fit Elastic Net with empty data");
	}

	// Check minimum observations
	idx_t min_obs = data.options.intercept ? (p + 2) : (p + 1);
	if (n < min_obs) {
		if (data.options.intercept) {
			throw InvalidInputException(
			    "Insufficient observations: need at least %d observations for %d features + intercept, got %d", min_obs,
			    p, n);
		} else {
			throw InvalidInputException(
			    "Insufficient observations: need at least %d observations for %d features, got %d", min_obs, p, n);
		}
	}

	data.n_obs = n;
	data.n_features = p;

	ANOFOX_DEBUG("Computing Elastic Net with " << n << " observations and " << p << " features, alpha="
	                                           << data.options.alpha << ", lambda=" << data.options.lambda
	                                           << (data.options.intercept ? " (with intercept)" : " (no intercept)"));

	// Build design matrix X and response vector y
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	// Fill X matrix with features
	for (idx_t j = 0; j < p; j++) {
		if (data.x_values[j].size() != n) {
			throw InvalidInputException("Feature %d has %d values, expected %d", j, data.x_values[j].size(), n);
		}
		for (idx_t i = 0; i < n; i++) {
			X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = data.x_values[j][i];
		}
	}

	// Fill y vector
	for (idx_t i = 0; i < n; i++) {
		y(static_cast<Eigen::Index>(i)) = data.y_values[i];
	}

	// If intercept requested, center the data before fitting
	Eigen::VectorXd y_centered = y;
	Eigen::MatrixXd X_centered = X;
	double y_mean = 0.0;
	Eigen::VectorXd x_means(static_cast<Eigen::Index>(p));

	if (data.options.intercept) {
		// Compute means
		y_mean = y.mean();
		x_means = X.colwise().mean();

		// Center data
		y_centered = y.array() - y_mean;
		for (idx_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			X_centered.col(j_idx) = X.col(j_idx).array() - x_means(j_idx);
		}
	}

	// Fit Elastic Net using coordinate descent
	auto result = ElasticNetSolver::Fit(y_centered, X_centered, data.options.alpha, data.options.lambda);

	// Store coefficients
	data.coefficients.resize(p);
	for (idx_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		data.coefficients[j] = result.coefficients[j_idx];
	}

	// Compute intercept
	if (data.options.intercept) {
		// intercept = ȳ - sum(beta_j * x̄_j)
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			beta_dot_xmean += result.coefficients[j_idx] * x_means(j_idx);
		}
		data.intercept = y_mean - beta_dot_xmean;
	} else {
		data.intercept = 0.0;
	}

	// Store fit statistics
	data.r_squared = result.r_squared;
	data.adj_r_squared = result.adj_r_squared;
	data.mse = result.mse;
	data.rmse = result.rmse;
	data.n_nonzero = result.n_nonzero;

	// Compute extended metadata if full_output=true
	if (data.options.full_output) {
		data.df_residual = n > data.n_nonzero ? (n - data.n_nonzero) : 0;

		// Track which coefficients are zero (feature selection result)
		data.is_zero.resize(p);
		for (idx_t j = 0; j < p; j++) {
			data.is_zero[j] = (std::abs(data.coefficients[j]) < 1e-10);
		}

		// Store x_train_means
		if (data.options.intercept) {
			data.x_train_means.resize(p);
			for (idx_t j = 0; j < p; j++) {
				data.x_train_means[j] = x_means(j);
			}
		}

		// Note: Elastic Net standard errors are complex due to regularization bias
		// We provide approximate SEs assuming the selected model is correct
		// For proper post-selection inference, use specialized methods
		data.coefficient_std_errors.resize(p);
		for (idx_t j = 0; j < p; j++) {
			if (data.is_zero[j]) {
				// Zero coefficient -> SE is undefined/not meaningful
				data.coefficient_std_errors[j] = std::numeric_limits<double>::quiet_NaN();
			} else {
				// Approximate SE for non-zero coefficients (naive, ignores selection)
				// SE ≈ sqrt(MSE) / sqrt(n) as a rough estimate
				data.coefficient_std_errors[j] = std::sqrt(data.mse / static_cast<double>(n));
			}
		}

		// Approximate intercept SE
		if (data.options.intercept) {
			data.intercept_std_error = std::sqrt(data.mse / static_cast<double>(n));
		} else {
			data.intercept_std_error = std::numeric_limits<double>::quiet_NaN();
		}
	}

	ANOFOX_DEBUG("Elastic Net complete: R² = " << data.r_squared << ", nonzero = " << data.n_nonzero << "/" << p
	                                           << ", converged = " << result.converged
	                                           << ", iterations = " << result.n_iterations);
}

static unique_ptr<FunctionData> ElasticNetFitBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("Elastic Net fit (array-based) bind phase");

	auto result = make_uniq<ElasticNetFitBindData>();

	// Expected parameters: y (DOUBLE[]), x (DOUBLE[][]), [options (MAP)]

	if (input.inputs.size() < 2) {
		throw InvalidInputException("anofox_statistics_elastic_net requires at least 2 parameters: "
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

	ANOFOX_INFO("Fitting Elastic Net with " << n << " observations and " << result->x_values.size()
	                                        << " features, alpha=" << result->options.alpha
	                                        << ", lambda=" << result->options.lambda);

	// Perform Elastic Net fitting
	ComputeElasticNet(*result);

	ANOFOX_INFO("Elastic Net fit completed: R² = " << result->r_squared << ", nonzero=" << result->n_nonzero);

	// Set return schema (basic columns)
	names = {"coefficients", "intercept",  "r_squared", "adj_r_squared", "mse",      "rmse",
	         "n_obs",        "n_features", "alpha",     "lambda",        "n_nonzero"};

	return_types = {LogicalType::LIST(LogicalType::DOUBLE),
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT};

	// Add extended columns if full_output=true
	if (result->options.full_output) {
		names.push_back("coefficient_std_errors");
		names.push_back("intercept_std_error");
		names.push_back("df_residual");
		names.push_back("is_zero");
		names.push_back("x_train_means");

		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // coefficient_std_errors
		return_types.push_back(LogicalType::DOUBLE);                    // intercept_std_error
		return_types.push_back(LogicalType::BIGINT);                    // df_residual
		return_types.push_back(LogicalType::LIST(LogicalType::BOOLEAN)); // is_zero
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // x_train_means
	}

	return std::move(result);
}

static void ElasticNetFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->CastNoConst<ElasticNetFitBindData>();

	if (bind_data.result_returned) {
		return; // Already returned the single result row
	}

	bind_data.result_returned = true;
	output.SetCardinality(1);

	// Return results
	vector<Value> coeffs_values;
	for (idx_t i = 0; i < bind_data.coefficients.size(); i++) {
		coeffs_values.push_back(Value(bind_data.coefficients[i]));
	}

	// Basic columns (always present)
	idx_t col_idx = 0;
	output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, coeffs_values));
	output.data[col_idx++].SetValue(0, Value(bind_data.intercept));
	output.data[col_idx++].SetValue(0, Value(bind_data.r_squared));
	output.data[col_idx++].SetValue(0, Value(bind_data.adj_r_squared));
	output.data[col_idx++].SetValue(0, Value(bind_data.mse));
	output.data[col_idx++].SetValue(0, Value(bind_data.rmse));
	output.data[col_idx++].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_obs)));
	output.data[col_idx++].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_features)));
	output.data[col_idx++].SetValue(0, Value(bind_data.options.alpha));
	output.data[col_idx++].SetValue(0, Value(bind_data.options.lambda));
	output.data[col_idx++].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_nonzero)));

	// Extended columns (only if full_output=true)
	if (bind_data.options.full_output) {
		// coefficient_std_errors
		vector<Value> se_values;
		for (idx_t i = 0; i < bind_data.coefficient_std_errors.size(); i++) {
			double se = bind_data.coefficient_std_errors[i];
			if (std::isnan(se)) {
				se_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				se_values.push_back(Value(se));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, se_values));

		// intercept_std_error
		if (std::isnan(bind_data.intercept_std_error)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.intercept_std_error));
		}

		// df_residual
		output.data[col_idx++].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.df_residual)));

		// is_zero
		vector<Value> zero_values;
		for (idx_t i = 0; i < bind_data.is_zero.size(); i++) {
			zero_values.push_back(Value::BOOLEAN(bind_data.is_zero[i]));
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::BOOLEAN, zero_values));

		// x_train_means
		vector<Value> means_values;
		for (idx_t i = 0; i < bind_data.x_train_means.size(); i++) {
			means_values.push_back(Value(bind_data.x_train_means[i]));
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, means_values));
	}
}

void ElasticNetFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_elastic_net (array-based v1.4.1 with MAP options)");

	// Signature: anofox_statistics_elastic_net(y DOUBLE[], x DOUBLE[][], [options MAP])
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                   // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)) // x: DOUBLE[][]
	};

	TableFunction function("anofox_statistics_elastic_net", arguments, ElasticNetFitExecute, ElasticNetFitBind);
	function.named_parameters["options"] = LogicalType::ANY;

	loader.RegisterFunction(function);

	ANOFOX_INFO("anofox_statistics_elastic_net registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
