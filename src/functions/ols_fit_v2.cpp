#include "ols_fit.hpp"
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
 * @brief OLS fit using array inputs with MAP-based options (v1.4.1 compatible)
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_ols(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       options := MAP{'intercept': true}
 *   )
 *
 * Parameters:
 *   - y: Response variable (DOUBLE[])
 *   - x: Feature matrix (DOUBLE[][], each inner array is one feature)
 *   - options: Optional MAP with keys: 'intercept' (BOOLEAN, default true)
 */

struct OlsFitArrayBindData : public FunctionData {
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

	// Rank-deficiency tracking
	vector<bool> is_aliased; // True for constant/aliased features
	idx_t rank = 0;          // Numerical rank (rank <= n_features)

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<OlsFitArrayBindData>();
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
 * Multi-variable OLS implementation using rank-deficient solver
 * Handles constant and aliased features gracefully (R-like behavior)
 */
static void ComputeOLS(OlsFitArrayBindData &data) {
	idx_t n = data.y_values.size();
	idx_t p = data.x_values.size();

	if (n == 0 || p == 0) {
		throw InvalidInputException("Cannot fit OLS with empty data");
	}

	// Check minimum observations
	// With intercept: need n >= p + 2 (p features + intercept + 1 for degrees of freedom)
	// Without intercept: need n >= p + 1
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

	ANOFOX_DEBUG("Computing OLS with " << n << " observations and " << p << " features"
	                                   << (data.options.intercept ? " (with intercept)" : " (no intercept)"));

	// Build design matrix X (without intercept) and response vector y
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	// Fill X matrix with features (no intercept column)
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

	// Use rank-deficient solver on centered data (if intercept) or original data (if no intercept)
	auto result = RankDeficientOls::Fit(y_centered, X_centered);

	// Store rank and aliasing info
	data.rank = result.rank;
	data.is_aliased.resize(p);
	data.coefficients.resize(p);

	for (idx_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		data.is_aliased[j] = result.is_aliased[j_idx];
		data.coefficients[j] = result.coefficients[j_idx];
	}

	// Compute intercept
	if (data.options.intercept) {
		// intercept = ȳ - sum(beta_j * x̄_j) for non-aliased features
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			if (!result.is_aliased[j_idx]) {
				beta_dot_xmean += result.coefficients[j_idx] * x_means(j_idx);
			}
		}
		data.intercept = y_mean - beta_dot_xmean;
	} else {
		data.intercept = 0.0;
	}

	// Store fit statistics
	data.r_squared = result.r_squared;
	data.adj_r_squared = result.adj_r_squared;

	// Adjust MSE/RMSE for intercept if fitted
	// When intercept=true, we centered the data, so we need to account for
	// the implicit intercept parameter in the degrees of freedom
	if (data.options.intercept) {
		// Correct df: n - rank - 1 (subtract 1 for intercept)
		idx_t df = (n > result.rank + 1) ? (n - result.rank - 1) : 1;
		double ss_res = result.residuals.squaredNorm();
		data.mse = ss_res / static_cast<double>(df);
		data.rmse = std::sqrt(data.mse);
	} else {
		// No intercept: use df = n - rank
		data.mse = result.mse;
		data.rmse = result.rmse;
	}

	ANOFOX_DEBUG("OLS complete: R² = " << data.r_squared << ", rank = " << data.rank << "/" << p
	                                   << ", aliased = " << (p - data.rank));
}

static unique_ptr<FunctionData> OlsFitArrayBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("OLS fit (array-based) bind phase");

	auto result = make_uniq<OlsFitArrayBindData>();

	// Expected parameters: y (DOUBLE[]), x (DOUBLE[][]), [options (MAP)]

	if (input.inputs.size() < 2) {
		throw InvalidInputException("anofox_statistics_ols requires at least 2 parameters: "
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

	ANOFOX_INFO("Fitting OLS with " << n << " observations and " << result->x_values.size() << " features");

	// Perform OLS fitting
	ComputeOLS(*result);

	ANOFOX_INFO("OLS fit completed: R² = " << result->r_squared);

	// Set return schema
	names = {"coefficients", "intercept", "r_squared", "adj_r_squared", "mse", "rmse", "n_obs", "n_features"};

	return_types = {LogicalType::LIST(LogicalType::DOUBLE),
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT};

	return std::move(result);
}

static void OlsFitArrayExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->CastNoConst<OlsFitArrayBindData>();

	if (bind_data.result_returned) {
		return; // Already returned the single result row
	}

	bind_data.result_returned = true;
	output.SetCardinality(1);

	// Return results
	// Coefficients: Set NULL for aliased (NaN) coefficients
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
}

void OlsFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_ols (array-based v1.4.1 with MAP options)");

	// Signature: anofox_statistics_ols(y DOUBLE[], x DOUBLE[][], [options MAP])
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                     // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))   // x: DOUBLE[][]
	};

	TableFunction function("anofox_statistics_ols", arguments, OlsFitArrayExecute, OlsFitArrayBind);

	// Optional third parameter: options (MAP)
	function.varargs = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::ANY);

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_ols registered successfully with MAP-based options");
}

} // namespace anofox_statistics
} // namespace duckdb
