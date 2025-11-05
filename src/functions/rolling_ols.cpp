#include "rolling_ols.hpp"
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
 * Rolling Window OLS
 *
 * Computes OLS regression on sliding windows of observations.
 * For data with n observations and window size w:
 *   - Window 0: observations [0, w)
 *   - Window 1: observations [1, w+1)
 *   - ...
 *   - Window (n-w): observations [n-w, n)
 *
 * Returns one row per window with coefficients and statistics.
 */

struct WindowResult {
	idx_t window_start;
	idx_t window_end;
	vector<double> coefficients;
	double intercept;
	double r_squared;
	double mse;
	idx_t n_obs;
	idx_t n_features;

	// Rank-deficiency tracking
	vector<bool> is_aliased;
	idx_t rank;
};

struct RollingOlsBindData : public FunctionData {
	vector<double> y_values;
	vector<vector<double>> x_values;
	RegressionOptions options;

	// Results (one per window)
	vector<WindowResult> windows;
	idx_t result_index = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<RollingOlsBindData>();
		result->y_values = y_values;
		result->x_values = x_values;
		result->options = options;
		result->windows = windows;
		result->result_index = result_index;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Compute OLS on a single window of data
 */
static WindowResult ComputeWindowOLS(const vector<double> &y_values, const vector<vector<double>> &x_values,
                                     idx_t window_start, idx_t window_end, bool add_intercept) {

	WindowResult result;
	result.window_start = window_start;
	result.window_end = window_end;

	idx_t n = window_end - window_start; // Window size
	idx_t p = x_values.size();           // Number of features

	result.n_obs = n;
	result.n_features = p;

	// Build design matrix X (n x p) and response vector y (n x 1) for this window
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	// Fill X and y with window data
	for (idx_t j = 0; j < p; j++) {
		for (idx_t i = 0; i < n; i++) {
			X(i, j) = x_values[j][window_start + i];
		}
	}
	for (idx_t i = 0; i < n; i++) {
		y(i) = y_values[window_start + i];
	}

	// Center data if fitting with intercept (same as OLS, WLS, Ridge, VIF)
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

	// Use rank-deficient OLS solver on CENTERED data (if add_intercept=true)
	auto ols_result = RankDeficientOls::Fit(y_work, X_work);

	// Store rank and aliasing info
	result.rank = ols_result.rank;
	result.is_aliased.resize(p);
	for (idx_t i = 0; i < p; i++) {
		result.is_aliased[i] = ols_result.is_aliased[i];
	}

	// Store coefficients (NaN for aliased features)
	result.coefficients.resize(p);
	for (idx_t i = 0; i < p; i++) {
		result.coefficients[i] = ols_result.coefficients[i];
	}

	// Compute intercept (coefficients are now for centered data)
	if (add_intercept) {
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				beta_dot_xmean += ols_result.coefficients[j] * x_means(j);
			}
		}
		result.intercept = y_mean - beta_dot_xmean;
	} else {
		result.intercept = 0.0;
	}

	// Compute predictions using only non-aliased features
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			y_pred += ols_result.coefficients[j] * X.col(j);
		}
	}
	if (add_intercept) {
		y_pred.array() += result.intercept;
	}

	// Compute statistics
	Eigen::VectorXd residuals = y - y_pred;
	double ss_res = residuals.squaredNorm();

	double ss_tot;
	if (add_intercept) {
		// With intercept: total sum of squares from mean
		ss_tot = (y.array() - y_mean).square().sum();
	} else {
		// No intercept: total sum of squares from zero
		ss_tot = y.squaredNorm();
	}

	result.r_squared = (ss_tot > 0) ? 1.0 - (ss_res / ss_tot) : 0.0;
	result.mse = ss_res / n;

	return result;
}

/**
 * Compute all rolling windows
 */
static void ComputeRollingOLS(RollingOlsBindData &data) {
	idx_t n = data.y_values.size();
	idx_t p = data.x_values.size();
	idx_t w = data.options.window_size;

	if (n == 0 || p == 0) {
		throw InvalidInputException("Cannot fit rolling OLS with empty data");
	}

	if (w < p + 1) {
		throw InvalidInputException("Window size must be at least %d (n_features + 1), got %d", p + 1, w);
	}

	if (w > n) {
		throw InvalidInputException("Window size (%d) cannot exceed number of observations (%d)", w, n);
	}

	ANOFOX_DEBUG("Computing rolling OLS with " << n << " observations, " << p << " features, window size = " << w);

	// Compute OLS for each window
	idx_t n_windows = n - w + 1;
	data.windows.reserve(n_windows);

	for (idx_t i = 0; i < n_windows; i++) {
		idx_t window_start = i;
		idx_t window_end = i + w;

		WindowResult window_result =
		    ComputeWindowOLS(data.y_values, data.x_values, window_start, window_end, data.options.intercept);

		data.windows.push_back(window_result);

		ANOFOX_DEBUG("Window [" << window_start << "," << window_end << "): R² = " << window_result.r_squared
		                        << ", coeff[0] = " << window_result.coefficients[0]);
	}

	ANOFOX_INFO("Rolling OLS complete: computed " << n_windows << " windows");
}

static unique_ptr<FunctionData> RollingOlsBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("Rolling OLS bind phase");

	auto result = make_uniq<RollingOlsBindData>();

	// Expected parameters: y (DOUBLE[]), x (DOUBLE[][]), options (MAP)

	if (input.inputs.size() < 3) {
		throw InvalidInputException("anofox_statistics_rolling_ols requires 3 parameters: "
		                            "y (DOUBLE[]), x (DOUBLE[][]), options (MAP with 'window_size' key)");
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

	// Extract options (third parameter - MAP, required for window functions)
	if (input.inputs[2].type().id() != LogicalTypeId::MAP) {
		throw InvalidInputException("Third parameter (options) must be a MAP with 'window_size' key");
	}
	result->options = RegressionOptions::ParseFromMap(input.inputs[2]);
	result->options.Validate();

	// Validate window_size was provided
	if (result->options.window_size == 0) {
		throw InvalidInputException("options MAP must include 'window_size' (BIGINT)");
	}

	ANOFOX_INFO("Fitting rolling OLS with " << n << " observations, " << result->x_values.size()
	                                        << " features, window_size=" << result->options.window_size);

	// Perform rolling OLS computation
	ComputeRollingOLS(*result);

	ANOFOX_INFO("Rolling OLS completed: " << result->windows.size() << " windows");

	// Set return schema
	names = {"window_start", "window_end", "coefficients", "intercept", "r_squared", "mse", "n_obs", "n_features"};

	return_types = {
	    LogicalType::BIGINT,                    // window_start
	    LogicalType::BIGINT,                    // window_end
	    LogicalType::LIST(LogicalType::DOUBLE), // coefficients
	    LogicalType::DOUBLE,                    // intercept
	    LogicalType::DOUBLE,                    // r_squared
	    LogicalType::DOUBLE,                    // mse
	    LogicalType::BIGINT,                    // n_obs
	    LogicalType::BIGINT                     // n_features
	};

	return std::move(result);
}

static void RollingOlsExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {

	auto &bind_data = data.bind_data->CastNoConst<RollingOlsBindData>();

	idx_t output_idx = 0;
	idx_t max_output = STANDARD_VECTOR_SIZE;

	// Stream results one window at a time
	while (output_idx < max_output && bind_data.result_index < bind_data.windows.size()) {
		auto &window = bind_data.windows[bind_data.result_index];

		// Column 0: window_start
		output.data[0].SetValue(output_idx, Value::BIGINT(window.window_start));

		// Column 1: window_end
		output.data[1].SetValue(output_idx, Value::BIGINT(window.window_end));

		// Column 2: coefficients (convert NaN to NULL for aliased features)
		vector<Value> coeffs_values;
		for (idx_t i = 0; i < window.coefficients.size(); i++) {
			double coef = window.coefficients[i];
			if (std::isnan(coef)) {
				// Aliased coefficient -> NULL
				coeffs_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				coeffs_values.push_back(Value(coef));
			}
		}
		output.data[2].SetValue(output_idx, Value::LIST(LogicalType::DOUBLE, coeffs_values));

		// Column 3: intercept
		output.data[3].SetValue(output_idx, Value(window.intercept));

		// Column 4: r_squared
		output.data[4].SetValue(output_idx, Value(window.r_squared));

		// Column 5: mse
		output.data[5].SetValue(output_idx, Value(window.mse));

		// Column 6: n_obs
		output.data[6].SetValue(output_idx, Value::BIGINT(window.n_obs));

		// Column 7: n_features
		output.data[7].SetValue(output_idx, Value::BIGINT(window.n_features));

		bind_data.result_index++;
		output_idx++;
	}

	output.SetCardinality(output_idx);
}

void RollingOlsFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_rolling_ols with sliding window regression");

	// Signature: anofox_statistics_rolling_ols(y DOUBLE[], x DOUBLE[][], options MAP)
	// options MAP must include 'window_size' (BIGINT), and optionally 'intercept' (BOOLEAN)
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                     // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x: DOUBLE[][]
	    LogicalType::MAP(LogicalType::VARCHAR, LogicalType::ANY)    // options: MAP (required)
	};

	TableFunction function("anofox_statistics_rolling_ols", arguments, RollingOlsExecute, RollingOlsBind);

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_rolling_ols registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
