#include "elastic_net_fit.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"
#include "../bridge/libanostat_wrapper.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Elastic Net fit using array inputs with MAP-based options
 *
 * Signature:
 *   SELECT * FROM anofox_stats_elastic_net(
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

	// New statistical metrics (when full_output=true)
	double residual_standard_error = std::numeric_limits<double>::quiet_NaN();
	double f_statistic = std::numeric_limits<double>::quiet_NaN();
	double f_statistic_pvalue = std::numeric_limits<double>::quiet_NaN();
	double aic = std::numeric_limits<double>::quiet_NaN();
	double aicc = std::numeric_limits<double>::quiet_NaN();
	double bic = std::numeric_limits<double>::quiet_NaN();
	double log_likelihood = std::numeric_limits<double>::quiet_NaN();

	// Coefficient-level inference (when full_output=true)
	vector<double> coefficient_t_statistics;
	vector<double> coefficient_p_values;
	vector<double> coefficient_ci_lower;
	vector<double> coefficient_ci_upper;

	// Intercept-level inference (when full_output=true)
	double intercept_t_statistic = std::numeric_limits<double>::quiet_NaN();
	double intercept_p_value = std::numeric_limits<double>::quiet_NaN();
	double intercept_ci_lower = std::numeric_limits<double>::quiet_NaN();
	double intercept_ci_upper = std::numeric_limits<double>::quiet_NaN();

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
		result->residual_standard_error = residual_standard_error;
		result->f_statistic = f_statistic;
		result->f_statistic_pvalue = f_statistic_pvalue;
		result->aic = aic;
		result->aicc = aicc;
		result->bic = bic;
		result->log_likelihood = log_likelihood;
		result->coefficient_t_statistics = coefficient_t_statistics;
		result->coefficient_p_values = coefficient_p_values;
		result->coefficient_ci_lower = coefficient_ci_lower;
		result->coefficient_ci_upper = coefficient_ci_upper;
		result->intercept_t_statistic = intercept_t_statistic;
		result->intercept_p_value = intercept_p_value;
		result->intercept_ci_lower = intercept_ci_lower;
		result->intercept_ci_upper = intercept_ci_upper;
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

	// Fit Elastic Net using libanostat bridge layer
	auto result =
	    bridge::LibanostatWrapper::FitElasticNet(data.y_values, // vector<double>
	                                             data.x_values, // vector<vector<double>> (column-major)
	                                             data.options,  // RegressionOptions (includes alpha and lambda)
	                                             data.options.full_output, // compute std errors if full_output
	                                             false);                   // row_major=false (column-major data)

	// Extract coefficients
	data.coefficients = bridge::TypeConverters::ExtractCoefficients(result);

	// Compute intercept from centered coefficients
	if (data.options.intercept) {
		// Compute x_means and y_mean
		Eigen::VectorXd x_means(p);
		for (idx_t j = 0; j < p; j++) {
			double sum = 0.0;
			for (idx_t i = 0; i < n; i++) {
				sum += data.x_values[j][i];
			}
			x_means(j) = sum / static_cast<double>(n);
		}

		double y_mean = 0.0;
		for (idx_t i = 0; i < n; i++) {
			y_mean += data.y_values[i];
		}
		y_mean /= static_cast<double>(n);

		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			beta_dot_xmean += data.coefficients[j] * x_means[j];
		}
		data.intercept = y_mean - beta_dot_xmean;
	} else {
		data.intercept = 0.0;
	}

	// Extract fit statistics
	data.r_squared = bridge::TypeConverters::ExtractRSquared(result);
	data.adj_r_squared = bridge::TypeConverters::ExtractAdjRSquared(result);
	data.mse = bridge::TypeConverters::ExtractMSE(result);
	data.rmse = bridge::TypeConverters::ExtractRMSE(result);

	// Count non-zero coefficients
	data.n_nonzero = 0;
	for (idx_t j = 0; j < p; j++) {
		if (std::abs(data.coefficients[j]) >= 1e-10) {
			data.n_nonzero++;
		}
	}

	// Compute extended metadata if full_output=true
	if (data.options.full_output) {
		// Account for intercept in df calculation: df_residual = n - (p_full)
		// where p_full = n_nonzero + (intercept ? 1 : 0)
		idx_t df_model = data.n_nonzero + (data.options.intercept ? 1 : 0);
		data.df_residual = n > df_model ? (n - df_model) : 0;

		// Track which coefficients are zero (feature selection result)
		data.is_zero.resize(p);
		for (idx_t j = 0; j < p; j++) {
			data.is_zero[j] = (std::abs(data.coefficients[j]) < 1e-10);
		}

		// Store x_train_means for intercept computation
		if (data.options.intercept) {
			data.x_train_means.resize(p);
			for (idx_t j = 0; j < p; j++) {
				double sum = 0.0;
				for (idx_t i = 0; i < n; i++) {
					sum += data.x_values[j][i];
				}
				data.x_train_means[j] = sum / static_cast<double>(n);
			}
		}

		// Note: Elastic Net standard errors are complex due to regularization bias
		// libanostat returns NaN for Elastic Net standard errors (bootstrap recommended)
		if (result.has_std_errors) {
			data.coefficient_std_errors = bridge::TypeConverters::ExtractStdErrors(result);
		} else {
			// Provide approximate SEs assuming the selected model is correct
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
		}

		// Approximate intercept SE
		if (data.options.intercept) {
			data.intercept_std_error = std::sqrt(data.mse / static_cast<double>(n));
		} else {
			data.intercept_std_error = std::numeric_limits<double>::quiet_NaN();
		}

		// Extract new statistical metrics
		data.residual_standard_error = bridge::TypeConverters::ExtractResidualStandardError(result);
		data.f_statistic = bridge::TypeConverters::ExtractFStatistic(result);
		data.f_statistic_pvalue = bridge::TypeConverters::ExtractFStatisticPValue(result);
		data.aic = bridge::TypeConverters::ExtractAIC(result);
		data.aicc = bridge::TypeConverters::ExtractAICc(result);
		data.bic = bridge::TypeConverters::ExtractBIC(result);
		data.log_likelihood = bridge::TypeConverters::ExtractLogLikelihood(result);

		// Extract coefficient-level inference
		if (result.has_std_errors) {
			data.coefficient_t_statistics = bridge::TypeConverters::ExtractTStatistics(result);
			data.coefficient_p_values = bridge::TypeConverters::ExtractPValues(result);
			data.coefficient_ci_lower = bridge::TypeConverters::ExtractCILower(result);
			data.coefficient_ci_upper = bridge::TypeConverters::ExtractCIUpper(result);

			// Extract intercept-level inference
			data.intercept_t_statistic = bridge::TypeConverters::ExtractInterceptTStatistic(result);
			data.intercept_p_value = bridge::TypeConverters::ExtractInterceptPValue(result);
			data.intercept_ci_lower = bridge::TypeConverters::ExtractInterceptCILower(result);
			data.intercept_ci_upper = bridge::TypeConverters::ExtractInterceptCIUpper(result);
		}
	}

	ANOFOX_DEBUG("Elastic Net (via libanostat): R² = " << data.r_squared << ", nonzero = " << data.n_nonzero << "/" << p
	                                                   << ", α=" << data.options.alpha
	                                                   << ", λ=" << data.options.lambda);
}

static unique_ptr<FunctionData> ElasticNetFitBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("Elastic Net fit (array-based) bind phase");

	auto result = make_uniq<ElasticNetFitBindData>();

	// Expected parameters: y (DOUBLE[]), x (DOUBLE[][]), [options (MAP)]

	if (input.inputs.size() < 2) {
		throw InvalidInputException("anofox_stats_elastic_net requires at least 2 parameters: "
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

	// Validate that number of rows matches y
	if (x_outer.size() != n) {
		throw InvalidInputException("Array dimensions mismatch: y has %d elements, x has %d rows", n, x_outer.size());
	}

	// Get number of features from first row
	if (x_outer.empty()) {
		throw InvalidInputException("Second parameter (x) must have at least one row");
	}

	if (x_outer[0].type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("Second parameter (x) must be a 2D array where each element is DOUBLE[]");
	}

	auto first_row = ListValue::GetChildren(x_outer[0]);
	idx_t p = first_row.size();

	if (p == 0) {
		throw InvalidInputException("Second parameter (x) must have at least one feature");
	}

	// Initialize x_values with p features, each will hold n observations
	result->x_values.resize(p);

	// Transpose row-major input to column-major storage
	for (idx_t i = 0; i < n; i++) {
		if (x_outer[i].type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("Second parameter (x) must be a 2D array where each element is DOUBLE[]");
		}

		auto row = ListValue::GetChildren(x_outer[i]);
		if (row.size() != p) {
			throw InvalidInputException("Array dimensions mismatch: row 0 has %d features, row %d has %d features", p,
			                            i, row.size());
		}

		for (idx_t j = 0; j < p; j++) {
			result->x_values[j].push_back(row[j].GetValue<double>());
		}
	}

	// Extract options (third parameter - MAP, optional)
	if (input.inputs.size() >= 3) {
		if (input.inputs[2].type().id() == LogicalTypeId::MAP || input.inputs[2].type().id() == LogicalTypeId::STRUCT) {
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
	names = {"coefficients", "intercept",  "r2",    "adj_r2", "mse",      "rmse",
	         "n_obs",        "n_features", "alpha", "lambda", "n_nonzero"};

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

		// New statistical metrics
		names.push_back("residual_standard_error");
		names.push_back("f_statistic");
		names.push_back("f_statistic_pvalue");
		names.push_back("aic");
		names.push_back("aicc");
		names.push_back("bic");
		names.push_back("log_likelihood");

		// Coefficient-level inference
		names.push_back("coefficient_t_statistics");
		names.push_back("coefficient_p_values");
		names.push_back("coefficient_ci_lower");
		names.push_back("coefficient_ci_upper");

		// Intercept-level inference
		names.push_back("intercept_t_statistic");
		names.push_back("intercept_p_value");
		names.push_back("intercept_ci_lower");
		names.push_back("intercept_ci_upper");

		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // coefficient_std_errors
		return_types.push_back(LogicalType::DOUBLE);                     // intercept_std_error
		return_types.push_back(LogicalType::BIGINT);                     // df_residual
		return_types.push_back(LogicalType::LIST(LogicalType::BOOLEAN)); // is_zero
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // x_train_means

		// New statistical metrics
		return_types.push_back(LogicalType::DOUBLE); // residual_standard_error
		return_types.push_back(LogicalType::DOUBLE); // f_statistic
		return_types.push_back(LogicalType::DOUBLE); // f_statistic_pvalue
		return_types.push_back(LogicalType::DOUBLE); // aic
		return_types.push_back(LogicalType::DOUBLE); // aicc
		return_types.push_back(LogicalType::DOUBLE); // bic
		return_types.push_back(LogicalType::DOUBLE); // log_likelihood

		// Coefficient-level inference
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // coefficient_t_statistics
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // coefficient_p_values
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // coefficient_ci_lower
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // coefficient_ci_upper

		// Intercept-level inference
		return_types.push_back(LogicalType::DOUBLE); // intercept_t_statistic
		return_types.push_back(LogicalType::DOUBLE); // intercept_p_value
		return_types.push_back(LogicalType::DOUBLE); // intercept_ci_lower
		return_types.push_back(LogicalType::DOUBLE); // intercept_ci_upper
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

		// New statistical metrics
		if (std::isnan(bind_data.residual_standard_error)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.residual_standard_error));
		}

		if (std::isnan(bind_data.f_statistic)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.f_statistic));
		}

		if (std::isnan(bind_data.f_statistic_pvalue)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.f_statistic_pvalue));
		}

		if (std::isnan(bind_data.aic)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.aic));
		}

		if (std::isnan(bind_data.aicc)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.aicc));
		}

		if (std::isnan(bind_data.bic)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.bic));
		}

		if (std::isnan(bind_data.log_likelihood)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.log_likelihood));
		}

		// Coefficient-level inference
		vector<Value> t_stat_values;
		for (idx_t i = 0; i < bind_data.coefficient_t_statistics.size(); i++) {
			double val = bind_data.coefficient_t_statistics[i];
			if (std::isnan(val)) {
				t_stat_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				t_stat_values.push_back(Value(val));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, t_stat_values));

		vector<Value> p_val_values;
		for (idx_t i = 0; i < bind_data.coefficient_p_values.size(); i++) {
			double val = bind_data.coefficient_p_values[i];
			if (std::isnan(val)) {
				p_val_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				p_val_values.push_back(Value(val));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, p_val_values));

		vector<Value> ci_lower_values;
		for (idx_t i = 0; i < bind_data.coefficient_ci_lower.size(); i++) {
			double val = bind_data.coefficient_ci_lower[i];
			if (std::isnan(val)) {
				ci_lower_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				ci_lower_values.push_back(Value(val));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, ci_lower_values));

		vector<Value> ci_upper_values;
		for (idx_t i = 0; i < bind_data.coefficient_ci_upper.size(); i++) {
			double val = bind_data.coefficient_ci_upper[i];
			if (std::isnan(val)) {
				ci_upper_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				ci_upper_values.push_back(Value(val));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, ci_upper_values));

		// Intercept-level inference
		if (std::isnan(bind_data.intercept_t_statistic)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.intercept_t_statistic));
		}

		if (std::isnan(bind_data.intercept_p_value)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.intercept_p_value));
		}

		if (std::isnan(bind_data.intercept_ci_lower)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.intercept_ci_lower));
		}

		if (std::isnan(bind_data.intercept_ci_upper)) {
			output.data[col_idx++].SetValue(0, Value(LogicalType::DOUBLE));
		} else {
			output.data[col_idx++].SetValue(0, Value(bind_data.intercept_ci_upper));
		}
	}
}

void ElasticNetFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_stats_elastic_net (array-based with MAP options)");

	// Register 2-argument overload: (y DOUBLE[], x DOUBLE[][])
	vector<LogicalType> args_2 = {
	    LogicalType::LIST(LogicalType::DOUBLE),                   // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)) // x: DOUBLE[][]
	};

	// Register primary function with new naming convention
	TableFunction func_2("anofox_stats_elastic_net_fit", args_2, ElasticNetFitExecute, ElasticNetFitBind);
	loader.RegisterFunction(func_2);

	// Register alias without prefix
	TableFunction func_2_alias("elastic_net_fit", args_2, ElasticNetFitExecute, ElasticNetFitBind);
	loader.RegisterFunction(func_2_alias);

	// Register 3-argument overload: (y DOUBLE[], x DOUBLE[][], options MAP/STRUCT)
	vector<LogicalType> args_3 = {
	    LogicalType::LIST(LogicalType::DOUBLE),                    // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x: DOUBLE[][]
	    LogicalType::ANY                                           // options: MAP or STRUCT
	};

	TableFunction func_3("anofox_stats_elastic_net_fit", args_3, ElasticNetFitExecute, ElasticNetFitBind);
	loader.RegisterFunction(func_3);

	// Register alias without prefix
	TableFunction func_3_alias("elastic_net_fit", args_3, ElasticNetFitExecute, ElasticNetFitBind);
	loader.RegisterFunction(func_3_alias);

	ANOFOX_INFO("anofox_stats_elastic_net_fit registered successfully with alias elastic_net_fit");
}

} // namespace anofox_statistics
} // namespace duckdb
