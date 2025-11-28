#include "wls_fit.hpp"
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

	// Extended metadata (when full_output=true)
	vector<double> coefficient_std_errors;
	double intercept_std_error = std::numeric_limits<double>::quiet_NaN();
	idx_t df_residual = 0;
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
	double intercept_t_statistic = std::numeric_limits<double>::quiet_NaN();
	vector<double> coefficient_p_values;
	double intercept_p_value = std::numeric_limits<double>::quiet_NaN();
	vector<double> coefficient_ci_lower;
	double intercept_ci_lower = std::numeric_limits<double>::quiet_NaN();
	vector<double> coefficient_ci_upper;
	double intercept_ci_upper = std::numeric_limits<double>::quiet_NaN();

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
		result->coefficient_std_errors = coefficient_std_errors;
		result->intercept_std_error = intercept_std_error;
		result->df_residual = df_residual;
		result->x_train_means = x_train_means;
		result->residual_standard_error = residual_standard_error;
		result->f_statistic = f_statistic;
		result->f_statistic_pvalue = f_statistic_pvalue;
		result->aic = aic;
		result->aicc = aicc;
		result->bic = bic;
		result->log_likelihood = log_likelihood;
		result->coefficient_t_statistics = coefficient_t_statistics;
		result->intercept_t_statistic = intercept_t_statistic;
		result->coefficient_p_values = coefficient_p_values;
		result->intercept_p_value = intercept_p_value;
		result->coefficient_ci_lower = coefficient_ci_lower;
		result->intercept_ci_lower = intercept_ci_lower;
		result->coefficient_ci_upper = coefficient_ci_upper;
		result->intercept_ci_upper = intercept_ci_upper;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * WLS implementation using libanostat bridge layer
 * β = (X'WX)^(-1) X'Wy
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

	// Fit WLS using libanostat bridge layer
	auto result = bridge::LibanostatWrapper::FitWLS(data.y_values,            // vector<double>
	                                                data.x_values,            // vector<vector<double>> (column-major)
	                                                data.weights,             // vector<double>: observation weights
	                                                data.options,             // RegressionOptions
	                                                data.options.full_output, // compute std errors if full_output
	                                                false);                   // row_major=false (column-major data)

	// Extract coefficients and aliasing info
	data.coefficients = bridge::TypeConverters::ExtractCoefficients(result);
	data.is_aliased = bridge::TypeConverters::ExtractIsAliased(result);
	data.rank = bridge::TypeConverters::ExtractRank(result);

	// Compute intercept from centered coefficients
	if (data.options.intercept) {
		// Compute weighted means
		double sum_weights = 0.0;
		for (idx_t i = 0; i < n; i++) {
			sum_weights += data.weights[i];
		}

		double y_weighted_mean = 0.0;
		for (idx_t i = 0; i < n; i++) {
			y_weighted_mean += data.weights[i] * data.y_values[i];
		}
		y_weighted_mean /= sum_weights;

		Eigen::VectorXd x_weighted_means(p);
		for (idx_t j = 0; j < p; j++) {
			double sum = 0.0;
			for (idx_t i = 0; i < n; i++) {
				sum += data.weights[i] * data.x_values[j][i];
			}
			x_weighted_means(j) = sum / sum_weights;
		}

		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!data.is_aliased[j]) {
				beta_dot_xmean += data.coefficients[j] * x_weighted_means(j);
			}
		}
		data.intercept = y_weighted_mean - beta_dot_xmean;

		// Store weighted means for later use
		if (data.options.full_output) {
			data.x_train_means.resize(p);
			for (idx_t j = 0; j < p; j++) {
				data.x_train_means[j] = x_weighted_means(j);
			}
		}
	} else {
		data.intercept = 0.0;
	}

	// Extract fit statistics
	data.r_squared = bridge::TypeConverters::ExtractRSquared(result);
	data.adj_r_squared = bridge::TypeConverters::ExtractAdjRSquared(result);
	data.mse = bridge::TypeConverters::ExtractMSE(result);
	data.rmse = bridge::TypeConverters::ExtractRMSE(result);

	// Compute weighted MSE manually
	// (libanostat returns unweighted MSE)
	double sum_weights = 0.0;
	double ss_res_weighted = 0.0;
	for (idx_t i = 0; i < n; i++) {
		double y_pred = data.intercept;
		for (idx_t j = 0; j < p; j++) {
			if (!data.is_aliased[j]) {
				y_pred += data.coefficients[j] * data.x_values[j][i];
			}
		}
		double residual = data.y_values[i] - y_pred;
		ss_res_weighted += data.weights[i] * residual * residual;
		sum_weights += data.weights[i];
	}
	data.weighted_mse = ss_res_weighted / sum_weights;

	// Compute extended metadata if full_output=true
	if (data.options.full_output) {
		if (result.has_std_errors) {
			data.coefficient_std_errors = bridge::TypeConverters::ExtractStdErrors(result);
		}

		// Degrees of freedom
		idx_t df_model = data.rank + (data.options.intercept ? 1 : 0);
		data.df_residual = n > df_model ? (n - df_model) : 0;

		// Intercept standard error (approximation)
		if (data.options.intercept && result.has_std_errors) {
			double intercept_variance = data.weighted_mse / sum_weights;
			for (idx_t j = 0; j < p; j++) {
				if (!data.is_aliased[j] && !std::isnan(data.coefficient_std_errors[j])) {
					double se_beta_j = data.coefficient_std_errors[j];
					intercept_variance += se_beta_j * se_beta_j * data.x_train_means[j] * data.x_train_means[j];
				}
			}
			data.intercept_std_error = std::sqrt(intercept_variance);
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
			data.intercept_t_statistic = bridge::TypeConverters::ExtractInterceptTStatistic(result);
			data.intercept_p_value = bridge::TypeConverters::ExtractInterceptPValue(result);
			data.intercept_ci_lower = bridge::TypeConverters::ExtractInterceptCILower(result);
			data.intercept_ci_upper = bridge::TypeConverters::ExtractInterceptCIUpper(result);
		}
	}

	ANOFOX_DEBUG("WLS (via libanostat): R² = " << data.r_squared << ", weighted MSE = " << data.weighted_mse
	                                           << ", coefficients = [" << data.coefficients[0]
	                                           << (p > 1 ? ", ...]" : "]"));
}

//===--------------------------------------------------------------------===//
// Lateral Join Support Structures (must be declared before WlsFitBind)
//===--------------------------------------------------------------------===//

/**
 * Bind data for in-out mode (lateral join support)
 * Stores only options, not data (data comes from input rows)
 */
struct WlsFitInOutBindData : public FunctionData {
	RegressionOptions options;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<WlsFitInOutBindData>();
		result->options = options;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Local state for in-out mode
 * Tracks which input rows have been processed
 */
struct WlsFitInOutLocalState : public LocalTableFunctionState {
	idx_t current_input_row = 0;
};

static unique_ptr<LocalTableFunctionState>
WlsFitInOutLocalInit(ExecutionContext &context, TableFunctionInitInput &input, GlobalTableFunctionState *global_state) {
	return make_uniq<WlsFitInOutLocalState>();
}

//===--------------------------------------------------------------------===//
// Bind Function (supports both literal and lateral join modes)
//===--------------------------------------------------------------------===//

static unique_ptr<FunctionData> WlsFitBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("WLS regression bind phase");

	// Determine if full_output is requested
	bool full_output = false;
	if (input.inputs.size() >= 4 && !input.inputs[3].IsNull()) {
		try {
			if (input.inputs[3].type().id() == LogicalTypeId::MAP ||
			    input.inputs[3].type().id() == LogicalTypeId::STRUCT) {
				auto opts = RegressionOptions::ParseFromMap(input.inputs[3]);
				full_output = opts.full_output;
			}
		} catch (...) {
			// Ignore parsing errors
		}
	}

	// Set return schema (basic columns)
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

	// Add extended columns if full_output=true
	if (full_output) {
		names.push_back("coefficient_std_errors");
		names.push_back("intercept_std_error");
		names.push_back("df_residual");
		names.push_back("is_aliased");
		names.push_back("x_train_means");
		names.push_back("residual_standard_error");
		names.push_back("f_statistic");
		names.push_back("f_statistic_pvalue");
		names.push_back("aic");
		names.push_back("aicc");
		names.push_back("bic");
		names.push_back("log_likelihood");
		names.push_back("coefficient_t_statistics");
		names.push_back("intercept_t_statistic");
		names.push_back("coefficient_p_values");
		names.push_back("intercept_p_value");
		names.push_back("coefficient_ci_lower");
		names.push_back("intercept_ci_lower");
		names.push_back("coefficient_ci_upper");
		names.push_back("intercept_ci_upper");

		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // coefficient_std_errors
		return_types.push_back(LogicalType::DOUBLE);                     // intercept_std_error
		return_types.push_back(LogicalType::BIGINT);                     // df_residual
		return_types.push_back(LogicalType::LIST(LogicalType::BOOLEAN)); // is_aliased
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // x_train_means
		return_types.push_back(LogicalType::DOUBLE);                     // residual_standard_error
		return_types.push_back(LogicalType::DOUBLE);                     // f_statistic
		return_types.push_back(LogicalType::DOUBLE);                     // f_statistic_pvalue
		return_types.push_back(LogicalType::DOUBLE);                     // aic
		return_types.push_back(LogicalType::DOUBLE);                     // aicc
		return_types.push_back(LogicalType::DOUBLE);                     // bic
		return_types.push_back(LogicalType::DOUBLE);                     // log_likelihood
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // coefficient_t_statistics
		return_types.push_back(LogicalType::DOUBLE);                     // intercept_t_statistic
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // coefficient_p_values
		return_types.push_back(LogicalType::DOUBLE);                     // intercept_p_value
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // coefficient_ci_lower
		return_types.push_back(LogicalType::DOUBLE);                     // intercept_ci_lower
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE));  // coefficient_ci_upper
		return_types.push_back(LogicalType::DOUBLE);                     // intercept_ci_upper
	}

	// Check if this is being called for lateral joins (in-out function mode)
	// In that case, we don't have literal values to process
	if (input.inputs.size() >= 3 && !input.inputs[0].IsNull()) {
		// Check if the first input is actually a constant value
		bool is_literal = false;
		try {
			auto y_list = ListValue::GetChildren(input.inputs[0]);
			// If we can get children, it's a literal value - process it normally
			is_literal = true;

			auto result = make_uniq<WlsFitBindData>();

			// Extract y values
			for (const auto &val : y_list) {
				result->y_values.push_back(val.GetValue<double>());
			}

			idx_t n = result->y_values.size();
			ANOFOX_DEBUG("y has " << n << " observations");

			// Extract x values (second parameter - 2D array in row-major format)
			// Format: [[row1_feat1, row1_feat2, ...], [row2_feat1, row2_feat2, ...], ...]
			auto x_outer = ListValue::GetChildren(input.inputs[1]);

			// Validate we have the right number of rows
			if (x_outer.size() != n) {
				throw InvalidInputException("Array dimensions mismatch: y has %d elements, but x has %d rows", n,
				                            x_outer.size());
			}

			// Determine number of features from first row
			idx_t p = 0;
			if (n > 0) {
				auto first_row = ListValue::GetChildren(x_outer[0]);
				p = first_row.size();
			}

			// Initialize feature vectors
			result->x_values.resize(p);
			for (idx_t j = 0; j < p; j++) {
				result->x_values[j].reserve(n);
			}

			// Transpose: convert row-major input to column-major storage
			for (idx_t i = 0; i < n; i++) {
				auto row = ListValue::GetChildren(x_outer[i]);
				if (row.size() != p) {
					throw InvalidInputException(
					    "Inconsistent number of features: row 0 has %d features, but row %d has %d features", p, i,
					    row.size());
				}
				for (idx_t j = 0; j < p; j++) {
					result->x_values[j].push_back(row[j].GetValue<double>());
				}
			}

			// Extract weights (third parameter - DOUBLE[])
			auto weights_list = ListValue::GetChildren(input.inputs[2]);
			for (const auto &val : weights_list) {
				result->weights.push_back(val.GetValue<double>());
			}

			// Extract options (fourth parameter - MAP, optional)
			if (input.inputs.size() >= 4 && !input.inputs[3].IsNull()) {
				if (input.inputs[3].type().id() == LogicalTypeId::MAP ||
				    input.inputs[3].type().id() == LogicalTypeId::STRUCT) {
					result->options = RegressionOptions::ParseFromMap(input.inputs[3]);
					result->options.Validate();
				}
			}

			ANOFOX_INFO("Fitting WLS with " << n << " observations and " << result->x_values.size() << " features");

			// Perform WLS fitting
			ComputeWLS(*result);

			ANOFOX_INFO("WLS fit completed: R² = " << result->r_squared);

			return std::move(result);

		} catch (...) {
			// If is_literal is true, we successfully extracted y_list so we're in literal mode
			// Any exception after that point should propagate, not fall through to LATERAL mode
			if (is_literal) {
				throw;
			}

			// If we can't get children, it's probably a column reference (lateral join mode)
			// Return minimal bind data for in-out function mode
			auto result = make_uniq<WlsFitInOutBindData>();

			// Extract options if provided as literal
			if (input.inputs.size() >= 4 && !input.inputs[3].IsNull()) {
				try {
					if (input.inputs[3].type().id() == LogicalTypeId::MAP ||
					    input.inputs[3].type().id() == LogicalTypeId::STRUCT) {
						result->options = RegressionOptions::ParseFromMap(input.inputs[3]);
						result->options.Validate();
					}
				} catch (...) {
					// Ignore if we can't parse options
				}
			}

			return std::move(result);
		}
	}

	// Fallback: return minimal bind data
	auto result = make_uniq<WlsFitInOutBindData>();
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

	// Basic columns (always present)
	idx_t col_idx = 0;
	output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, coeffs_values));
	output.data[col_idx++].SetValue(0, Value(bind_data.intercept));
	output.data[col_idx++].SetValue(0, Value(bind_data.r_squared));
	output.data[col_idx++].SetValue(0, Value(bind_data.adj_r_squared));
	output.data[col_idx++].SetValue(0, Value(bind_data.mse));
	output.data[col_idx++].SetValue(0, Value(bind_data.rmse));
	output.data[col_idx++].SetValue(0, Value(bind_data.weighted_mse));
	output.data[col_idx++].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_obs)));
	output.data[col_idx++].SetValue(0, Value::BIGINT(static_cast<int64_t>(bind_data.n_features)));

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

		// is_aliased
		vector<Value> aliased_values;
		for (idx_t i = 0; i < bind_data.is_aliased.size(); i++) {
			aliased_values.push_back(Value::BOOLEAN(bind_data.is_aliased[i]));
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::BOOLEAN, aliased_values));

		// x_train_means
		vector<Value> means_values;
		for (idx_t i = 0; i < bind_data.x_train_means.size(); i++) {
			means_values.push_back(Value(bind_data.x_train_means[i]));
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, means_values));

		// New statistical metrics
		output.data[col_idx++].SetValue(0, Value(bind_data.residual_standard_error));
		output.data[col_idx++].SetValue(0, Value(bind_data.f_statistic));
		output.data[col_idx++].SetValue(0, Value(bind_data.f_statistic_pvalue));
		output.data[col_idx++].SetValue(0, Value(bind_data.aic));
		output.data[col_idx++].SetValue(0, Value(bind_data.aicc));
		output.data[col_idx++].SetValue(0, Value(bind_data.bic));
		output.data[col_idx++].SetValue(0, Value(bind_data.log_likelihood));

		// coefficient_t_statistics
		vector<Value> t_stat_values;
		for (idx_t i = 0; i < bind_data.coefficient_t_statistics.size(); i++) {
			double t = bind_data.coefficient_t_statistics[i];
			if (std::isnan(t)) {
				t_stat_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				t_stat_values.push_back(Value(t));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, t_stat_values));

		// intercept_t_statistic
		output.data[col_idx++].SetValue(0, Value(bind_data.intercept_t_statistic));

		// coefficient_p_values
		vector<Value> p_val_values;
		for (idx_t i = 0; i < bind_data.coefficient_p_values.size(); i++) {
			double p = bind_data.coefficient_p_values[i];
			if (std::isnan(p)) {
				p_val_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				p_val_values.push_back(Value(p));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, p_val_values));

		// intercept_p_value
		output.data[col_idx++].SetValue(0, Value(bind_data.intercept_p_value));

		// coefficient_ci_lower
		vector<Value> ci_lower_values;
		for (idx_t i = 0; i < bind_data.coefficient_ci_lower.size(); i++) {
			double ci = bind_data.coefficient_ci_lower[i];
			if (std::isnan(ci)) {
				ci_lower_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				ci_lower_values.push_back(Value(ci));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, ci_lower_values));

		// intercept_ci_lower
		output.data[col_idx++].SetValue(0, Value(bind_data.intercept_ci_lower));

		// coefficient_ci_upper
		vector<Value> ci_upper_values;
		for (idx_t i = 0; i < bind_data.coefficient_ci_upper.size(); i++) {
			double ci = bind_data.coefficient_ci_upper[i];
			if (std::isnan(ci)) {
				ci_upper_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				ci_upper_values.push_back(Value(ci));
			}
		}
		output.data[col_idx++].SetValue(0, Value::LIST(LogicalType::DOUBLE, ci_upper_values));

		// intercept_ci_upper
		output.data[col_idx++].SetValue(0, Value(bind_data.intercept_ci_upper));
	}
}

/**
 * In-out function for lateral join support
 * Processes rows from input table, computes regression for each row
 */
static OperatorResultType WlsFitInOut(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                      DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<WlsFitInOutBindData>();
	auto &state = data_p.local_state->Cast<WlsFitInOutLocalState>();

	// Process all input rows
	if (state.current_input_row >= input.size()) {
		// Finished processing all rows in this chunk
		state.current_input_row = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Flatten input vectors for easier access
	input.Flatten();

	idx_t output_count = 0;
	while (state.current_input_row < input.size() && output_count < STANDARD_VECTOR_SIZE) {
		idx_t row_idx = state.current_input_row;

		// Check for NULL inputs (WLS has 3 required inputs: y, x, weights)
		if (FlatVector::IsNull(input.data[0], row_idx) || FlatVector::IsNull(input.data[1], row_idx) ||
		    FlatVector::IsNull(input.data[2], row_idx)) {
			// Skip NULL rows - don't produce output
			state.current_input_row++;
			continue;
		}

		// Extract y values from column 0 (LIST of DOUBLE)
		auto y_list_value = FlatVector::GetValue<list_entry_t>(input.data[0], row_idx);
		auto &y_child_vector = ListVector::GetEntry(input.data[0]);
		vector<double> y_values;
		for (idx_t i = y_list_value.offset; i < y_list_value.offset + y_list_value.length; i++) {
			if (!FlatVector::IsNull(y_child_vector, i)) {
				y_values.push_back(FlatVector::GetValue<double>(y_child_vector, i));
			}
		}

		// Extract x values from column 1 (LIST of LIST of DOUBLE)
		auto x_outer_list = FlatVector::GetValue<list_entry_t>(input.data[1], row_idx);
		auto &x_outer_vector = ListVector::GetEntry(input.data[1]);
		vector<vector<double>> x_values;

		for (idx_t j = x_outer_list.offset; j < x_outer_list.offset + x_outer_list.length; j++) {
			if (FlatVector::IsNull(x_outer_vector, j)) {
				continue;
			}

			auto x_inner_list = FlatVector::GetValue<list_entry_t>(x_outer_vector, j);
			auto &x_inner_vector = ListVector::GetEntry(x_outer_vector);
			vector<double> x_feature;

			for (idx_t k = x_inner_list.offset; k < x_inner_list.offset + x_inner_list.length; k++) {
				if (!FlatVector::IsNull(x_inner_vector, k)) {
					x_feature.push_back(FlatVector::GetValue<double>(x_inner_vector, k));
				}
			}

			x_values.push_back(x_feature);
		}

		// Extract weights from column 2 (LIST of DOUBLE)
		auto weights_list_value = FlatVector::GetValue<list_entry_t>(input.data[2], row_idx);
		auto &weights_child_vector = ListVector::GetEntry(input.data[2]);
		vector<double> weights;
		for (idx_t i = weights_list_value.offset; i < weights_list_value.offset + weights_list_value.length; i++) {
			if (!FlatVector::IsNull(weights_child_vector, i)) {
				weights.push_back(FlatVector::GetValue<double>(weights_child_vector, i));
			}
		}

		// Create temporary bind data for computation (reuse existing WlsFitBindData)
		WlsFitBindData temp_data;
		temp_data.y_values = y_values;
		temp_data.x_values = x_values;
		temp_data.weights = weights;
		temp_data.options = bind_data.options;

		// Compute regression using existing logic
		try {
			ComputeWLS(temp_data);
		} catch (const Exception &e) {
			// If regression fails for this row, skip it
			state.current_input_row++;
			continue;
		}

		// Write results to output
		vector<Value> coeffs_values;
		for (idx_t i = 0; i < temp_data.coefficients.size(); i++) {
			double coef = temp_data.coefficients[i];
			if (std::isnan(coef)) {
				coeffs_values.push_back(Value(LogicalType::DOUBLE));
			} else {
				coeffs_values.push_back(Value(coef));
			}
		}

		idx_t col_idx = 0;
		output.data[col_idx++].SetValue(output_count, Value::LIST(LogicalType::DOUBLE, coeffs_values));
		output.data[col_idx++].SetValue(output_count, Value(temp_data.intercept));
		output.data[col_idx++].SetValue(output_count, Value(temp_data.r_squared));
		output.data[col_idx++].SetValue(output_count, Value(temp_data.adj_r_squared));
		output.data[col_idx++].SetValue(output_count, Value(temp_data.mse));
		output.data[col_idx++].SetValue(output_count, Value(temp_data.rmse));
		output.data[col_idx++].SetValue(output_count, Value(temp_data.weighted_mse));
		output.data[col_idx++].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.n_obs)));
		output.data[col_idx++].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.n_features)));

		// Extended columns (only if full_output=true)
		if (bind_data.options.full_output) {
			// coefficient_std_errors
			vector<Value> se_values;
			for (idx_t i = 0; i < temp_data.coefficient_std_errors.size(); i++) {
				double se = temp_data.coefficient_std_errors[i];
				if (std::isnan(se)) {
					se_values.push_back(Value(LogicalType::DOUBLE));
				} else {
					se_values.push_back(Value(se));
				}
			}
			output.data[col_idx++].SetValue(output_count, Value::LIST(LogicalType::DOUBLE, se_values));

			// intercept_std_error
			if (std::isnan(temp_data.intercept_std_error)) {
				output.data[col_idx++].SetValue(output_count, Value(LogicalType::DOUBLE));
			} else {
				output.data[col_idx++].SetValue(output_count, Value(temp_data.intercept_std_error));
			}

			// df_residual
			output.data[col_idx++].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.df_residual)));

			// is_aliased
			vector<Value> aliased_values;
			for (idx_t i = 0; i < temp_data.is_aliased.size(); i++) {
				aliased_values.push_back(Value::BOOLEAN(temp_data.is_aliased[i]));
			}
			output.data[col_idx++].SetValue(output_count, Value::LIST(LogicalType::BOOLEAN, aliased_values));

			// x_train_means
			vector<Value> means_values;
			for (idx_t i = 0; i < temp_data.x_train_means.size(); i++) {
				means_values.push_back(Value(temp_data.x_train_means[i]));
			}
			output.data[col_idx++].SetValue(output_count, Value::LIST(LogicalType::DOUBLE, means_values));
		}

		output_count++;
		state.current_input_row++;
	}

	output.SetCardinality(output_count);

	if (output_count > 0) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	} else {
		return OperatorResultType::NEED_MORE_INPUT;
	}
}

//===--------------------------------------------------------------------===//
// Registration - Dual Mode Support
//===--------------------------------------------------------------------===//

void WlsFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_wls (dual mode: literals + lateral joins)");

	// Register 3-argument overload: (y DOUBLE[], x DOUBLE[][], weights DOUBLE[])
	vector<LogicalType> args_3 = {
	    LogicalType::LIST(LogicalType::DOUBLE),                    // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x: DOUBLE[][]
	    LogicalType::LIST(LogicalType::DOUBLE)                     // weights: DOUBLE[]
	};

	TableFunction func_3("anofox_statistics_wls_fit", args_3, WlsFitExecute, WlsFitBind, nullptr, WlsFitInOutLocalInit);
	func_3.in_out_function = WlsFitInOut;
	loader.RegisterFunction(func_3);

	// Register 4-argument overload: (y DOUBLE[], x DOUBLE[][], weights DOUBLE[], options MAP/STRUCT)
	vector<LogicalType> args_4 = {
	    LogicalType::LIST(LogicalType::DOUBLE),                    // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // x: DOUBLE[][]
	    LogicalType::LIST(LogicalType::DOUBLE),                    // weights: DOUBLE[]
	    LogicalType::ANY                                           // options: MAP or STRUCT
	};

	TableFunction func_4("anofox_statistics_wls_fit", args_4, WlsFitExecute, WlsFitBind, nullptr, WlsFitInOutLocalInit);
	func_4.in_out_function = WlsFitInOut;
	loader.RegisterFunction(func_4);

	ANOFOX_DEBUG("anofox_statistics_wls registered successfully (both modes)");
}

} // namespace anofox_statistics
} // namespace duckdb
