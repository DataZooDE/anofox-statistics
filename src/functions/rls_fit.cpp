#include "rls_fit.hpp"
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
 * Recursive Least Squares (RLS) with MAP-based options
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_rls(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       options := MAP{'intercept': true, 'forgetting_factor': 1.0}
 *   )
 *
 * Online learning algorithm that updates coefficients sequentially:
 *
 * β_t = β_{t-1} + K_t * (y_t - x_t'β_{t-1})
 * K_t = P_{t-1}x_t / (λ + x_t'P_{t-1}x_t)
 * P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})
 *
 * Where:
 * - λ is the forgetting factor (0 < λ ≤ 1)
 * - P_t is the inverse covariance matrix
 * - K_t is the Kalman gain vector
 */

struct RlsFitBindData : public FunctionData {
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

	// Extended metadata (when full_output=true)
	vector<double> coefficient_std_errors;
	double intercept_std_error = std::numeric_limits<double>::quiet_NaN();
	idx_t df_residual = 0;
	vector<double> x_train_means;

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<RlsFitBindData>();
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
		result->coefficient_std_errors = coefficient_std_errors;
		result->intercept_std_error = intercept_std_error;
		result->df_residual = df_residual;
		result->x_train_means = x_train_means;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * RLS implementation: Sequential update of coefficients
 */
static void ComputeRLS(RlsFitBindData &data) {
	idx_t n = data.y_values.size();
	idx_t p = data.x_values.size();

	if (n == 0 || p == 0) {
		throw InvalidInputException("Cannot fit RLS with empty data");
	}

	if (n < p + 1) {
		throw InvalidInputException("Insufficient observations: need at least %d observations for %d features, got %d",
		                            p + 1, p, n);
	}

	if (data.options.forgetting_factor <= 0.0 || data.options.forgetting_factor > 1.0) {
		throw InvalidInputException("Forgetting factor must be in range (0, 1], got %f",
		                            data.options.forgetting_factor);
	}

	data.n_obs = n;
	data.n_features = p;

	ANOFOX_DEBUG("Computing RLS with " << n << " observations and " << p
	                                   << " features, forgetting_factor = " << data.options.forgetting_factor);

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

	// If options.intercept is true, augment X with column of 1s for intercept
	// This is the standard way to handle intercepts in online RLS
	Eigen::MatrixXd X_work;
	idx_t p_work;               // Number of columns in working matrix
	idx_t intercept_offset = 0; // Offset for feature indices

	if (data.options.intercept) {
		// Augment: X_work = [1, X] where 1 is column of ones
		p_work = p + 1;
		X_work = Eigen::MatrixXd(n, p_work);
		X_work.col(0) = Eigen::VectorXd::Ones(n); // Intercept column
		for (idx_t j = 0; j < p; j++) {
			X_work.col(j + 1) = X.col(j);
		}
		intercept_offset = 1;
		ANOFOX_DEBUG("RLS with intercept: augmented design matrix to " << p_work << " columns");
	} else {
		// No intercept: use X as-is
		p_work = p;
		X_work = X;
		ANOFOX_DEBUG("RLS without intercept: using " << p_work << " columns");
	}

	// Detect constant columns (these will be aliased)
	auto constant_cols_work = RankDeficientOls::DetectConstantColumns(X_work);

	// If we have an intercept column, it should never be marked as constant
	// (it's a column of 1s, which is constant, but we always want to keep it)
	if (data.options.intercept) {
		constant_cols_work[0] = false; // Never alias the intercept
	}

	// Build list of non-constant column indices in X_work
	vector<idx_t> valid_indices_work;
	for (idx_t j = 0; j < p_work; j++) {
		if (!constant_cols_work[j]) {
			valid_indices_work.push_back(j);
		}
	}

	idx_t p_valid = valid_indices_work.size();
	if (p_valid == 0) {
		throw InvalidInputException("All features are constant - cannot fit RLS");
	}

	// Create reduced X matrix with only non-constant columns
	Eigen::MatrixXd X_valid(n, p_valid);
	for (idx_t j_new = 0; j_new < p_valid; j_new++) {
		X_valid.col(j_new) = X_work.col(valid_indices_work[j_new]);
	}

	ANOFOX_DEBUG("RLS using " << p_valid << " non-constant features out of " << p_work
	                          << " total (including intercept if requested)");

	// Initialize RLS state (for reduced matrix)
	// β_0 = 0 (start with zero coefficients for valid features)
	// P_0 = large_value * I (large uncertainty initially)
	Eigen::VectorXd beta_valid = Eigen::VectorXd::Zero(p_valid);
	double initial_p = 1000.0; // Large initial uncertainty
	Eigen::MatrixXd P = Eigen::MatrixXd::Identity(p_valid, p_valid) * initial_p;

	ANOFOX_DEBUG("RLS initialization: beta = [0, ...], P_0 = " << initial_p << " * I");

	// Sequential RLS updates for each observation
	for (idx_t t = 0; t < n; t++) {
		// Get current observation x_t (p_valid x 1 vector) from reduced matrix
		Eigen::VectorXd x_t = X_valid.row(t).transpose();
		double y_t = y(t);

		// Prediction: ŷ_t = x_t' β_{t-1}
		double y_pred = x_t.dot(beta_valid);

		// Prediction error: e_t = y_t - ŷ_t
		double error = y_t - y_pred;

		// Compute denominator: λ + x_t' P_{t-1} x_t
		double denominator = data.options.forgetting_factor + x_t.dot(P * x_t);

		// Kalman gain: K_t = P_{t-1} x_t / (λ + x_t' P_{t-1} x_t)
		Eigen::VectorXd K = P * x_t / denominator;

		// Update coefficients: β_t = β_{t-1} + K_t * e_t
		beta_valid = beta_valid + K * error;

		// Update covariance: P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})
		P = (P - K * x_t.transpose() * P) / data.options.forgetting_factor;

		ANOFOX_DEBUG("RLS step " << t << ": error = " << error
		                         << ", beta_valid[0] = " << (p_valid > 0 ? beta_valid(0) : 0.0));
	}

	// Map coefficients back to full feature space (with NaN for constant columns)
	data.coefficients.resize(p);
	data.is_aliased.resize(p);

	if (data.options.intercept) {
		// With intercept: extract from augmented system
		// valid_indices_work contains indices in X_work [0=intercept, 1=x1, 2=x2, ...]

		// Find intercept in valid set (should be index 0 in X_work)
		data.intercept = 0.0;
		bool found_intercept = false;
		for (idx_t k = 0; k < valid_indices_work.size(); k++) {
			if (valid_indices_work[k] == 0) {
				data.intercept = beta_valid(k);
				found_intercept = true;
				break;
			}
		}
		if (!found_intercept) {
			throw InternalException("RLS: Intercept not found in valid indices (this should never happen)");
		}

		// Extract feature coefficients
		// For feature j in original X, its column in X_work is j+1
		for (idx_t j = 0; j < p; j++) {
			idx_t j_work = j + 1; // Column index in X_work

			if (constant_cols_work[j_work]) {
				// This feature is constant (aliased)
				data.coefficients[j] = std::numeric_limits<double>::quiet_NaN();
				data.is_aliased[j] = true;
			} else {
				// Find j_work in valid_indices_work
				idx_t j_valid = 0;
				bool found = false;
				for (idx_t k = 0; k < valid_indices_work.size(); k++) {
					if (valid_indices_work[k] == j_work) {
						j_valid = k;
						found = true;
						break;
					}
				}
				if (!found) {
					throw InternalException("RLS: Feature index not found in valid indices");
				}
				data.coefficients[j] = beta_valid(j_valid);
				data.is_aliased[j] = false;
			}
		}

		// Rank is number of valid features INCLUDING intercept, minus 1 for intercept
		data.rank = p_valid - 1; // Don't count intercept in rank

	} else {
		// Without intercept: map coefficients directly (original logic)
		data.intercept = 0.0;

		for (idx_t j = 0; j < p; j++) {
			if (constant_cols_work[j]) {
				data.coefficients[j] = std::numeric_limits<double>::quiet_NaN();
				data.is_aliased[j] = true;
			} else {
				// Find index in valid set
				idx_t j_valid = 0;
				for (idx_t k = 0; k < valid_indices_work.size(); k++) {
					if (valid_indices_work[k] == j) {
						j_valid = k;
						break;
					}
				}
				data.coefficients[j] = beta_valid(j_valid);
				data.is_aliased[j] = false;
			}
		}

		data.rank = p_valid;
	}

	// Compute final predictions using only non-aliased features
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
	for (idx_t j = 0; j < p; j++) {
		if (!data.is_aliased[j]) {
			y_pred += data.coefficients[j] * X.col(j);
		}
	}
	if (data.options.intercept) {
		y_pred.array() += data.intercept;
	}

	// Compute residuals and statistics
	Eigen::VectorXd residuals = y - y_pred;
	double ss_res = residuals.squaredNorm();
	double y_mean = y.mean();
	double ss_tot = (y.array() - y_mean).square().sum();

	// R²
	data.r_squared = (ss_tot > 0) ? 1.0 - (ss_res / ss_tot) : 0.0;

	// Adjusted R² using effective rank
	if (n > data.rank + 1) {
		data.adj_r_squared =
		    1.0 - ((1.0 - data.r_squared) * static_cast<double>(n - 1) / static_cast<double>(n - data.rank - 1));
	} else {
		data.adj_r_squared = data.r_squared;
	}

	// MSE and RMSE
	data.mse = ss_res / static_cast<double>(n);
	data.rmse = std::sqrt(data.mse);

	// Compute extended metadata if full_output=true
	// Note: RLS standard errors are complex due to time-varying nature
	// We provide approximate statistics for the final model state
	if (data.options.full_output) {
		data.df_residual = n > data.rank ? (n - data.rank) : 0;

		// Compute x_train_means from original X matrix (not X_work)
		Eigen::VectorXd x_means = Eigen::VectorXd::Zero(p);
		for (idx_t j = 0; j < p; j++) {
			x_means(j) = X.col(j).mean();
		}

		// Approximate standard errors from final covariance matrix
		// SE ≈ sqrt(MSE * diag(P))
		// Need to map from original feature index to valid index in P
		data.coefficient_std_errors.resize(p);
		for (idx_t j = 0; j < p; j++) {
			if (data.is_aliased[j]) {
				data.coefficient_std_errors[j] = std::numeric_limits<double>::quiet_NaN();
			} else {
				// Find j_work in valid_indices_work (same mapping as for coefficients)
				idx_t j_work = data.options.intercept ? (j + 1) : j;
				idx_t j_valid = 0;
				bool found = false;
				for (idx_t k = 0; k < valid_indices_work.size(); k++) {
					if (valid_indices_work[k] == j_work) {
						j_valid = k;
						found = true;
						break;
					}
				}
				if (!found) {
					throw InternalException("RLS: Feature index not found in valid indices for SE computation");
				}

				// Approximate SE from covariance diagonal
				auto j_valid_idx = static_cast<Eigen::Index>(j_valid);
				double var_j = data.mse * P(j_valid_idx, j_valid_idx);
				data.coefficient_std_errors[j] = std::sqrt(std::max(0.0, var_j));
			}
		}

		// Store x_train_means
		data.x_train_means.resize(p);
		for (idx_t j = 0; j < p; j++) {
			data.x_train_means[j] = x_means(j);
		}

		// Approximate intercept SE
		if (data.options.intercept) {
			double intercept_variance = data.mse / static_cast<double>(n);
			for (idx_t j = 0; j < p; j++) {
				if (!data.is_aliased[j]) {
					double se_beta_j = data.coefficient_std_errors[j];
					intercept_variance += se_beta_j * se_beta_j * x_means(j) * x_means(j);
				}
			}
			data.intercept_std_error = std::sqrt(intercept_variance);
		} else {
			data.intercept_std_error = std::numeric_limits<double>::quiet_NaN();
		}
	}

	ANOFOX_DEBUG("RLS complete: R² = " << data.r_squared << ", MSE = " << data.mse << ", coefficients = ["
	                                   << data.coefficients[0] << (p > 1 ? ", ...]" : "]"));
}

//===--------------------------------------------------------------------===//
// Lateral Join Support Structures (must be declared before RlsFitBind)
//===--------------------------------------------------------------------===//

/**
 * Bind data for in-out mode (lateral join support)
 * Stores only options, not data (data comes from input rows)
 */
struct RlsFitInOutBindData : public FunctionData {
	RegressionOptions options;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<RlsFitInOutBindData>();
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
struct RlsFitInOutLocalState : public LocalTableFunctionState {
	idx_t current_input_row = 0;
};

static unique_ptr<LocalTableFunctionState>
RlsFitInOutLocalInit(ExecutionContext &context, TableFunctionInitInput &input, GlobalTableFunctionState *global_state) {
	return make_uniq<RlsFitInOutLocalState>();
}

//===--------------------------------------------------------------------===//
// Bind Function (supports both literal and lateral join modes)
//===--------------------------------------------------------------------===//

static unique_ptr<FunctionData> RlsFitBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("RLS regression bind phase");

	// Determine if full_output is requested
	bool full_output = false;
	if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
		try {
			if (input.inputs[2].type().id() == LogicalTypeId::MAP || input.inputs[2].type().id() == LogicalTypeId::STRUCT) {
				auto opts = RegressionOptions::ParseFromMap(input.inputs[2]);
				full_output = opts.full_output;
			}
		} catch (...) {
			// Ignore parsing errors
		}
	}

	// Set return schema (basic columns)
	names = {"coefficients", "intercept",         "r_squared", "adj_r_squared", "mse",
	         "rmse",         "forgetting_factor", "n_obs",     "n_features"};
	return_types = {
	    LogicalType::LIST(LogicalType::DOUBLE), // coefficients
	    LogicalType::DOUBLE,                    // intercept
	    LogicalType::DOUBLE,                    // r_squared
	    LogicalType::DOUBLE,                    // adj_r_squared
	    LogicalType::DOUBLE,                    // mse
	    LogicalType::DOUBLE,                    // rmse
	    LogicalType::DOUBLE,                    // forgetting_factor
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

		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // coefficient_std_errors
		return_types.push_back(LogicalType::DOUBLE);                    // intercept_std_error
		return_types.push_back(LogicalType::BIGINT);                    // df_residual
		return_types.push_back(LogicalType::LIST(LogicalType::BOOLEAN)); // is_aliased
		return_types.push_back(LogicalType::LIST(LogicalType::DOUBLE)); // x_train_means
	}

	// Check if this is being called for lateral joins (in-out function mode)
	// In that case, we don't have literal values to process
	if (input.inputs.size() >= 2 && !input.inputs[0].IsNull()) {
		// Check if the first input is actually a constant value
		try {
			auto y_list = ListValue::GetChildren(input.inputs[0]);
			// If we can get children, it's a literal value - process it normally

			auto result = make_uniq<RlsFitBindData>();

			// Extract y values
			for (const auto &val : y_list) {
				result->y_values.push_back(val.GetValue<double>());
			}

			idx_t n = result->y_values.size();
			ANOFOX_DEBUG("y has " << n << " observations");

			// Extract x values (second parameter - 2D array)
			auto x_outer = ListValue::GetChildren(input.inputs[1]);
			for (const auto &x_inner_val : x_outer) {
				auto x_feature_list = ListValue::GetChildren(x_inner_val);
				vector<double> x_feature;
				for (const auto &val : x_feature_list) {
					x_feature.push_back(val.GetValue<double>());
				}

				// Validate dimensions
				if (x_feature.size() != n) {
					throw InvalidInputException(
					    "Array dimensions mismatch: y has %d elements, feature %d has %d elements", n,
					    result->x_values.size() + 1, x_feature.size());
				}

				result->x_values.push_back(x_feature);
			}

			// Extract options (third parameter - MAP, optional)
			if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
				if (input.inputs[2].type().id() == LogicalTypeId::MAP) {
					result->options = RegressionOptions::ParseFromMap(input.inputs[2]);
					result->options.Validate();
				}
			}

			ANOFOX_INFO("Fitting RLS with " << n << " observations, " << result->x_values.size()
			                                << " features, forgetting_factor=" << result->options.forgetting_factor);

			// Perform RLS fitting
			ComputeRLS(*result);

			ANOFOX_INFO("RLS fit completed: R² = " << result->r_squared);

			return std::move(result);

		} catch (...) {
			// If we can't get children, it's probably a column reference (lateral join mode)
			// Return minimal bind data for in-out function mode
			auto result = make_uniq<RlsFitInOutBindData>();

			// Extract options if provided as literal
			if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
				try {
					if (input.inputs[2].type().id() == LogicalTypeId::MAP) {
						result->options = RegressionOptions::ParseFromMap(input.inputs[2]);
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
	auto result = make_uniq<RlsFitInOutBindData>();
	return std::move(result);
}

static void RlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {

	auto &bind_data = data.bind_data->CastNoConst<RlsFitBindData>();

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
	output.data[col_idx++].SetValue(0, Value(bind_data.options.forgetting_factor));
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
	}
}

/**
 * In-out function for lateral join support
 * Processes rows from input table, computes regression for each row
 */
static OperatorResultType RlsFitInOut(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                      DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RlsFitInOutBindData>();
	auto &state = data_p.local_state->Cast<RlsFitInOutLocalState>();

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

		// Check for NULL inputs
		if (FlatVector::IsNull(input.data[0], row_idx) || FlatVector::IsNull(input.data[1], row_idx)) {
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

		// Create temporary bind data for computation (reuse existing RlsFitBindData)
		RlsFitBindData temp_data;
		temp_data.y_values = y_values;
		temp_data.x_values = x_values;
		temp_data.options = bind_data.options;

		// Compute regression using existing logic
		try {
			ComputeRLS(temp_data);
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

		output.data[0].SetValue(output_count, Value::LIST(LogicalType::DOUBLE, coeffs_values));
		output.data[1].SetValue(output_count, Value(temp_data.intercept));
		output.data[2].SetValue(output_count, Value(temp_data.r_squared));
		output.data[3].SetValue(output_count, Value(temp_data.adj_r_squared));
		output.data[4].SetValue(output_count, Value(temp_data.mse));
		output.data[5].SetValue(output_count, Value(temp_data.rmse));
		output.data[6].SetValue(output_count, Value(temp_data.options.forgetting_factor));
		output.data[7].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.n_obs)));
		output.data[8].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.n_features)));

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

void RlsFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_rls (dual mode: literals + lateral joins)");

	// Register single function with BOTH literal and lateral join support
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                   // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)) // x: DOUBLE[][]
	};

	// Register with literal mode (bind + execute)
	TableFunction function("anofox_statistics_rls", arguments, RlsFitExecute, RlsFitBind, nullptr,
	                       RlsFitInOutLocalInit);

	// Add lateral join support (in_out_function)
	function.in_out_function = RlsFitInOut;
	function.varargs = LogicalType::ANY;

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_rls registered successfully (both modes)");
}

} // namespace anofox_statistics
} // namespace duckdb
