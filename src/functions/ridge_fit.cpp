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

	// Extended metadata (when full_output=true)
	vector<double> coefficient_std_errors;
	double intercept_std_error = std::numeric_limits<double>::quiet_NaN();
	idx_t df_residual = 0;
	vector<double> x_train_means;

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

	ANOFOX_DEBUG("Computing Ridge regression with " << n << " observations, " << p
	                                                << " features, λ=" << data.options.lambda);

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
		// Use FitWithStdErrors if full_output requested
		auto result = data.options.full_output ?
			RankDeficientOls::FitWithStdErrors(y, X) :
			RankDeficientOls::Fit(y, X);

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

		// Store extended metadata if full_output=true
		if (data.options.full_output && result.has_std_errors) {
			data.coefficient_std_errors.resize(p);
			for (idx_t i = 0; i < p; i++) {
				data.coefficient_std_errors[i] = result.std_errors[i];
			}
			data.df_residual = n > result.rank ? (n - result.rank) : 0;
		}

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

			// Store x_means and compute intercept SE if full_output=true
			if (data.options.full_output) {
				data.x_train_means.resize(p);
				for (idx_t j = 0; j < p; j++) {
					data.x_train_means[j] = x_means[j];
				}

				if (result.has_std_errors) {
					double intercept_variance = data.mse / static_cast<double>(n);
					for (idx_t j = 0; j < p; j++) {
						if (!result.is_aliased[j] && !std::isnan(result.std_errors[j])) {
							double se_beta_j = result.std_errors[j];
							intercept_variance += se_beta_j * se_beta_j * x_means[j] * x_means[j];
						}
					}
					data.intercept_std_error = std::sqrt(intercept_variance);
				}
			}
		} else {
			data.intercept = 0.0;
			data.intercept_std_error = std::numeric_limits<double>::quiet_NaN();
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
		data.adj_r_squared = 1.0 - ((1.0 - data.r_squared) * (static_cast<double>(n) - 1.0) /
		                            (static_cast<double>(n) - static_cast<double>(effective_params) - 1.0));
	} else {
		data.adj_r_squared = data.r_squared;
	}

	data.mse = ss_res / static_cast<double>(n);
	data.rmse = std::sqrt(data.mse);

	// Compute extended metadata if full_output=true
	if (data.options.full_output) {
		data.df_residual = n > data.rank ? (n - data.rank) : 0;

		// Store x_train_means
		if (data.options.intercept) {
			data.x_train_means.resize(p);
			for (idx_t j = 0; j < p; j++) {
				data.x_train_means[j] = x_means[j];
			}
		}

		// Compute approximate standard errors for Ridge
		// Note: Ridge SE formula is: sqrt(MSE * diag((X'X + λI)^-1 * X'X * (X'X + λI)^-1))
		// For simplicity, we use: sqrt(MSE * diag((X'X + λI)^-1))
		try {
			Eigen::MatrixXd XtX_reg_inv = XtX_regularized.inverse();
			data.coefficient_std_errors.resize(p);
			for (idx_t j = 0; j < p; j++) {
				if (constant_features[j]) {
					data.coefficient_std_errors[j] = std::numeric_limits<double>::quiet_NaN();
				} else {
					double var_j = data.mse * XtX_reg_inv(j, j);
					data.coefficient_std_errors[j] = std::sqrt(std::max(0.0, var_j));
				}
			}

			// Compute intercept standard error if needed
			if (data.options.intercept) {
				double intercept_variance = data.mse / static_cast<double>(n);
				for (idx_t j = 0; j < p; j++) {
					if (!constant_features[j]) {
						double se_beta_j = data.coefficient_std_errors[j];
						intercept_variance += se_beta_j * se_beta_j * x_means[j] * x_means[j];
					}
				}
				data.intercept_std_error = std::sqrt(intercept_variance);
			} else {
				data.intercept_std_error = std::numeric_limits<double>::quiet_NaN();
			}
		} catch (...) {
			// If inversion fails, leave standard errors as empty/NaN
			data.coefficient_std_errors.clear();
			data.intercept_std_error = std::numeric_limits<double>::quiet_NaN();
		}
	}

	ANOFOX_DEBUG("Ridge complete: R² = " << data.r_squared << ", λ=" << data.options.lambda << ", rank = " << data.rank
	                                     << "/" << p);
}

//===--------------------------------------------------------------------===//
// Lateral Join Support Structures (must be declared before RidgeFitBind)
//===--------------------------------------------------------------------===//

/**
 * Bind data for in-out mode (lateral join support)
 * Stores only options, not data (data comes from input rows)
 */
struct RidgeFitInOutBindData : public FunctionData {
	RegressionOptions options;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<RidgeFitInOutBindData>();
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
struct RidgeFitInOutLocalState : public LocalTableFunctionState {
	idx_t current_input_row = 0;
};

static unique_ptr<LocalTableFunctionState> RidgeFitInOutLocalInit(ExecutionContext &context,
                                                                  TableFunctionInitInput &input,
                                                                  GlobalTableFunctionState *global_state) {
	return make_uniq<RidgeFitInOutLocalState>();
}

//===--------------------------------------------------------------------===//
// Bind Function (supports both literal and lateral join modes)
//===--------------------------------------------------------------------===//

static unique_ptr<FunctionData> RidgeFitBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {

	ANOFOX_INFO("Ridge regression bind phase");

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

			auto result = make_uniq<RidgeFitBindData>();

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

			ANOFOX_INFO("Fitting Ridge with " << n << " observations, " << result->x_values.size()
			                                  << " features, λ=" << result->options.lambda);

			// Perform Ridge fitting
			ComputeRidge(*result);

			ANOFOX_INFO("Ridge fit completed: R² = " << result->r_squared);

			return std::move(result);

		} catch (...) {
			// If we can't get children, it's probably a column reference (lateral join mode)
			// Return minimal bind data for in-out function mode
			auto result = make_uniq<RidgeFitInOutBindData>();

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
	auto result = make_uniq<RidgeFitInOutBindData>();
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
	output.data[col_idx++].SetValue(0, Value(bind_data.options.lambda));

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
static OperatorResultType RidgeFitInOut(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                        DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RidgeFitInOutBindData>();
	auto &state = data_p.local_state->Cast<RidgeFitInOutLocalState>();

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

		// Create temporary bind data for computation (reuse existing RidgeFitBindData)
		RidgeFitBindData temp_data;
		temp_data.y_values = y_values;
		temp_data.x_values = x_values;
		temp_data.options = bind_data.options;

		// Compute regression using existing logic
		try {
			ComputeRidge(temp_data);
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
		output.data[6].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.n_obs)));
		output.data[7].SetValue(output_count, Value::BIGINT(static_cast<int64_t>(temp_data.n_features)));
		output.data[8].SetValue(output_count, Value(temp_data.options.lambda));

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

void RidgeFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_ridge (dual mode: literals + lateral joins)");

	// Register single function with BOTH literal and lateral join support
	vector<LogicalType> arguments = {
	    LogicalType::LIST(LogicalType::DOUBLE),                   // y: DOUBLE[]
	    LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)) // x: DOUBLE[][]
	};

	// Register with literal mode (bind + execute)
	TableFunction function("anofox_statistics_ridge", arguments, RidgeFitExecute, RidgeFitBind, nullptr,
	                       RidgeFitInOutLocalInit);

	// Add lateral join support (in_out_function)
	function.in_out_function = RidgeFitInOut;
	function.varargs = LogicalType::ANY;

	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_ridge registered successfully (both modes)");
}

} // namespace anofox_statistics
} // namespace duckdb
