#include "rls_fit_predict.hpp"
#include "fit_predict_base.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

// State for RLS fit-predict
struct RlsFitPredictState {
	PartitionDataCache cache; // Cached partition data (loaded once)
};

// Initialize state
static void RlsFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) RlsFitPredictState();
}

/**
 * Window callback for RLS fit-predict
 * Fits Recursive Least Squares regression and predicts for current row
 */
static void RlsFitPredictWindow(duckdb::AggregateInputData &aggr_input_data,
                                const duckdb::WindowPartitionInput &partition, duckdb::const_data_ptr_t g_state,
                                duckdb::data_ptr_t l_state, const duckdb::SubFrames &subframes, duckdb::Vector &result,
                                duckdb::idx_t rid) {

	// Access result validity
	auto &result_validity = FlatVector::Validity(result);

	// Access global state (cached partition data)
	auto &state = *reinterpret_cast<RlsFitPredictState *>(const_cast<data_ptr_t>(g_state));

	// Load partition data once (cached for all rows)
	LoadPartitionData(partition, state.cache);

	// Use cached data
	auto &all_y = state.cache.all_y;
	auto &all_x = state.cache.all_x;
	auto &options = state.cache.options;
	idx_t n_features = state.cache.n_features;

	// Compute frame signature (training row indices)
	vector<idx_t> frame_indices = ComputeFrameSignature(subframes, all_y, all_x, options);

	// Collect training data from frame signature
	vector<double> train_y;
	vector<vector<double>> train_x;

	for (idx_t idx : frame_indices) {
		train_y.push_back(all_y[idx]);
		train_x.push_back(all_x[idx]);
	}

	idx_t n_train = train_y.size();
	idx_t p = n_features;

	// Minimum observations:
	// - With intercept and p <= 1: n >= 1 (allows intercept-only models)
	// - With intercept and p >= 2: n >= p + 1 (need degrees of freedom)
	// - Without intercept: n >= p
	idx_t min_required = options.intercept ? (p <= 1 ? 1 : p + 1) : p;
	if (n_train < min_required || p == 0) {
		result_validity.SetInvalid(rid);
		return;
	}

	// Result is a STRUCT with fields: yhat, yhat_lower, yhat_upper, std_error, _dummy
	// NOTE: Must access struct AFTER validation checks to avoid assertion failures
	auto &struct_entries = StructVector::GetEntries(result);

	// Access DOUBLE fields
	auto yhat_data = FlatVector::GetData<double>(*struct_entries[0]);
	auto yhat_lower_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto yhat_upper_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto std_error_data = FlatVector::GetData<double>(*struct_entries[3]);

	// Trigger struct initialization via dummy LIST field (DuckDB limitation workaround)
	auto &dummy_list = *struct_entries[4];
	ListVector::Reserve(dummy_list, rid + 1);
	auto list_entries = FlatVector::GetData<list_entry_t>(dummy_list);
	list_entries[rid] = list_entry_t {0, 0}; // Empty list for this row

	// Validate forgetting factor
	if (options.forgetting_factor <= 0.0 || options.forgetting_factor > 1.0) {
		result_validity.SetInvalid(rid);
		return;
	}

	// Build training design matrix
	Eigen::MatrixXd X_train(n_train, p);
	Eigen::VectorXd y_train(n_train);

	for (idx_t row = 0; row < n_train; row++) {
		y_train(row) = train_y[row];
		for (idx_t col = 0; col < p; col++) {
			X_train(row, col) = train_x[row][col];
		}
	}

	// Fit RLS model using sequential Kalman updates
	double intercept = 0.0;
	Eigen::VectorXd beta;
	Eigen::VectorXd x_means;
	idx_t rank;

	// Special case: n=1 with intercept → intercept-only model
	// With a single observation, features provide no information
	if (options.intercept && n_train == 1) {
		intercept = y_train(0);
		beta = Eigen::VectorXd::Zero(p);
		x_means = X_train.row(0); // Single observation
		rank = 1;                 // Only intercept estimated
	} else {
		// For intercept, augment X with column of ones
		Eigen::MatrixXd X_work;
		idx_t p_work;

		if (options.intercept) {
			// Augment: X_work = [1, X] where 1 is column of ones
			p_work = p + 1;
			X_work = Eigen::MatrixXd(n_train, p_work);
			X_work.col(0) = Eigen::VectorXd::Ones(n_train); // Intercept column
			for (idx_t j = 0; j < p; j++) {
				X_work.col(j + 1) = X_train.col(j);
			}
		} else {
			// No intercept: use X as-is
			p_work = p;
			X_work = X_train;
		}

		// Initialize RLS state
		// β_0 = 0 (start with zero coefficients)
		// P_0 = large_value * I (large uncertainty initially)
		Eigen::VectorXd beta_work = Eigen::VectorXd::Zero(p_work);
		double initial_p = 1000.0; // Large initial uncertainty
		Eigen::MatrixXd P = Eigen::MatrixXd::Identity(p_work, p_work) * initial_p;

		// Sequential RLS updates for each observation
		for (idx_t t = 0; t < n_train; t++) {
			// Get current observation x_t (p_work x 1 vector)
			Eigen::VectorXd x_t = X_work.row(t).transpose();
			double y_t = y_train(t);

			// Prediction: ŷ_t = x_t' β_{t-1}
			double y_pred = x_t.dot(beta_work);

			// Prediction error: e_t = y_t - ŷ_t
			double error = y_t - y_pred;

			// Compute denominator: λ + x_t' P_{t-1} x_t
			double denominator = options.forgetting_factor + x_t.dot(P * x_t);

			// Kalman gain: K_t = P_{t-1} x_t / (λ + x_t' P_{t-1} x_t)
			Eigen::VectorXd K = P * x_t / denominator;

			// Update coefficients: β_t = β_{t-1} + K_t * e_t
			beta_work = beta_work + K * error;

			// Update covariance: P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})
			P = (P - K * x_t.transpose() * P) / options.forgetting_factor;
		}

		// Extract intercept and coefficients
		if (options.intercept) {
			intercept = beta_work(0);
			beta = beta_work.tail(p);
			x_means = X_train.colwise().mean();
		} else {
			intercept = 0.0;
			beta = beta_work;
			x_means = Eigen::VectorXd::Zero(p);
		}

		rank = p_work; // RLS typically maintains full rank
	}

	// Compute MSE
	Eigen::VectorXd y_pred_train = Eigen::VectorXd::Constant(n_train, intercept);
	y_pred_train += X_train * beta;

	Eigen::VectorXd residuals = y_train - y_pred_train;
	double ss_res = residuals.squaredNorm();

	// df_model includes intercept if present
	// Note: rank is for X_work (which includes intercept column if intercept=true)
	// So rank already includes intercept, but we need to check
	// Actually, for RLS, rank = p_work which includes intercept column, so it already includes intercept
	idx_t df_model = rank; // rank already includes intercept if present
	idx_t df_residual = n_train - df_model;
	double mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

	// Predict for current row
	if (rid >= all_x.size() || all_x[rid].empty()) {
		result_validity.SetInvalid(rid);
		return;
	}

	// Compute prediction with interval
	PredictionResult pred = ComputePredictionWithInterval(all_x[rid], intercept, beta, mse, x_means, X_train,
	                                                      df_residual, 0.95, "prediction");

	if (!pred.is_valid) {
		result_validity.SetInvalid(rid);
		return;
	}

	yhat_data[rid] = pred.yhat;
	yhat_lower_data[rid] = pred.yhat_lower;
	yhat_upper_data[rid] = pred.yhat_upper;
	std_error_data[rid] = pred.std_error;

	ANOFOX_DEBUG("RLS fit-predict: n_train=" << n_train << ", forgetting_factor=" << options.forgetting_factor
	                                         << ", yhat=" << pred.yhat);
}

void RlsFitPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering RLS fit-predict function");

	// Define struct fields inline
	// NOTE: Includes dummy LIST field to trigger proper DuckDB struct initialization in window aggregates
	child_list_t<LogicalType> fit_predict_struct_fields;
	fit_predict_struct_fields.push_back(make_pair("yhat", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("_dummy", LogicalType::LIST(LogicalType::DOUBLE)));

	// Use ONLY window callback to force WindowCustomAggregator path
	// This prevents WindowConstantAggregator from broadcasting a single result to all rows
	AggregateFunction anofox_stats_rls_fit_predict(
	    "anofox_stats_rls_fit_predict",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<RlsFitPredictState>,
	    RlsFitPredictInitialize,
	    nullptr, // No update - force WindowCustomAggregator
	    nullptr, // No combine - force WindowCustomAggregator
	    nullptr, // No finalize - force WindowCustomAggregator
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr, nullptr,
	    RlsFitPredictWindow, // RLS-specific window callback - called per row
	    nullptr, nullptr);

	loader.RegisterFunction(anofox_stats_rls_fit_predict);

	// Register alias
	AggregateFunction rls_fit_predict(
	    "rls_fit_predict",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<RlsFitPredictState>,
	    RlsFitPredictInitialize,
	    nullptr, // No update - force WindowCustomAggregator
	    nullptr, // No combine - force WindowCustomAggregator
	    nullptr, // No finalize - force WindowCustomAggregator
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr, nullptr,
	    RlsFitPredictWindow, // RLS-specific window callback - called per row
	    nullptr, nullptr);

	loader.RegisterFunction(rls_fit_predict);

	ANOFOX_DEBUG("RLS fit-predict function registered successfully (including alias rls_fit_predict)");
}

} // namespace anofox_statistics
} // namespace duckdb
