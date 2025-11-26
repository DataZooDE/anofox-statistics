#include "ridge_fit_predict.hpp"
#include "fit_predict_base.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>
#include <fstream>

namespace duckdb {
namespace anofox_statistics {

// DEBUG helper
#define RIDGE_DEBUG(msg)                                                                                               \
	do {                                                                                                               \
		std::ofstream debug("/tmp/ridge_debug.txt", std::ios::app);                                                    \
		debug << msg << std::endl;                                                                                     \
		debug.close();                                                                                                 \
	} while (0)

// State for Ridge fit-predict
struct RidgeFitPredictState {
	PartitionDataCache cache; // Cached partition data (loaded once)
};

// Initialize state
void RidgeFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) RidgeFitPredictState();
}

// Update (no-op for window-only function)
void RidgeFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
	// No-op: window callback reads partition directly
}

// Combine (no-op for window-only function)
void RidgeFitPredictCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	// No-op
}

// Finalize (returns NULL for non-window mode)
void RidgeFitPredictFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	// Return NULL for non-window mode (user must use OVER clause)
	auto &result_validity = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		result_validity.SetInvalid(i);
	}
}

/**
 * Window callback for Ridge fit-predict
 * Fits Ridge regression and predicts for current row
 */
static void RidgeFitPredictWindow(duckdb::AggregateInputData &aggr_input_data,
                                  const duckdb::WindowPartitionInput &partition, duckdb::const_data_ptr_t g_state,
                                  duckdb::data_ptr_t l_state, const duckdb::SubFrames &subframes,
                                  duckdb::Vector &result, duckdb::idx_t rid) {

	ANOFOX_FIT_PREDICT_DEBUG("RidgeFitPredictWindow called for row " << rid);

	// DEBUG: Print for first few rows
	if (rid < 3) {
		RIDGE_DEBUG("[START] Window callback for row " << rid);
	}

	// Access result validity
	auto &result_validity = FlatVector::Validity(result);

	// Access global state (cached partition data)
	auto &state = *reinterpret_cast<RidgeFitPredictState *>(const_cast<data_ptr_t>(g_state));

	// Load partition data once (cached for all rows)
	LoadPartitionData(partition, state.cache);

	// Use cached data
	auto &all_y = state.cache.all_y;
	auto &all_x = state.cache.all_x;
	auto &options = state.cache.options;
	idx_t n_features = state.cache.n_features;

	ANOFOX_FIT_PREDICT_DEBUG("Partition data loaded: n_rows=" << all_y.size() << ", n_features=" << n_features
	                                                          << ", lambda=" << options.lambda
	                                                          << ", intercept=" << options.intercept);

	// DEBUG
	if (rid < 3) {
	}

	// Build frame signature for caching (detect if frame changes across rows)
	vector<idx_t> frame_indices = ComputeFrameSignature(subframes, all_y, all_x, options);

	// Check if model is cached and frame matches
	bool use_cache = false;
	if (state.cache.model_cache && state.cache.model_cache->initialized) {
		if (state.cache.model_cache->train_indices == frame_indices) {
			use_cache = true;
		}
	}

	// Get model parameters (either from cache or by fitting)
	double intercept;
	Eigen::VectorXd beta;
	Eigen::VectorXd x_means;
	Eigen::MatrixXd XtX_inv;
	idx_t n_train;
	idx_t df_residual;
	double mse;
	idx_t p = n_features;

	if (use_cache) {
		// Use cached model - avoids refitting for every row!
		intercept = state.cache.model_cache->intercept;
		beta = state.cache.model_cache->coefficients;
		x_means = state.cache.model_cache->x_means;
		XtX_inv = state.cache.model_cache->XtX_inv;
		n_train = state.cache.model_cache->n_train;
		df_residual = state.cache.model_cache->df_residual;
		mse = state.cache.model_cache->mse;

		if (rid < 3) {
			RIDGE_DEBUG("[CACHE] Using cached model for row " << rid << ": intercept=" << intercept
			                                                  << ", beta.size()=" << beta.size()
			                                                  << ", beta[0]=" << (beta.size() > 0 ? beta(0) : 0.0));
		}
	} else {
		// Fit new model
		n_train = frame_indices.size();

		if (rid < 3) {
			RIDGE_DEBUG("[FIT] Fitting new model: n_train=" << n_train << ", p=" << p);
		}

		// Validation:
		// - With intercept and p<=1: can fit with n=1 (intercept-only or simple regression)
		// - With intercept and p>=2: need n >= p+1 to ensure df_residual >= 1
		// - Without intercept: need n >= p
		idx_t min_required = options.intercept ? (p <= 1 ? 1 : p + 1) : p;
		if (n_train < min_required || p == 0) {
			result_validity.SetInvalid(rid);
			return;
		}

		// Build training matrices from frame indices
		Eigen::MatrixXd X_train(n_train, p);
		Eigen::VectorXd y_train(n_train);

		for (idx_t i = 0; i < n_train; i++) {
			idx_t data_idx = frame_indices[i];
			y_train(i) = all_y[data_idx];
			for (idx_t col = 0; col < p; col++) {
				X_train(i, col) = all_x[data_idx][col];
			}
		}

		// Fit Ridge regression
		idx_t rank;
		if (options.intercept) {
			double y_mean = y_train.mean();
			x_means = X_train.colwise().mean();

			// Special case: n=1 with intercept → X_centered is all zeros (degenerate)
			// Solution: intercept = y[0], beta = 0 (intercept-only model)
			if (n_train == 1) {
				beta = Eigen::VectorXd::Zero(p);
				rank = 0; // Rank of centered X (all zeros) is 0
				intercept = y_mean;
				XtX_inv = Eigen::MatrixXd::Zero(p, p);
			} else {
				Eigen::VectorXd y_centered = y_train.array() - y_mean;
				Eigen::MatrixXd X_centered = X_train;
				for (idx_t j = 0; j < p; j++) {
					X_centered.col(j).array() -= x_means(j);
				}

				Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;
				Eigen::VectorXd Xty = X_centered.transpose() * y_centered;
				Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
				Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

				Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
				beta = qr.solve(Xty);
				rank = qr.rank();

				intercept = y_mean - beta.dot(x_means);

				// Compute XtX_inv for prediction intervals
				Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
				XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));
			}
		} else {
			x_means = Eigen::VectorXd::Zero(p);

			Eigen::MatrixXd XtX = X_train.transpose() * X_train;
			Eigen::VectorXd Xty = X_train.transpose() * y_train;
			Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
			Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

			Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
			beta = qr.solve(Xty);
			rank = qr.rank();

			intercept = 0.0;

			Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
			XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));
		}

		// Compute MSE
		Eigen::VectorXd y_pred_train = Eigen::VectorXd::Constant(n_train, intercept);
		y_pred_train += X_train * beta;
		Eigen::VectorXd residuals = y_train - y_pred_train;
		double ss_res = residuals.squaredNorm();

		// NOTE: rank is from QR of centered X (p features), so add 1 for intercept
		idx_t df_model = rank + (options.intercept ? 1 : 0);
		df_residual = n_train - df_model;
		mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

		// Cache the model for reuse across rows
		if (!state.cache.model_cache) {
			state.cache.model_cache = new OlsModelCache();
		}
		state.cache.model_cache->coefficients = beta;
		state.cache.model_cache->intercept = intercept;
		state.cache.model_cache->mse = mse;
		state.cache.model_cache->XtX_inv = XtX_inv;
		state.cache.model_cache->x_means = x_means;
		state.cache.model_cache->df_residual = df_residual;
		state.cache.model_cache->rank = rank;
		state.cache.model_cache->n_train = n_train;
		state.cache.model_cache->train_indices = frame_indices;
		state.cache.model_cache->initialized = true;

		if (rid < 3) {
			RIDGE_DEBUG("[FIT] Model fitted and cached: intercept=" << intercept
			                                                        << ", beta[0]=" << (p > 0 ? beta(0) : 0.0)
			                                                        << ", mse=" << mse << ", n_train=" << n_train);
		}
	}

	// Predict for current row
	if (rid >= all_x.size() || all_x[rid].empty()) {
		if (rid < 3) {
			RIDGE_DEBUG("[INVALID] Row " << rid << " invalid: rid=" << rid << ", all_x.size()=" << all_x.size()
			                             << ", empty=" << (rid < all_x.size() ? all_x[rid].empty() : true));
		}
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

	// DEBUG: Show input data for prediction
	if (rid < 3) {
		RIDGE_DEBUG("[PREDICT] Predicting for row " << rid << ": x.size()=" << all_x[rid].size()
		                                            << ", x[0]=" << (all_x[rid].size() > 0 ? all_x[rid][0] : 0.0)
		                                            << ", intercept=" << intercept << ", beta.size()=" << beta.size());
	}

	// Compute prediction with interval (using precomputed XtX_inv - memory efficient!)
	// NOTE: Ridge is biased, so intervals are approximate
	PredictionResult pred = ComputePredictionWithIntervalXtXInv(all_x[rid], intercept, beta, mse, x_means, XtX_inv,
	                                                            n_train, df_residual, 0.95, "prediction");

	if (!pred.is_valid) {
		if (rid < 3) {
			RIDGE_DEBUG("[INVALID_PRED] Prediction INVALID for row " << rid);
		}
		ANOFOX_FIT_PREDICT_DEBUG("Prediction invalid: returning NULL");
		result_validity.SetInvalid(rid);
		return;
	}

	ANOFOX_FIT_PREDICT_DEBUG("Prediction computed: yhat=" << pred.yhat << ", lower=" << pred.yhat_lower << ", upper="
	                                                      << pred.yhat_upper << ", std_error=" << pred.std_error);

	// DEBUG
	if (rid < 3) {
		RIDGE_DEBUG("[SUCCESS] Prediction SUCCESS for row "
		            << rid << ": yhat=" << pred.yhat << ", lower=" << pred.yhat_lower << ", upper=" << pred.yhat_upper);
	}

	yhat_data[rid] = pred.yhat;
	yhat_lower_data[rid] = pred.yhat_lower;
	yhat_upper_data[rid] = pred.yhat_upper;
	std_error_data[rid] = pred.std_error;

	ANOFOX_FIT_PREDICT_DEBUG("Results assigned to row " << rid << ": yhat_data[" << rid << "]=" << yhat_data[rid]);

	ANOFOX_DEBUG("Ridge fit-predict: n_train=" << n_train << ", lambda=" << options.lambda << ", yhat=" << pred.yhat);
}

void RidgeFitPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering Ridge fit-predict function");

	// Define struct fields inline
	// NOTE: Includes dummy LIST field to trigger proper DuckDB struct initialization in window aggregates
	child_list_t<LogicalType> fit_predict_struct_fields;
	fit_predict_struct_fields.push_back(make_pair("yhat", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("_dummy", LogicalType::LIST(LogicalType::DOUBLE)));

	// Use real callback functions (not lambdas)
	AggregateFunction anofox_statistics_ridge_fit_predict(
	    "anofox_statistics_ridge_fit_predict",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<RidgeFitPredictState>,
	    RidgeFitPredictInitialize, RidgeFitPredictUpdate, RidgeFitPredictCombine, RidgeFitPredictFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr,
	    nullptr, // destroy - use nullptr like the working function
	    nullptr,
	    RidgeFitPredictWindow, // Ridge-specific window callback
	    nullptr, nullptr);

	loader.RegisterFunction(anofox_statistics_ridge_fit_predict);

	ANOFOX_DEBUG("Ridge fit-predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
