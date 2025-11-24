#include "ols_fit_predict.hpp"
#include "../bridge/libanostat_wrapper.hpp"
#include "../bridge/type_converters.hpp"
#include "../utils/options_parser.hpp"
#include "fit_predict_base.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

// State for OLS fit-predict
struct OlsFitPredictState {
	PartitionDataCache cache; // Cached partition data (loaded once in window callback)
};

/**
 * Initialize state (called once per partition)
 * For window-only functions, state is unused but must be initialized
 */
void OlsFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) OlsFitPredictState();
}

// Update (no-op for window-only function)
void OlsFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
	// No-op: window callback reads partition directly via LoadPartitionData
}

// Combine (no-op for window-only function)
void OlsFitPredictCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	// No-op
}

/**
 * Finalize: For non-window aggregate mode
 * Window functions use the window callback instead, but DuckDB still requires a finalize function
 */
void OlsFitPredictFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &aggr_input_data,
                           duckdb::Vector &result, idx_t count, idx_t offset) {
	// std::cerr << "[OLS FINALIZE] Called with count=" << count << ", offset=" << offset << " (should use OVER
	// clause!)" << std::endl; Not used in window mode If called in non-window mode, return NULL (user should use OVER
	// clause)
	auto &result_validity = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		result_validity.SetInvalid(i);
	}
}

/**
 * Window callback: Fit model on training data, predict for current row
 * This is called once per row in the window
 */
static void OlsFitPredictWindow(duckdb::AggregateInputData &aggr_input_data,
                                const duckdb::WindowPartitionInput &partition, duckdb::const_data_ptr_t g_state,
                                duckdb::data_ptr_t l_state, const duckdb::SubFrames &subframes, duckdb::Vector &result,
                                duckdb::idx_t rid) {

	// std::cerr << "[OLS WINDOW] Called for row " << rid << std::endl;

	// Access result validity
	auto &result_validity = FlatVector::Validity(result);

	// Access global state (cached partition data)
	auto &state = *reinterpret_cast<OlsFitPredictState *>(const_cast<data_ptr_t>(g_state));

	// Load partition data once (cached for all rows in this partition)
	LoadPartitionData(partition, state.cache);

	// Use cached data
	auto &all_y = state.cache.all_y;
	auto &all_x = state.cache.all_x;
	auto &options = state.cache.options;
	idx_t n_features = state.cache.n_features;

	// Compute frame signature to check if we can use cached model
	vector<idx_t> current_frame_sig = ComputeFrameSignature(subframes, all_y, all_x, options);

	// Check if we have a cached model that matches this frame
	OlsModelCache *cache = state.cache.model_cache;
	bool use_cache = false;
	if (cache && cache->initialized) {
		if (FrameSignaturesMatch(cache->train_indices, current_frame_sig)) {
			use_cache = true;
			ANOFOX_DEBUG("Using cached OLS model for row " << rid << ", frame_sig_size=" << current_frame_sig.size());
		} else {
			ANOFOX_DEBUG("Cache miss for row " << rid << ": cached_sig_size=" << cache->train_indices.size()
			                                   << ", current_sig_size=" << current_frame_sig.size());
		}
	} else {
		ANOFOX_DEBUG("No cache available for row " << rid << ", will fit new model");
	}

	// Variables to hold model parameters (either from cache or newly computed)
	double intercept = 0.0;
	Eigen::VectorXd coefficients;
	double mse = 0.0;
	Eigen::VectorXd x_means;
	Eigen::MatrixXd XtX_inv; // Precomputed (X'X)^(-1) for leverage calculation (always use this, never store X_train)
	idx_t df_residual = 0;
	idx_t n_train = 0; // Number of training samples (for debugging)

	if (!use_cache) {
		// Collect training data using frame signature (works for both expanding and fixed modes)
		vector<double> train_y;
		vector<vector<double>> train_x;

		for (idx_t data_idx : current_frame_sig) {
			train_y.push_back(all_y[data_idx]);
			train_x.push_back(all_x[data_idx]);
		}

		n_train = train_y.size();
		idx_t p = n_features;

		// Minimum observations:
		// - With intercept and p<=1: can fit with n=1 (intercept-only or simple regression)
		// - With intercept and p>=2: need n >= p+1 to ensure df_residual >= 1
		// - Without intercept: need n >= p
		idx_t min_required = options.intercept ? (p <= 1 ? 1 : p + 1) : p;
		if (n_train < min_required || p == 0) {
			ANOFOX_DEBUG("Insufficient data: n_train=" << n_train << " < " << min_required << " (p=" << p
			                                           << ", intercept=" << options.intercept << "): returning NULL");
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

		// Fit OLS model
		libanostat::core::RegressionResult ols_result;

		if (options.intercept) {
			// With intercept: center data, solve, compute intercept
			double mean_y = y_train.mean();
			x_means = X_train.colwise().mean();

			// Special case: n=1 with intercept → X_centered is all zeros (degenerate)
			// Solution: intercept = y[0], beta = 0 (intercept-only model)
			if (n_train == 1) {
				ols_result.coefficients = Eigen::VectorXd::Zero(p);
				ols_result.std_errors = Eigen::VectorXd::Zero(p);
				ols_result.is_aliased.assign(p, false);
				ols_result.rank = 0; // Rank of centered X (all zeros) is 0
				intercept = mean_y;

				// XtX_inv is undefined for n=1, but not used (df_residual=0 → no intervals)
				XtX_inv = Eigen::MatrixXd::Zero(p, p);
			} else {
				Eigen::VectorXd y_centered = y_train.array() - mean_y;
				Eigen::MatrixXd X_centered = X_train;
				for (idx_t j = 0; j < p; j++) {
					X_centered.col(j).array() -= x_means(j);
				}

				// Convert Eigen to DuckDB vectors
				vector<double> y_vec(n_train);
				vector<vector<double>> x_vec(n_train, vector<double>(p));
				for (size_t row = 0; row < n_train; row++) {
					y_vec[row] = y_centered(row);
					for (size_t col = 0; col < p; col++) {
						x_vec[row][col] = X_centered(row, col);
					}
				}
				// Use libanostat on centered data
				RegressionOptions centered_opts;
				centered_opts.intercept = false;
				ols_result = bridge::LibanostatWrapper::FitOLS(y_vec, x_vec, centered_opts, true);

				// Compute intercept
				double beta_dot_xmean = 0.0;
				for (idx_t j = 0; j < p; j++) {
					if (!std::isnan(ols_result.coefficients(j))) {
						beta_dot_xmean += ols_result.coefficients(j) * x_means(j);
					}
				}
				intercept = mean_y - beta_dot_xmean;

				// Compute XtX_inv for leverage calculation (using already-centered X)
				Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;
				// Use pseudo-inverse for numerical stability
				Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
				XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));
			}
		} else {
			// No intercept
			// Convert Eigen to DuckDB vectors
			vector<double> y_vec(n_train);
			vector<vector<double>> x_vec(n_train, vector<double>(p));
			for (size_t row = 0; row < n_train; row++) {
				y_vec[row] = y_train(row);
				for (size_t col = 0; col < p; col++) {
					x_vec[row][col] = X_train(row, col);
				}
			}
			RegressionOptions no_intercept_opts;
			no_intercept_opts.intercept = false;
			ols_result = bridge::LibanostatWrapper::FitOLS(y_vec, x_vec, no_intercept_opts, true);
			intercept = 0.0;
			x_means = Eigen::VectorXd::Zero(p);

			// Compute XtX_inv for leverage calculation
			Eigen::MatrixXd XtX = X_train.transpose() * X_train;
			// Use pseudo-inverse for numerical stability
			Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
			XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));
		}

		// Compute MSE and df_residual
		// NOTE: ols_result.coefficients are for centered X, but we can use them with uncentered X
		// as long as we add the intercept: yhat = intercept + beta * X
		Eigen::VectorXd y_pred_train(n_train);
		for (idx_t row = 0; row < n_train; row++) {
			y_pred_train(row) = intercept;
			for (idx_t j = 0; j < p; j++) {
				if (!std::isnan(ols_result.coefficients(j))) {
					y_pred_train(row) += ols_result.coefficients(j) * X_train(row, j);
				}
			}
		}

		Eigen::VectorXd residuals = y_train - y_pred_train;
		double ss_res = residuals.squaredNorm();

		// std::cerr << "[OLS FIT] y_pred_train[0]=" << y_pred_train(0) << ", residuals[0]=" << residuals(0) <<
		// std::endl; std::cerr << "[OLS FIT] ss_res=" << ss_res << std::endl;

		// NOTE: After ba2334b fix, ols_result.rank now includes intercept if fitted
		// df_model = rank directly
		idx_t df_model = ols_result.rank;
		df_residual = n_train - df_model;
		mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

		// std::cerr << "[OLS FIT] df_model=" << df_model << ", df_residual=" << df_residual << ", mse=" << mse <<
		// std::endl;

		// Store coefficients
		coefficients = ols_result.coefficients;

		// Verify coefficients are valid
		if (coefficients.size() == 0) {
			ANOFOX_DEBUG("Fitted model has empty coefficients, returning NULL for row " << rid);
			result_validity.SetInvalid(rid);
			return;
		}

		// X_train goes out of scope here - we only keep XtX_inv (much smaller)

		ANOFOX_DEBUG("Fitted OLS model: n_train=" << n_train << ", p=" << p << ", coefficients.size()="
		                                          << coefficients.size() << ", intercept=" << intercept);

		// Cache the model if frame is constant (caching is beneficial)
		// We cache when the frame covers all training rows (common case with OVER ())
		// For rolling windows, each row may have different frames, so caching may not help
		// But we still cache to avoid recomputation if frames happen to match
		// Only delete old cache if frame signature changed (optimization: reuse cache object if possible)
		if (state.cache.model_cache) {
			// Check if frame signature changed
			if (!FrameSignaturesMatch(state.cache.model_cache->train_indices, current_frame_sig)) {
				delete state.cache.model_cache;
				state.cache.model_cache = new OlsModelCache();
			}
			// Otherwise reuse existing cache object
		} else {
			state.cache.model_cache = new OlsModelCache();
		}
		cache = state.cache.model_cache;
		cache->coefficients = coefficients;
		cache->intercept = intercept;
		cache->mse = mse;
		cache->XtX_inv = XtX_inv; // Store XtX_inv instead of X_train (much smaller: P×P vs N×P)
		cache->x_means = x_means;
		cache->df_residual = df_residual;
		cache->rank = ols_result.rank;
		cache->n_train = n_train;
		cache->train_indices = current_frame_sig;
		cache->initialized = true;

		ANOFOX_DEBUG("Fitted and cached OLS model: n_train=" << n_train << ", p=" << p);
	} else {
		// Use cached model
		if (!cache || !cache->initialized || cache->coefficients.size() == 0) {
			ANOFOX_DEBUG("Cache invalid or empty, refitting model for row " << rid);
			// Fall back to fitting (this shouldn't happen, but be defensive)
			use_cache = false;
			// Recursively call the fitting logic - actually, better to just return NULL
			result_validity.SetInvalid(rid);
			return;
		}
		intercept = cache->intercept;
		coefficients = cache->coefficients;
		mse = cache->mse;
		x_means = cache->x_means;
		XtX_inv = cache->XtX_inv; // Use precomputed XtX_inv from cache
		df_residual = cache->df_residual;
		n_train = cache->n_train; // Get from cache for debugging
		ANOFOX_DEBUG("Using cached OLS model: n_train=" << n_train << ", coefficients.size()=" << coefficients.size());
	}

	// Now predict for the current row (rid)
	if (rid >= all_x.size() || all_x[rid].empty()) {
		// std::cerr << "[OLS PREDICT] Row " << rid << " invalid: size=" << all_x.size() << std::endl;
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

	// if (rid < 3) {
	// 	std::cerr << "[OLS PREDICT] Row " << rid << ": x=" << all_x[rid][0]
	// 	          << ", intercept=" << intercept
	// 	          << ", coef[0]=" << coefficients(0)
	// 	          << ", x_means[0]=" << x_means(0) << std::endl;
	// }

	// Compute prediction with interval (always use XtX_inv for memory efficiency)
	PredictionResult pred =
	    ComputePredictionWithIntervalXtXInv(all_x[rid], intercept, coefficients, mse, x_means, XtX_inv, n_train,
	                                        df_residual, 0.95, // TODO: Get from options
	                                        "prediction"       // TODO: Get from options
	    );

	if (!pred.is_valid) {
		// std::cerr << "[OLS PREDICT] Row " << rid << " prediction INVALID" << std::endl;
		result_validity.SetInvalid(rid);
		return;
	}

	if (rid < 3) {
		// std::cerr << "[OLS PREDICT] Row " << rid << " SUCCESS: yhat=" << pred.yhat << std::endl;
	}

	// Fill result
	yhat_data[rid] = pred.yhat;
	yhat_lower_data[rid] = pred.yhat_lower;
	yhat_upper_data[rid] = pred.yhat_upper;
	std_error_data[rid] = pred.std_error;

	ANOFOX_DEBUG("OLS fit-predict window: n_train=" << n_train << ", rid=" << rid << ", yhat=" << pred.yhat);
}

void OlsFitPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering OLS fit-predict function");

	// Define struct fields inline
	// NOTE: Includes dummy LIST field to trigger proper DuckDB struct initialization in window aggregates
	child_list_t<LogicalType> fit_predict_struct_fields;
	fit_predict_struct_fields.push_back(make_pair("yhat", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("_dummy", LogicalType::LIST(LogicalType::DOUBLE)));

	// Use real callback functions (not lambdas) like the working function does
	AggregateFunction anofox_statistics_fit_predict_ols(
	    "anofox_statistics_fit_predict_ols",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<OlsFitPredictState>,
	    OlsFitPredictInitialize,                     // initialize
	    OlsFitPredictUpdate,                         // update
	    OlsFitPredictCombine,                        // combine
	    OlsFitPredictFinalize,                       // finalize
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, // null_handling
	    nullptr,                                     // simple_update (not needed, we use window)
	    nullptr,                                     // bind
	    nullptr,                                     // destructor
	    nullptr,                                     // statistics
	    OlsFitPredictWindow,                         // window callback - THIS IS THE KEY!
	    nullptr,                                     // serialize
	    nullptr);                                    // deserialize

	// Mark as order-dependent to ensure window processing is used
	anofox_statistics_fit_predict_ols.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;

	loader.RegisterFunction(anofox_statistics_fit_predict_ols);

	ANOFOX_DEBUG("OLS fit-predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
