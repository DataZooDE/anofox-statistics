#include "ols_fit_predict.hpp"
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
	PartitionDataCache cache; // Cached partition data (loaded once)
};

/**
 * Initialize state (called once per partition)
 * For window-only functions, state is unused but must be initialized
 */
void OlsFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) OlsFitPredictState();
}

/**
 * Update state with new row (called for each row in partition)
 * Accumulates both training data (y NOT NULL) and all data for prediction
 * Non-static so it can be reused by other fit-predict functions
 */
void OlsFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<FitPredictState *>(state_data);

	// Get y, x_array, and options vectors
	auto &y_vector = inputs[0];
	auto &x_array_vector = inputs[1];
	auto &options_vector = inputs[2];

	UnifiedVectorFormat y_data;
	UnifiedVectorFormat x_array_data;
	UnifiedVectorFormat options_data;
	y_vector.ToUnifiedFormat(count, y_data);
	x_array_vector.ToUnifiedFormat(count, x_array_data);
	options_vector.ToUnifiedFormat(count, options_data);

	auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);

	// Process each row
	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_idx = y_data.sel->get_index(i);
		auto x_array_idx = x_array_data.sel->get_index(i);
		auto options_idx = options_data.sel->get_index(i);

		// Initialize options from first row
		if (!state.options_initialized && options_data.validity.RowIsValid(options_idx)) {
			auto options_value = options_vector.GetValue(options_idx);
			state.options = RegressionOptions::ParseFromMap(options_value);
			state.options_initialized = true;
		}

		// Extract features from x_array
		vector<double> features;
		bool x_valid = false;

		if (x_array_data.validity.RowIsValid(x_array_idx)) {
			auto x_array_entry = UnifiedVectorFormat::GetData<list_entry_t>(x_array_data)[x_array_idx];
			auto &x_child = ListVector::GetEntry(x_array_vector);

			UnifiedVectorFormat x_child_data;
			x_child.ToUnifiedFormat(ListVector::GetListSize(x_array_vector), x_child_data);
			auto x_child_ptr = UnifiedVectorFormat::GetData<double>(x_child_data);

			for (idx_t j = 0; j < x_array_entry.length; j++) {
				auto child_idx = x_child_data.sel->get_index(x_array_entry.offset + j);
				if (x_child_data.validity.RowIsValid(child_idx)) {
					features.push_back(x_child_ptr[child_idx]);
				} else {
					// NULL in features array - skip this row entirely
					features.clear();
					break;
				}
			}

			if (!features.empty()) {
				x_valid = true;

				// Initialize n_features from first valid row
				if (state.n_features == 0) {
					state.n_features = features.size();
				} else if (features.size() != state.n_features) {
					// Feature count mismatch - skip this row
					x_valid = false;
				}
			}
		}

		// Store x_all for prediction (even if y is NULL)
		if (x_valid) {
			state.x_all.push_back(features);

			// Check if this is a training row (y NOT NULL)
			bool is_train = y_data.validity.RowIsValid(y_idx);
			state.is_train_row.push_back(is_train);

			if (is_train) {
				// Add to training data
				state.y_train.push_back(y_ptr[y_idx]);
				state.x_train.push_back(features);
			}
		} else {
			// Invalid x - mark as non-trainable
			state.x_all.push_back(vector<double>());
			state.is_train_row.push_back(false);
		}
	}
}

/**
 * Combine two states (for parallel aggregation)
 * Non-static so it can be reused by other fit-predict functions
 */
void OlsFitPredictCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<FitPredictState *>(source);
	auto target_ptr = FlatVector::GetData<FitPredictState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

		// Merge training data
		target_state.y_train.insert(target_state.y_train.end(), source_state.y_train.begin(),
		                            source_state.y_train.end());
		target_state.x_train.insert(target_state.x_train.end(), source_state.x_train.begin(),
		                            source_state.x_train.end());

		// Merge all data
		target_state.x_all.insert(target_state.x_all.end(), source_state.x_all.begin(), source_state.x_all.end());
		target_state.is_train_row.insert(target_state.is_train_row.end(), source_state.is_train_row.begin(),
		                                 source_state.is_train_row.end());

		if (target_state.n_features == 0) {
			target_state.n_features = source_state.n_features;
		}
		if (!target_state.options_initialized && source_state.options_initialized) {
			target_state.options = source_state.options;
			target_state.options_initialized = true;
		}
	}
}

/**
 * Destroy: Clean up state
 */
void OlsFitPredictDestroy(duckdb::Vector &state_vector, duckdb::AggregateInputData &aggr_input_data, idx_t count) {
	auto states = FlatVector::GetData<FitPredictState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		states[i]->~FitPredictState();
	}
}

/**
 * Finalize: For non-window aggregate mode
 * Window functions use the window callback instead, but DuckDB still requires a finalize function
 */
void OlsFitPredictFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &aggr_input_data,
                           duckdb::Vector &result, idx_t count, idx_t offset) {
	// Not used in window mode
	// If called in non-window mode, return NULL (user should use OVER clause)
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

	// Access result validity
	auto &result_validity = FlatVector::Validity(result);

	// Result is a STRUCT with fields: yhat, yhat_lower, yhat_upper, std_error, _dummy
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

	// Access global state (cached partition data)
	auto &state = *reinterpret_cast<OlsFitPredictState *>(const_cast<data_ptr_t>(g_state));

	// Load partition data once (cached for all rows)
	LoadPartitionData(partition, state.cache);

	// Use cached data
	auto &all_y = state.cache.all_y;
	auto &all_x = state.cache.all_x;
	auto &options = state.cache.options;
	idx_t n_features = state.cache.n_features;

	// Collect training data from window frame
	vector<double> train_y;
	vector<vector<double>> train_x;

	for (const auto &frame : subframes) {
		for (idx_t frame_idx = frame.start; frame_idx < frame.end; frame_idx++) {
			if (frame_idx < all_y.size() && !std::isnan(all_y[frame_idx]) && !all_x[frame_idx].empty()) {
				train_y.push_back(all_y[frame_idx]);
				train_x.push_back(all_x[frame_idx]);
			}
		}
	}

	idx_t n_train = train_y.size();
	idx_t p = n_features;

	// Need at least p+1 observations for p features
	if (n_train < p + 1 || p == 0) {
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
	double intercept = 0.0;
	RankDeficientOlsResult ols_result;
	Eigen::VectorXd x_means;

	if (options.intercept) {
		// With intercept: center data, solve, compute intercept
		double mean_y = y_train.mean();
		x_means = X_train.colwise().mean();

		Eigen::VectorXd y_centered = y_train.array() - mean_y;
		Eigen::MatrixXd X_centered = X_train;
		for (idx_t j = 0; j < p; j++) {
			X_centered.col(j).array() -= x_means(j);
		}

		ols_result = RankDeficientOls::FitWithStdErrors(y_centered, X_centered);

		// Compute intercept
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				beta_dot_xmean += ols_result.coefficients[j] * x_means(j);
			}
		}
		intercept = mean_y - beta_dot_xmean;
	} else {
		// No intercept
		ols_result = RankDeficientOls::FitWithStdErrors(y_train, X_train);
		intercept = 0.0;
		x_means = Eigen::VectorXd::Zero(p);
	}

	// Compute MSE and df_residual
	Eigen::VectorXd y_pred_train = Eigen::VectorXd::Constant(n_train, intercept);
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			y_pred_train += ols_result.coefficients[j] * X_train.col(j);
		}
	}

	Eigen::VectorXd residuals = y_train - y_pred_train;
	double ss_res = residuals.squaredNorm();

	idx_t df_model = ols_result.rank;
	idx_t df_residual = n_train - df_model;
	double mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

	// Now predict for the current row (rid)
	if (rid >= all_x.size() || all_x[rid].empty()) {
		result_validity.SetInvalid(rid);
		return;
	}

	// Compute prediction with interval
	PredictionResult pred = ComputePredictionWithInterval(all_x[rid], intercept, ols_result.coefficients, mse, x_means,
	                                                      X_train, df_residual,
	                                                      0.95,        // TODO: Get from options
	                                                      "prediction" // TODO: Get from options
	);

	if (!pred.is_valid) {
		result_validity.SetInvalid(rid);
		return;
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
	    OlsFitPredictInitialize, OlsFitPredictUpdate, OlsFitPredictCombine, OlsFitPredictFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr,
	    nullptr, // destroy - use nullptr like the working function
	    nullptr,
	    OlsFitPredictWindow, // Window callback does all the work
	    nullptr, nullptr);

	loader.RegisterFunction(anofox_statistics_fit_predict_ols);

	ANOFOX_DEBUG("OLS fit-predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
