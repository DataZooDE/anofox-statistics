#include "wls_fit_predict.hpp"
#include "fit_predict_base.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

// State for WLS fit-predict
struct WlsFitPredictState {
	PartitionDataCache cache; // Cached partition data (loaded once)
	vector<double> all_weights; // WLS-specific: weights for each row
};

// Initialize state
static void WlsFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) WlsFitPredictState();
}

/**
 * Window callback for WLS fit-predict
 * Fits Weighted Least Squares regression and predicts for current row
 */
static void WlsFitPredictWindow(duckdb::AggregateInputData &aggr_input_data,
                                const duckdb::WindowPartitionInput &partition, duckdb::const_data_ptr_t g_state,
                                duckdb::data_ptr_t l_state, const duckdb::SubFrames &subframes, duckdb::Vector &result,
                                duckdb::idx_t rid) {

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
	auto &state = *reinterpret_cast<WlsFitPredictState *>(const_cast<data_ptr_t>(g_state));

	// Load partition data once (cached for all rows)
	// Note: WLS has weights as 3rd input, so we need custom loading
	if (!state.cache.initialized) {
		ColumnDataScanState scan_state;
		partition.inputs->InitializeScan(scan_state);

		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), partition.inputs->Types());

		while (partition.inputs->Scan(scan_state, chunk)) {
			auto &y_chunk = chunk.data[0];
			auto &x_array_chunk = chunk.data[1];
			auto &weights_chunk = chunk.data[2];
			auto &options_chunk = chunk.data[3];

			UnifiedVectorFormat y_data;
			UnifiedVectorFormat x_array_data;
			UnifiedVectorFormat weights_data;
			UnifiedVectorFormat options_data;
			y_chunk.ToUnifiedFormat(chunk.size(), y_data);
			x_array_chunk.ToUnifiedFormat(chunk.size(), x_array_data);
			weights_chunk.ToUnifiedFormat(chunk.size(), weights_data);
			options_chunk.ToUnifiedFormat(chunk.size(), options_data);

			auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);
			auto weights_ptr = UnifiedVectorFormat::GetData<double>(weights_data);

			for (idx_t i = 0; i < chunk.size(); i++) {
				auto y_idx = y_data.sel->get_index(i);
				auto x_array_idx = x_array_data.sel->get_index(i);
				auto weights_idx = weights_data.sel->get_index(i);
				auto options_idx = options_data.sel->get_index(i);

				if (!state.cache.options_initialized && options_data.validity.RowIsValid(options_idx)) {
					auto options_value = options_chunk.GetValue(options_idx);
					state.cache.options = RegressionOptions::ParseFromMap(options_value);
					state.cache.options_initialized = true;
				}

				vector<double> features;
				bool x_valid = false;

				if (x_array_data.validity.RowIsValid(x_array_idx)) {
					auto x_array_entry = UnifiedVectorFormat::GetData<list_entry_t>(x_array_data)[x_array_idx];
					auto &x_child = ListVector::GetEntry(x_array_chunk);

					UnifiedVectorFormat x_child_data;
					x_child.ToUnifiedFormat(ListVector::GetListSize(x_array_chunk), x_child_data);
					auto x_child_ptr = UnifiedVectorFormat::GetData<double>(x_child_data);

					for (idx_t j = 0; j < x_array_entry.length; j++) {
						auto child_idx = x_child_data.sel->get_index(x_array_entry.offset + j);
						if (x_child_data.validity.RowIsValid(child_idx)) {
							features.push_back(x_child_ptr[child_idx]);
						} else {
							features.clear();
							break;
						}
					}

					if (!features.empty()) {
						if (state.cache.n_features == 0) {
							state.cache.n_features = features.size();
						}
						x_valid = (features.size() == state.cache.n_features);
					}
				}

				// Extract weight (default to 1.0 if NULL)
				double weight = 1.0;
				if (weights_data.validity.RowIsValid(weights_idx)) {
					weight = weights_ptr[weights_idx];
				}

				if (x_valid) {
					state.cache.all_x.push_back(features);
					state.all_weights.push_back(weight);
					if (y_data.validity.RowIsValid(y_idx)) {
						state.cache.all_y.push_back(y_ptr[y_idx]);
					} else {
						state.cache.all_y.push_back(std::numeric_limits<double>::quiet_NaN());
					}
				} else {
					state.cache.all_x.push_back(vector<double>());
					state.all_weights.push_back(weight);
					state.cache.all_y.push_back(std::numeric_limits<double>::quiet_NaN());
				}
			}
		}

		state.cache.initialized = true;
	}

	// Use cached data
	auto &all_y = state.cache.all_y;
	auto &all_x = state.cache.all_x;
	auto &all_weights = state.all_weights;
	auto &options = state.cache.options;
	idx_t n_features = state.cache.n_features;

	// Compute frame signature (training row indices)
	vector<idx_t> frame_indices = ComputeFrameSignature(subframes, all_y, all_x, options);

	// Collect training data from frame signature
	vector<double> train_y;
	vector<vector<double>> train_x;
	vector<double> train_weights;

	for (idx_t idx : frame_indices) {
		train_y.push_back(all_y[idx]);
		train_x.push_back(all_x[idx]);
		train_weights.push_back(all_weights[idx]);
	}

	idx_t n_train = train_y.size();
	idx_t p = n_features;

	// Need at least p + (intercept ? 1 : 0) + 1 observations for p features
	idx_t min_required = p + (options.intercept ? 1 : 0) + 1;
	if (n_train < min_required || p == 0) {
		FlatVector::SetNull(result, rid, true);
		return;
	}

	// Validate weights are positive
	for (idx_t i = 0; i < n_train; i++) {
		if (train_weights[i] <= 0.0) {
			FlatVector::SetNull(result, rid, true);
			return;
		}
	}

	// Build training matrices
	Eigen::MatrixXd X_train(n_train, p);
	Eigen::VectorXd y_train(n_train);
	Eigen::VectorXd w(n_train);

	for (idx_t row = 0; row < n_train; row++) {
		y_train(row) = train_y[row];
		w(row) = train_weights[row];
		for (idx_t col = 0; col < p; col++) {
			X_train(row, col) = train_x[row][col];
		}
	}

	// Fit WLS model
	double intercept = 0.0;
	Eigen::VectorXd beta;
	Eigen::VectorXd x_means;
	idx_t rank;

	// Compute sqrt(w) for transformation
	Eigen::VectorXd sqrt_w = w.array().sqrt();

	if (options.intercept) {
		// Compute weighted means
		double sum_weights = w.sum();
		double y_weighted_mean = (w.array() * y_train.array()).sum() / sum_weights;
		x_means = Eigen::VectorXd::Zero(p);
		for (idx_t j = 0; j < p; j++) {
			x_means(j) = (w.array() * X_train.col(j).array()).sum() / sum_weights;
		}

		// Center data using weighted means
		Eigen::VectorXd y_centered = y_train.array() - y_weighted_mean;
		Eigen::MatrixXd X_centered = X_train;
		for (idx_t j = 0; j < p; j++) {
			X_centered.col(j).array() -= x_means(j);
		}

		// Apply sqrt(W) transformation to centered data
		Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_centered;
		Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_centered;

		// Solve weighted OLS on transformed data
		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_weighted);
		beta = qr.solve(y_weighted);
		rank = qr.rank();

		// Compute intercept
		double beta_dot_xmean = beta.dot(x_means);
		intercept = y_weighted_mean - beta_dot_xmean;
	} else {
		// No intercept - apply sqrt(W) transformation directly
		Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_train;
		Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_train;

		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_weighted);
		beta = qr.solve(y_weighted);
		rank = qr.rank();

		intercept = 0.0;
		x_means = Eigen::VectorXd::Zero(p);
	}

	// Compute MSE (weighted)
	Eigen::VectorXd y_pred_train = Eigen::VectorXd::Constant(n_train, intercept);
	y_pred_train += X_train * beta;

	Eigen::VectorXd residuals = y_train - y_pred_train;
	// Weighted sum of squared residuals
	double ss_res = (w.array() * residuals.array() * residuals.array()).sum();

	// NOTE: rank is from QR of centered X (p features), so add 1 for intercept
	idx_t df_model = rank + (options.intercept ? 1 : 0);
	idx_t df_residual = n_train - df_model;
	
	// Check that we have sufficient degrees of freedom for residual
	if (df_residual <= 0) {
		FlatVector::SetNull(result, rid, true);
		return;
	}
	
	double mse = ss_res / df_residual;
	
	// Check that MSE is valid and positive
	if (mse <= 0 || std::isnan(mse) || std::isinf(mse)) {
		FlatVector::SetNull(result, rid, true);
		return;
	}

	// Predict for current row
	if (rid >= all_x.size() || all_x[rid].empty()) {
		FlatVector::SetNull(result, rid, true);
		return;
	}

	// Compute prediction with interval
	PredictionResult pred = ComputePredictionWithInterval(all_x[rid], intercept, beta, mse, x_means, X_train,
	                                                      df_residual, 0.95, "prediction");

	if (!pred.is_valid) {
		FlatVector::SetNull(result, rid, true);
		return;
	}

	yhat_data[rid] = pred.yhat;
	yhat_lower_data[rid] = pred.yhat_lower;
	yhat_upper_data[rid] = pred.yhat_upper;
	std_error_data[rid] = pred.std_error;

	ANOFOX_DEBUG("WLS fit-predict: n_train=" << n_train << ", yhat=" << pred.yhat);
}

void WlsFitPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering WLS fit-predict function");

	// Define struct fields inline
	// NOTE: Includes dummy LIST field to trigger proper DuckDB struct initialization in window aggregates
	child_list_t<LogicalType> fit_predict_struct_fields;
	fit_predict_struct_fields.push_back(make_pair("yhat", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));
	fit_predict_struct_fields.push_back(make_pair("_dummy", LogicalType::LIST(LogicalType::DOUBLE)));

	// Use ONLY window callback to force WindowCustomAggregator path
	AggregateFunction anofox_statistics_fit_predict_wls(
	    "anofox_statistics_fit_predict_wls",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<WlsFitPredictState>,
	    WlsFitPredictInitialize,
	    nullptr, // No update - force WindowCustomAggregator
	    nullptr, // No combine - force WindowCustomAggregator
	    nullptr, // No finalize - force WindowCustomAggregator
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr, nullptr,
	    WlsFitPredictWindow, // WLS-specific window callback - called per row
	    nullptr, nullptr);

	loader.RegisterFunction(anofox_statistics_fit_predict_wls);

	ANOFOX_DEBUG("WLS fit-predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
