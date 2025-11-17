#include "ridge_fit_predict.hpp"
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

// State for Ridge fit-predict
struct RidgeFitPredictState {
	// Empty state - window callback reads partition directly
};

// Initialize state
static void RidgeFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) RidgeFitPredictState();
}

// Update (no-op for window-only function)
static void RidgeFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
	// No-op: window callback reads partition directly
}

// Combine (no-op for window-only function)
static void RidgeFitPredictCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	// No-op
}

// Finalize (returns NULL for non-window mode)
static void RidgeFitPredictFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
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

	// Extract data for the entire partition
	vector<double> all_y;
	vector<vector<double>> all_x;
	RegressionOptions options;
	bool options_initialized = false;
	idx_t n_features = 0;

	// Read input data from the partition
	ColumnDataScanState scan_state;
	partition.inputs->InitializeScan(scan_state);

	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), partition.inputs->Types());

	while (partition.inputs->Scan(scan_state, chunk)) {
		auto &y_chunk = chunk.data[0];
		auto &x_array_chunk = chunk.data[1];
		auto &options_chunk = chunk.data[2];

		UnifiedVectorFormat y_data;
		UnifiedVectorFormat x_array_data;
		UnifiedVectorFormat options_data;
		y_chunk.ToUnifiedFormat(chunk.size(), y_data);
		x_array_chunk.ToUnifiedFormat(chunk.size(), x_array_data);
		options_chunk.ToUnifiedFormat(chunk.size(), options_data);

		auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);

		for (idx_t i = 0; i < chunk.size(); i++) {
			auto y_idx = y_data.sel->get_index(i);
			auto x_array_idx = x_array_data.sel->get_index(i);
			auto options_idx = options_data.sel->get_index(i);

			if (!options_initialized && options_data.validity.RowIsValid(options_idx)) {
				auto options_value = options_chunk.GetValue(options_idx);
				options = RegressionOptions::ParseFromMap(options_value);
				options_initialized = true;
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
					if (n_features == 0) {
						n_features = features.size();
					}
					x_valid = (features.size() == n_features);
				}
			}

			if (x_valid) {
				all_x.push_back(features);
				if (y_data.validity.RowIsValid(y_idx)) {
					all_y.push_back(y_ptr[y_idx]);
				} else {
					all_y.push_back(std::numeric_limits<double>::quiet_NaN());
				}
			} else {
				all_x.push_back(vector<double>());
				all_y.push_back(std::numeric_limits<double>::quiet_NaN());
			}
		}
	}

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

	// Fit Ridge model: β = (X'X + λI)^(-1) X'y
	double intercept = 0.0;
	Eigen::VectorXd beta;
	Eigen::VectorXd x_means;
	idx_t rank;

	if (options.intercept) {
		// Center data
		double y_mean = y_train.mean();
		x_means = X_train.colwise().mean();

		Eigen::VectorXd y_centered = y_train.array() - y_mean;
		Eigen::MatrixXd X_centered = X_train;
		for (idx_t j = 0; j < p; j++) {
			X_centered.col(j).array() -= x_means(j);
		}

		// Ridge: β = (X'X + λI)^(-1) X'y
		Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;
		Eigen::VectorXd Xty = X_centered.transpose() * y_centered;
		Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
		Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
		beta = qr.solve(Xty);
		rank = qr.rank();

		// Compute intercept
		double beta_dot_xmean = beta.dot(x_means);
		intercept = y_mean - beta_dot_xmean;
	} else {
		// No intercept
		Eigen::MatrixXd XtX = X_train.transpose() * X_train;
		Eigen::VectorXd Xty = X_train.transpose() * y_train;
		Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
		Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
		beta = qr.solve(Xty);
		rank = qr.rank();

		intercept = 0.0;
		x_means = Eigen::VectorXd::Zero(p);
	}

	// Compute MSE
	Eigen::VectorXd y_pred_train = Eigen::VectorXd::Constant(n_train, intercept);
	y_pred_train += X_train * beta;

	Eigen::VectorXd residuals = y_train - y_pred_train;
	double ss_res = residuals.squaredNorm();

	idx_t df_model = rank;
	idx_t df_residual = n_train - df_model;
	double mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

	// Predict for current row
	if (rid >= all_x.size() || all_x[rid].empty()) {
		result_validity.SetInvalid(rid);
		return;
	}

	// Compute prediction with interval
	// NOTE: Ridge is biased, so intervals are approximate
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
	AggregateFunction anofox_statistics_fit_predict_ridge(
	    "anofox_statistics_fit_predict_ridge",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<RidgeFitPredictState>,
	    RidgeFitPredictInitialize, RidgeFitPredictUpdate, RidgeFitPredictCombine, RidgeFitPredictFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr,
	    nullptr, // destroy - use nullptr like the working function
	    nullptr,
	    RidgeFitPredictWindow, // Ridge-specific window callback
	    nullptr, nullptr);

	loader.RegisterFunction(anofox_statistics_fit_predict_ridge);

	ANOFOX_DEBUG("Ridge fit-predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
