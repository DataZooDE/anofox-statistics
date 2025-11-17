#include "elastic_net_fit_predict.hpp"
#include "fit_predict_base.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"
#include "../utils/options_parser.hpp"
#include "../utils/elastic_net_solver.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

// State for Elastic Net fit-predict
struct ElasticNetFitPredictState {
	PartitionDataCache cache; // Cached partition data (loaded once)
};

// Initialize state
static void ElasticNetFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	new (state_ptr) ElasticNetFitPredictState();
}

/**
 * Window callback for Elastic Net fit-predict
 * Fits Elastic Net regression and predicts for current row
 */
static void ElasticNetFitPredictWindow(duckdb::AggregateInputData &aggr_input_data,
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

	// Access global state (cached partition data)
	auto &state = *reinterpret_cast<ElasticNetFitPredictState *>(const_cast<data_ptr_t>(g_state));

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

	// Fit Elastic Net model using ElasticNetSolver
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

		// Fit Elastic Net on centered data
		auto result = ElasticNetSolver::Fit(y_centered, X_centered, options.alpha, options.lambda);
		beta = result.coefficients;
		rank = (beta.array().abs() > 1e-10).count(); // Count non-zero coefficients

		// Compute intercept
		double beta_dot_xmean = beta.dot(x_means);
		intercept = y_mean - beta_dot_xmean;
	} else {
		// No intercept
		auto result = ElasticNetSolver::Fit(y_train, X_train, options.alpha, options.lambda);
		beta = result.coefficients;
		rank = (beta.array().abs() > 1e-10).count(); // Count non-zero coefficients

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
	// NOTE: Elastic Net is biased, so intervals are approximate
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

	ANOFOX_DEBUG("Elastic Net fit-predict: n_train=" << n_train << ", alpha=" << options.alpha
	                                                 << ", lambda=" << options.lambda << ", yhat=" << pred.yhat);
}

void ElasticNetFitPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering Elastic Net fit-predict function");

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
	AggregateFunction anofox_statistics_fit_predict_elastic_net(
	    "anofox_statistics_fit_predict_elastic_net",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(fit_predict_struct_fields), AggregateFunction::StateSize<ElasticNetFitPredictState>,
	    ElasticNetFitPredictInitialize,
	    nullptr, // No update - force WindowCustomAggregator
	    nullptr, // No combine - force WindowCustomAggregator
	    nullptr, // No finalize - force WindowCustomAggregator
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr, nullptr,
	    ElasticNetFitPredictWindow, // Elastic Net-specific window callback - called per row
	    nullptr, nullptr);

	loader.RegisterFunction(anofox_statistics_fit_predict_elastic_net);

	ANOFOX_DEBUG("Elastic Net fit-predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
