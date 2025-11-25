#include "ridge_predict_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"
#include "../utils/statistical_distributions.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

struct RidgePredictAggState {
	// Model parameters (constant across all rows)
	Eigen::VectorXd coefficients;
	double intercept;
	double mse;
	Eigen::VectorXd x_train_means;
	Eigen::VectorXd coeff_std_errors;
	double intercept_std_error;
	idx_t df_residual;

	// Configuration
	double confidence_level;
	bool is_prediction_interval;

	// XtX_inv for leverage calculation (computed once)
	Eigen::MatrixXd XtX_inv;
	bool xtx_inv_computed = false;

	bool initialized = false;
};

struct RidgePredictAggOperation {
	template <class RESULT, class STATE>
	static void Initialize(STATE &state) {
		new (&state) STATE();
	}

	template <class RESULT, class STATE>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
		// Predict aggregates don't combine states (applied per-row or per-group)
	}

	template <class RESULT, class STATE>
	static void Finalize(STATE &state, RESULT &result, AggregateFinalizeData &finalize_data) {
		// For predict aggregates, we don't finalize - each row produces its own result
		// This is handled in the window function path
		finalize_data.ReturnNull();
	}

	static bool IgnoreNull() {
		return false;
	}
};

/**
 * Update function - called for each row
 */
static void RidgePredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                          Vector &state_vector, idx_t count) {

	// This aggregate is designed for window functions primarily
	// For GROUP BY, it would need a different approach
	// For now, we'll implement the window function path
}

/**
 * Window function - processes each row in window context
 */
static void RidgePredictAggWindow(Vector inputs[], const ValidityMask &filter, AggregateInputData &aggr_input_data,
                                          idx_t input_count, data_ptr_t state_p, const SubFrames &frames,
                                          Vector &result, idx_t rid) {

	auto &state = *reinterpret_cast<RidgePredictAggState *>(state_p);

	// Initialize state from first row of partition
	if (!state.initialized) {
		// Extract model parameters from inputs[0-6]
		auto &coeff_input = inputs[0];
		auto &intercept_input = inputs[1];
		auto &mse_input = inputs[2];
		auto &x_means_input = inputs[3];
		auto &coeff_se_input = inputs[4];
		auto &intercept_se_input = inputs[5];
		auto &df_input = inputs[6];
		auto &options_input = inputs[8];

		// Get values from first valid row
		idx_t first_valid = 0;
		auto coeff_data = FlatVector::GetData<list_entry_t>(coeff_input);
		auto &coeff_list = ListVector::GetEntry(coeff_input);
		auto coeff_values = FlatVector::GetData<double>(coeff_list);

		idx_t coeff_len = coeff_data[first_valid].length;
		state.coefficients = Eigen::VectorXd(coeff_len);
		for (idx_t j = 0; j < coeff_len; j++) {
			state.coefficients(j) = coeff_values[coeff_data[first_valid].offset + j];
		}

		state.intercept = FlatVector::GetData<double>(intercept_input)[first_valid];
		state.mse = FlatVector::GetData<double>(mse_input)[first_valid];
		state.df_residual = FlatVector::GetData<int64_t>(df_input)[first_valid];
		state.intercept_std_error = FlatVector::GetData<double>(intercept_se_input)[first_valid];

		// Extract x_train_means
		auto x_means_data = FlatVector::GetData<list_entry_t>(x_means_input);
		auto &x_means_list = ListVector::GetEntry(x_means_input);
		auto x_means_values = FlatVector::GetData<double>(x_means_list);

		idx_t x_means_len = x_means_data[first_valid].length;
		state.x_train_means = Eigen::VectorXd(x_means_len);
		for (idx_t j = 0; j < x_means_len; j++) {
			state.x_train_means(j) = x_means_values[x_means_data[first_valid].offset + j];
		}

		// Extract coeff_std_errors
		auto coeff_se_data = FlatVector::GetData<list_entry_t>(coeff_se_input);
		auto &coeff_se_list = ListVector::GetEntry(coeff_se_input);
		auto coeff_se_values = FlatVector::GetData<double>(coeff_se_list);

		state.coeff_std_errors = Eigen::VectorXd(coeff_len);
		for (idx_t j = 0; j < coeff_len; j++) {
			state.coeff_std_errors(j) = coeff_se_values[coeff_se_data[first_valid].offset + j];
		}

		// Parse options
		auto &options_value = options_input.GetValue(first_valid);
		RegressionOptions options = RegressionOptions::ParseFromMap(options_value);
		state.confidence_level = options.confidence_level;
		state.is_prediction_interval = true; // Default

		state.initialized = true;
	}

	// Get current row's features (inputs[7])
	auto &x_input = inputs[7];
	auto x_data = FlatVector::GetData<list_entry_t>(x_input);
	auto &x_list = ListVector::GetEntry(x_input);
	auto x_values = FlatVector::GetData<double>(x_list);

	idx_t x_len = x_data[rid].length;
	Eigen::VectorXd x_current(x_len);
	for (idx_t j = 0; j < x_len; j++) {
		x_current(j) = x_values[x_data[rid].offset + j];
	}

	// Point prediction
	double y_pred = state.intercept + state.coefficients.dot(x_current);

	// Compute standard error (simplified - assumes no leverage term)
	// For full implementation, would need XtX_inv computation
	double se;
	if (state.is_prediction_interval) {
		se = std::sqrt(state.mse); // Simplified
	} else {
		se = std::sqrt(state.mse * 0.1); // Simplified confidence interval
	}

	// Compute interval
	double alpha = 1.0 - state.confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, static_cast<int>(state.df_residual));
	double ci_lower = y_pred - t_crit * se;
	double ci_upper = y_pred + t_crit * se;

	// Write result as STRUCT
	auto &result_validity = FlatVector::Validity(result);
	auto result_data = FlatVector::GetData<double>(result);

	// Return STRUCT with 4 fields: yhat, yhat_lower, yhat_upper, std_error
	// Note: DuckDB struct handling - this needs proper struct construction
	// For now, simplified version that returns the prediction
	result_data[rid] = y_pred;
	result_validity.SetValid(rid);
}

/**
 * Create return type for predict aggregate (STRUCT)
 */
static LogicalType CreatePredictAggReturnType() {
	child_list_t<LogicalType> struct_children;
	struct_children.push_back({"yhat", LogicalType::DOUBLE});
	struct_children.push_back({"yhat_lower", LogicalType::DOUBLE});
	struct_children.push_back({"yhat_upper", LogicalType::DOUBLE});
	struct_children.push_back({"std_error", LogicalType::DOUBLE});
	return LogicalType::STRUCT(struct_children);
}

void RidgePredictAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering Ridge predict aggregate function");

	// Note: This is a simplified implementation
	// Full implementation would require proper struct handling in window function

	AggregateFunction func(
		"anofox_statistics_predict_ridge_agg",
		{
			LogicalType::LIST(LogicalType::DOUBLE),  // coefficients
			LogicalType::DOUBLE,                     // intercept
			LogicalType::DOUBLE,                     // mse
			LogicalType::LIST(LogicalType::DOUBLE),  // x_train_means
			LogicalType::LIST(LogicalType::DOUBLE),  // coefficient_std_errors
			LogicalType::DOUBLE,                     // intercept_std_error
			LogicalType::BIGINT,                     // df_residual
			LogicalType::LIST(LogicalType::DOUBLE),  // x (current row features)
			LogicalType::MAP(LogicalType::VARCHAR, LogicalType::ANY) // options
		},
		CreatePredictAggReturnType(),
		AggregateFunction::StateSize<RidgePredictAggState>,
		AggregateFunction::StateInitialize<RidgePredictAggState, RidgePredictAggOperation>,
		RidgePredictAggUpdate,
		AggregateFunction::StateCombine<RidgePredictAggState, RidgePredictAggOperation>,
		AggregateFunction::StateFinalize<RidgePredictAggState, double, RidgePredictAggOperation>,
		nullptr, // simple_update
		nullptr, // bind
		nullptr, // destructor
		nullptr, // cardinality
		RidgePredictAggWindow
	);

	func.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	func.window = FunctionNullHandling::SPECIAL_HANDLING;

	loader.RegisterFunction(func);

	ANOFOX_DEBUG("Ridge predict aggregate function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
