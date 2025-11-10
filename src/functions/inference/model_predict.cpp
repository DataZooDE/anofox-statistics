#include "model_predict.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"
#include "../utils/statistical_distributions.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

namespace duckdb {
namespace anofox_statistics {

/**
 * Bind data for model prediction
 */
struct ModelPredictBindData : public TableFunctionData {
	// Model parameters
	double intercept;
	vector<double> coefficients;
	double mse;
	vector<double> x_train_means;
	vector<double> coefficient_std_errors;
	double intercept_std_error;
	idx_t df_residual;

	// Prediction settings
	double confidence_level;
	string interval_type; // 'confidence', 'prediction', or 'none'

	// New observations to predict
	vector<vector<double>> x_new; // [n_new x p] matrix

	// Computed results
	vector<double> predictions;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<double> std_errors;

	idx_t current_row = 0;
};

/**
 * Local state for in-out mode (lateral join support)
 */
struct ModelPredictInOutLocalState : public LocalTableFunctionState {
	idx_t current_input_row = 0;
};

/**
 * Local state initializer for in-out mode
 */
static unique_ptr<LocalTableFunctionState> ModelPredictInOutLocalInit(ExecutionContext &context,
                                                                      TableFunctionInitInput &input,
                                                                      GlobalTableFunctionState *global_state) {
	return make_uniq<ModelPredictInOutLocalState>();
}

/**
 * Compute approximate leverage for a new observation
 * h ≈ 1/n + squared Mahalanobis distance scaled by variance
 *
 * This is an approximation. For exact leverage, we'd need (X'X)⁻¹
 */
static double ComputeApproximateLeverage(const vector<double> &x_new, const vector<double> &x_means,
                                         const vector<double> &coef_std_errors, double mse, idx_t n_train) {
	// Start with baseline leverage
	double h = 1.0 / static_cast<double>(n_train);

	// Add contribution from distance from mean
	// For each feature j: add ((x_j - mean_j) / SE(β_j))²
	// This approximates x'(X'X)⁻¹x using diagonal approximation
	idx_t p = x_new.size();
	for (idx_t j = 0; j < p; j++) {
		if (std::isnan(coef_std_errors[j]) || coef_std_errors[j] == 0.0) {
			continue; // Skip aliased/constant features
		}

		double x_centered = x_new[j] - x_means[j];
		double se_beta = coef_std_errors[j];

		// Variance of β_j is Var(β_j) = MSE * (X'X)⁻¹_jj
		// So (X'X)⁻¹_jj ≈ (SE(β_j))² / MSE
		double xx_inv_jj = (se_beta * se_beta) / mse;

		h += x_centered * x_centered * xx_inv_jj;
	}

	return h;
}

/**
 * Bind function - handles both literal mode and LATERAL join mode
 */
static unique_ptr<FunctionData> ModelPredictBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {

	// Set return types and names first (required for both modes)
	return_types = {LogicalType::BIGINT,  // observation_id
	                LogicalType::DOUBLE,  // predicted
	                LogicalType::DOUBLE,  // ci_lower
	                LogicalType::DOUBLE,  // ci_upper
	                LogicalType::DOUBLE}; // se

	names = {"observation_id", "predicted", "ci_lower", "ci_upper", "se"};

	// Detect if this is literal mode or LATERAL join mode
	// In LATERAL mode, inputs might be empty or contain non-constant values
	// Expected parameters: intercept, coefficients, mse, x_train_means, coefficient_std_errors,
	//                      intercept_std_error, df_residual, x_new, confidence_level, interval_type (10 total)

	// If we have fewer inputs than expected, or if the first input isn't a constant, this is LATERAL mode
	bool is_lateral = false;

	if (input.inputs.size() < 10) {
		// Not enough inputs - likely LATERAL mode where inputs aren't available at bind time
		is_lateral = true;
	} else {
		// Try to detect if this is LATERAL mode by checking if the first input is constant
		try {
			if (input.inputs[0].IsNull()) {
				is_lateral = false; // NULL is a valid literal
			} else {
				// Try to access as a constant double value
				input.inputs[0].GetValue<double>();
				is_lateral = false; // Success means it's a literal
			}
		} catch (...) {
			// Not a constant value - this is LATERAL join mode
			is_lateral = true;
		}
	}

	// In LATERAL mode, return minimal bind data - actual processing happens in in-out function
	if (is_lateral) {
		auto bind_data = make_uniq<ModelPredictBindData>();
		// Return empty bind data - in_out function will handle the actual processing
		return bind_data;
	}

	// Validate we have all required inputs for literal mode
	if (input.inputs.size() < 10) {
		throw InvalidInputException(
		    "model_predict requires 10 parameters: intercept, coefficients, mse, x_train_means, "
		    "coefficient_std_errors, intercept_std_error, df_residual, x_new, confidence_level, interval_type");
	}

	// Literal mode - process the inputs and compute predictions
	auto bind_data = make_uniq<ModelPredictBindData>();

	// Parameter 0: intercept (DOUBLE)
	bind_data->intercept = input.inputs[0].GetValue<double>();

	// Parameter 1: coefficients (DOUBLE[])
	auto &coef_list = ListValue::GetChildren(input.inputs[1]);
	for (auto &val : coef_list) {
		if (val.IsNull()) {
			bind_data->coefficients.push_back(std::numeric_limits<double>::quiet_NaN());
		} else {
			bind_data->coefficients.push_back(val.GetValue<double>());
		}
	}
	idx_t p = bind_data->coefficients.size();

	// Parameter 2: mse (DOUBLE)
	bind_data->mse = input.inputs[2].GetValue<double>();

	// Parameter 3: x_train_means (DOUBLE[])
	auto &means_list = ListValue::GetChildren(input.inputs[3]);
	for (auto &val : means_list) {
		bind_data->x_train_means.push_back(val.GetValue<double>());
	}

	// Parameter 4: coefficient_std_errors (DOUBLE[])
	auto &se_list = ListValue::GetChildren(input.inputs[4]);
	for (auto &val : se_list) {
		if (val.IsNull()) {
			bind_data->coefficient_std_errors.push_back(std::numeric_limits<double>::quiet_NaN());
		} else {
			bind_data->coefficient_std_errors.push_back(val.GetValue<double>());
		}
	}

	// Parameter 5: intercept_std_error (DOUBLE)
	bind_data->intercept_std_error = input.inputs[5].GetValue<double>();

	// Parameter 6: df_residual (BIGINT)
	bind_data->df_residual = input.inputs[6].GetValue<int64_t>();

	// Parameter 7: x_new (DOUBLE[] or DOUBLE[][])
	// Check if it's a 1D array (single feature) or 2D array (multiple features)
	auto &x_new_value = input.inputs[7];
	auto &x_new_outer = ListValue::GetChildren(x_new_value);

	if (x_new_outer.empty()) {
		throw InvalidInputException("x_new cannot be empty");
	}

	// Detect if it's 1D or 2D array
	bool is_2d = x_new_outer[0].type().id() == LogicalTypeId::LIST;

	if (is_2d) {
		// 2D array: [[x11, x12], [x21, x22], ...]
		for (auto &row_val : x_new_outer) {
			auto &row_list = ListValue::GetChildren(row_val);
			vector<double> row;
			for (auto &val : row_list) {
				row.push_back(val.GetValue<double>());
			}
			if (row.size() != p) {
				throw InvalidInputException("x_new row has %llu features but model has %llu", row.size(), p);
			}
			bind_data->x_new.push_back(row);
		}
	} else {
		// 1D array: [x1, x2, x3, ...] - treat as multiple observations of single feature
		if (p != 1) {
			throw InvalidInputException(
			    "Model has %llu features but x_new is 1D array. Use [[x11, x12], ...] for multiple features", p);
		}
		for (auto &val : x_new_outer) {
			vector<double> row = {val.GetValue<double>()};
			bind_data->x_new.push_back(row);
		}
	}

	idx_t n_new = bind_data->x_new.size();

	// Parameter 8: confidence_level (DOUBLE, default 0.95)
	bind_data->confidence_level = 0.95;
	if (input.inputs.size() > 8 && !input.inputs[8].IsNull()) {
		bind_data->confidence_level = input.inputs[8].GetValue<double>();
		if (bind_data->confidence_level <= 0.0 || bind_data->confidence_level >= 1.0) {
			throw InvalidInputException("confidence_level must be between 0 and 1");
		}
	}

	// Parameter 9: interval_type (VARCHAR, default 'prediction')
	bind_data->interval_type = "prediction";
	if (input.inputs.size() > 9 && !input.inputs[9].IsNull()) {
		bind_data->interval_type = input.inputs[9].GetValue<string>();
		if (bind_data->interval_type != "confidence" && bind_data->interval_type != "prediction" &&
		    bind_data->interval_type != "none") {
			throw InvalidInputException("interval_type must be 'confidence', 'prediction', or 'none'");
		}
	}

	// Validation
	if (bind_data->x_train_means.size() != p) {
		throw InvalidInputException("x_train_means has %llu elements but model has %llu features",
		                            bind_data->x_train_means.size(), p);
	}
	if (bind_data->coefficient_std_errors.size() != p) {
		throw InvalidInputException("coefficient_std_errors has %llu elements but model has %llu features",
		                            bind_data->coefficient_std_errors.size(), p);
	}
	if (bind_data->df_residual == 0 && bind_data->interval_type != "none") {
		throw InvalidInputException("Cannot compute intervals with df_residual=0");
	}

	ANOFOX_DEBUG("ModelPredict: n_new=" << n_new << ", p=" << p << ", interval_type=" << bind_data->interval_type
	                                    << ", confidence_level=" << bind_data->confidence_level);

	// Compute predictions and intervals
	bind_data->predictions.resize(n_new);
	bind_data->ci_lowers.resize(n_new);
	bind_data->ci_uppers.resize(n_new);
	bind_data->std_errors.resize(n_new);

	// Get t-critical value if needed
	double t_crit = 0.0;
	if (bind_data->interval_type != "none") {
		double alpha = 1.0 - bind_data->confidence_level;
		t_crit = student_t_critical(alpha / 2.0, static_cast<int>(bind_data->df_residual));
		ANOFOX_DEBUG("t-critical value: " << t_crit << " (df=" << bind_data->df_residual << ")");
	}

	// Approximate n_train from df_residual
	// df_residual = n - p - 1 (with intercept) or n - p (without intercept)
	// Assume with intercept: n ≈ df_residual + p + 1
	idx_t n_train_approx = bind_data->df_residual + p + 1;

	// Compute predictions for each new observation
	for (idx_t i = 0; i < n_new; i++) {
		const auto &x_i = bind_data->x_new[i];

		// Point prediction: ŷ = intercept + Σ(β_j * x_j)
		double y_pred = bind_data->intercept;
		for (idx_t j = 0; j < p; j++) {
			if (!std::isnan(bind_data->coefficients[j])) {
				y_pred += bind_data->coefficients[j] * x_i[j];
			}
		}
		bind_data->predictions[i] = y_pred;

		// Compute intervals if requested
		if (bind_data->interval_type == "none") {
			bind_data->ci_lowers[i] = std::numeric_limits<double>::quiet_NaN();
			bind_data->ci_uppers[i] = std::numeric_limits<double>::quiet_NaN();
			bind_data->std_errors[i] = std::numeric_limits<double>::quiet_NaN();
		} else {
			// Compute approximate leverage
			double h = ComputeApproximateLeverage(x_i, bind_data->x_train_means, bind_data->coefficient_std_errors,
			                                      bind_data->mse, n_train_approx);

			// Compute standard error
			double se;
			if (bind_data->interval_type == "confidence") {
				// Confidence interval for mean: SE(ŷ) = √(MSE * h)
				se = std::sqrt(bind_data->mse * h);
			} else {
				// Prediction interval for individual: SE(pred) = √(MSE * (1 + h))
				se = std::sqrt(bind_data->mse * (1.0 + h));
			}

			bind_data->std_errors[i] = se;
			bind_data->ci_lowers[i] = y_pred - t_crit * se;
			bind_data->ci_uppers[i] = y_pred + t_crit * se;

			ANOFOX_DEBUG("Observation " << i << ": pred=" << y_pred << ", h=" << h << ", se=" << se << ", CI=["
			                            << bind_data->ci_lowers[i] << ", " << bind_data->ci_uppers[i] << "]");
		}
	}

	return std::move(bind_data);
}

/**
 * Execute function - returns one row at a time
 */
static void ModelPredictExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<ModelPredictBindData>();

	idx_t count = 0;
	idx_t n_new = bind_data.predictions.size();

	while (bind_data.current_row < n_new && count < STANDARD_VECTOR_SIZE) {
		idx_t row = bind_data.current_row;

		output.SetValue(0, count, Value::BIGINT(row + 1));                    // observation_id (1-indexed)
		output.SetValue(1, count, Value::DOUBLE(bind_data.predictions[row])); // predicted
		output.SetValue(2, count,
		                std::isnan(bind_data.ci_lowers[row]) ? Value(LogicalType::DOUBLE)
		                                                     : Value::DOUBLE(bind_data.ci_lowers[row])); // ci_lower
		output.SetValue(3, count,
		                std::isnan(bind_data.ci_uppers[row]) ? Value(LogicalType::DOUBLE)
		                                                     : Value::DOUBLE(bind_data.ci_uppers[row])); // ci_upper
		output.SetValue(4, count,
		                std::isnan(bind_data.std_errors[row]) ? Value(LogicalType::DOUBLE)
		                                                      : Value::DOUBLE(bind_data.std_errors[row])); // se

		bind_data.current_row++;
		count++;
	}

	output.SetCardinality(count);
}

/**
 * In-out function for lateral join support
 * Processes rows from input table, generates predictions for each row
 */
static OperatorResultType ModelPredictInOut(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                            DataChunk &output) {
	auto &state = data_p.local_state->Cast<ModelPredictInOutLocalState>();

	// Check if we have input data
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Validate we have enough columns
	if (input.ColumnCount() < 8) {
		throw InvalidInputException("model_predict requires at least 8 input columns, got %d", input.ColumnCount());
	}

	idx_t output_count = 0;

	// Process input rows and generate predictions
	while (state.current_input_row < input.size() && output_count < STANDARD_VECTOR_SIZE) {
		idx_t row = state.current_input_row;

		// Extract model parameters from input row
		// Parameter 0: intercept (DOUBLE)
		double intercept = input.data[0].GetValue(row).GetValue<double>();

		// Parameter 1: coefficients (DOUBLE[])
		// Store the Value to keep it alive while we access its children
		auto coef_value = input.data[1].GetValue(row);
		auto &coef_list = ListValue::GetChildren(coef_value);
		vector<double> coefficients;
		for (auto &val : coef_list) {
			if (val.IsNull()) {
				coefficients.push_back(std::numeric_limits<double>::quiet_NaN());
			} else {
				coefficients.push_back(val.GetValue<double>());
			}
		}
		idx_t p = coefficients.size();

		// Parameter 2: mse (DOUBLE)
		double mse = input.data[2].GetValue(row).GetValue<double>();

		// Parameter 3: x_train_means (DOUBLE[])
		// Store the Value to keep it alive while we access its children
		auto means_value = input.data[3].GetValue(row);
		auto &means_list = ListValue::GetChildren(means_value);
		vector<double> x_train_means;
		for (auto &val : means_list) {
			x_train_means.push_back(val.GetValue<double>());
		}

		// Parameter 4: coefficient_std_errors (DOUBLE[])
		// Store the Value to keep it alive while we access its children
		auto se_value = input.data[4].GetValue(row);
		auto &se_list = ListValue::GetChildren(se_value);
		vector<double> coefficient_std_errors;
		for (auto &val : se_list) {
			if (val.IsNull()) {
				coefficient_std_errors.push_back(std::numeric_limits<double>::quiet_NaN());
			} else {
				coefficient_std_errors.push_back(val.GetValue<double>());
			}
		}

		// Parameter 5: intercept_std_error (DOUBLE) - not used directly in prediction
		// (it's included in the input but not needed for the computation)

		// Parameter 6: df_residual (BIGINT)
		idx_t df_residual = input.data[6].GetValue(row).GetValue<int64_t>();

		// Parameter 7: x_new (DOUBLE[] or DOUBLE[][])
		auto x_new_value = input.data[7].GetValue(row);
		auto &x_new_outer = ListValue::GetChildren(x_new_value);

		if (x_new_outer.empty()) {
			state.current_input_row++;
			continue; // Skip empty x_new
		}

		// Parse x_new (detect 1D vs 2D)
		vector<vector<double>> x_new;
		bool is_2d = x_new_outer[0].type().id() == LogicalTypeId::LIST;

		if (is_2d) {
			// 2D array: [[x11, x12], [x21, x22], ...]
			for (auto &row_val : x_new_outer) {
				auto &row_list = ListValue::GetChildren(row_val);
				vector<double> row_data;
				for (auto &val : row_list) {
					row_data.push_back(val.GetValue<double>());
				}
				x_new.push_back(row_data);
			}
		} else {
			// 1D array: [x1, x2, ...] - single feature for multiple observations
			for (auto &val : x_new_outer) {
				x_new.push_back({val.GetValue<double>()});
			}
		}

		// Parameter 8: confidence_level (DOUBLE, optional)
		double confidence_level = 0.95;
		if (input.ColumnCount() >= 9 && !input.data[8].GetValue(row).IsNull()) {
			confidence_level = input.data[8].GetValue(row).GetValue<double>();
		}

		// Parameter 9: interval_type (VARCHAR, optional)
		string interval_type = "none";
		if (input.ColumnCount() >= 10 && !input.data[9].GetValue(row).IsNull()) {
			interval_type = input.data[9].GetValue(row).GetValue<string>();
		}

		// Compute predictions for all x_new observations
		double t_crit = 0.0;
		if (interval_type != "none") {
			double alpha = 1.0 - confidence_level;
			t_crit = student_t_critical(alpha / 2.0, static_cast<int>(df_residual));
			ANOFOX_DEBUG("ModelPredictInOut: df_residual=" << df_residual << ", alpha=" << alpha
			                                               << ", t_crit=" << t_crit << ", mse=" << mse << ", p=" << p);
		}

		idx_t n_train_approx = df_residual + p + 1;
		ANOFOX_DEBUG("ModelPredictInOut: n_train_approx=" << n_train_approx << ", x_new.size()=" << x_new.size());

		for (idx_t i = 0; i < x_new.size() && output_count < STANDARD_VECTOR_SIZE; i++) {
			const auto &x_i = x_new[i];

			// Point prediction
			double y_pred = intercept;
			for (idx_t j = 0; j < p && j < x_i.size(); j++) {
				if (!std::isnan(coefficients[j])) {
					y_pred += coefficients[j] * x_i[j];
				}
			}

			// Compute intervals
			double ci_lower, ci_upper, se;
			if (interval_type == "none") {
				ci_lower = std::numeric_limits<double>::quiet_NaN();
				ci_upper = std::numeric_limits<double>::quiet_NaN();
				se = std::numeric_limits<double>::quiet_NaN();
			} else {
				double h = ComputeApproximateLeverage(x_i, x_train_means, coefficient_std_errors, mse, n_train_approx);

				if (interval_type == "confidence") {
					se = std::sqrt(mse * h);
				} else {
					se = std::sqrt(mse * (1.0 + h));
				}

				ANOFOX_DEBUG("  obs " << i << ": x=" << x_i[0] << ", y_pred=" << y_pred << ", h=" << h
				                      << ", se=" << se);

				ci_lower = y_pred - t_crit * se;
				ci_upper = y_pred + t_crit * se;
			}

			// Write output row
			output.SetValue(0, output_count, Value::BIGINT(i + 1));  // observation_id
			output.SetValue(1, output_count, Value::DOUBLE(y_pred)); // predicted
			output.SetValue(2, output_count,
			                std::isnan(ci_lower) ? Value(LogicalType::DOUBLE) : Value::DOUBLE(ci_lower)); // ci_lower
			output.SetValue(3, output_count,
			                std::isnan(ci_upper) ? Value(LogicalType::DOUBLE) : Value::DOUBLE(ci_upper)); // ci_upper
			output.SetValue(4, output_count, std::isnan(se) ? Value(LogicalType::DOUBLE) : Value::DOUBLE(se)); // se

			output_count++;
		}

		state.current_input_row++;
	}

	output.SetCardinality(output_count);

	// Determine return value based on whether we have more work to do
	if (state.current_input_row < input.size()) {
		// We have more input rows to process in this batch
		return OperatorResultType::HAVE_MORE_OUTPUT;
	} else {
		// We've finished processing all input rows in this batch
		state.current_input_row = 0; // Reset for next input batch
		return OperatorResultType::NEED_MORE_INPUT;
	}
}

/**
 * Register the function
 */
void AnofoxStatisticsModelPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_model_predict function (with LATERAL join support)");

	TableFunction func("anofox_statistics_model_predict",
	                   {LogicalType::DOUBLE,                    // intercept
	                    LogicalType::LIST(LogicalType::DOUBLE), // coefficients
	                    LogicalType::DOUBLE,                    // mse
	                    LogicalType::LIST(LogicalType::DOUBLE), // x_train_means
	                    LogicalType::LIST(LogicalType::DOUBLE), // coefficient_std_errors
	                    LogicalType::DOUBLE,                    // intercept_std_error
	                    LogicalType::BIGINT,                    // df_residual
	                    LogicalType::ANY,                       // x_new (DOUBLE[] or DOUBLE[][])
	                    LogicalType::DOUBLE,                    // confidence_level (optional)
	                    LogicalType::VARCHAR},                  // interval_type (optional)
	                   ModelPredictExecute, ModelPredictBind, nullptr, ModelPredictInOutLocalInit);

	// Add LATERAL join support (in_out_function)
	func.in_out_function = ModelPredictInOut;

	func.named_parameters["intercept"] = LogicalType::DOUBLE;
	func.named_parameters["coefficients"] = LogicalType::LIST(LogicalType::DOUBLE);
	func.named_parameters["mse"] = LogicalType::DOUBLE;
	func.named_parameters["x_train_means"] = LogicalType::LIST(LogicalType::DOUBLE);
	func.named_parameters["coefficient_std_errors"] = LogicalType::LIST(LogicalType::DOUBLE);
	func.named_parameters["intercept_std_error"] = LogicalType::DOUBLE;
	func.named_parameters["df_residual"] = LogicalType::BIGINT;
	func.named_parameters["x_new"] = LogicalType::ANY;
	func.named_parameters["confidence_level"] = LogicalType::DOUBLE;
	func.named_parameters["interval_type"] = LogicalType::VARCHAR;

	loader.RegisterFunction(func);

	ANOFOX_DEBUG("anofox_statistics_model_predict function registered successfully (both literal and LATERAL modes)");
}

} // namespace anofox_statistics
} // namespace duckdb
