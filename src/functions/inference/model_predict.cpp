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
	string interval_type;  // 'confidence', 'prediction', or 'none'

	// New observations to predict
	vector<vector<double>> x_new;  // [n_new x p] matrix

	// Computed results
	vector<double> predictions;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<double> std_errors;

	idx_t current_row = 0;
};

/**
 * Compute approximate leverage for a new observation
 * h ≈ 1/n + squared Mahalanobis distance scaled by variance
 *
 * This is an approximation. For exact leverage, we'd need (X'X)⁻¹
 */
static double ComputeApproximateLeverage(const vector<double> &x_new,
                                         const vector<double> &x_means,
                                         const vector<double> &coef_std_errors,
                                         double mse,
                                         idx_t n_train) {
	// Start with baseline leverage
	double h = 1.0 / static_cast<double>(n_train);

	// Add contribution from distance from mean
	// For each feature j: add ((x_j - mean_j) / SE(β_j))²
	// This approximates x'(X'X)⁻¹x using diagonal approximation
	idx_t p = x_new.size();
	for (idx_t j = 0; j < p; j++) {
		if (std::isnan(coef_std_errors[j]) || coef_std_errors[j] == 0.0) {
			continue;  // Skip aliased/constant features
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
 * Bind function
 */
static unique_ptr<FunctionData> ModelPredictBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {

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
			throw InvalidInputException("Model has %llu features but x_new is 1D array. Use [[x11, x12], ...] for multiple features", p);
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
		if (bind_data->interval_type != "confidence" &&
		    bind_data->interval_type != "prediction" &&
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

	ANOFOX_DEBUG("ModelPredict: n_new=" << n_new << ", p=" << p
	            << ", interval_type=" << bind_data->interval_type
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
			double h = ComputeApproximateLeverage(x_i, bind_data->x_train_means,
			                                     bind_data->coefficient_std_errors,
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

			ANOFOX_DEBUG("Observation " << i << ": pred=" << y_pred
			            << ", h=" << h << ", se=" << se
			            << ", CI=[" << bind_data->ci_lowers[i] << ", " << bind_data->ci_uppers[i] << "]");
		}
	}

	// Define return types and names
	return_types = {LogicalType::BIGINT,    // observation_id
	                LogicalType::DOUBLE,    // predicted
	                LogicalType::DOUBLE,    // ci_lower
	                LogicalType::DOUBLE,    // ci_upper
	                LogicalType::DOUBLE};   // se

	names = {"observation_id", "predicted", "ci_lower", "ci_upper", "se"};

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

		output.SetValue(0, count, Value::BIGINT(row + 1));                       // observation_id (1-indexed)
		output.SetValue(1, count, Value::DOUBLE(bind_data.predictions[row]));   // predicted
		output.SetValue(2, count, Value::DOUBLE(bind_data.ci_lowers[row]));     // ci_lower
		output.SetValue(3, count, Value::DOUBLE(bind_data.ci_uppers[row]));     // ci_upper
		output.SetValue(4, count, Value::DOUBLE(bind_data.std_errors[row]));    // se

		bind_data.current_row++;
		count++;
	}

	output.SetCardinality(count);
}

/**
 * Register the function
 */
void AnofoxStatisticsModelPredictFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_model_predict function");

	TableFunction func("anofox_statistics_model_predict",
	                   {LogicalType::DOUBLE,                // intercept
	                    LogicalType::LIST(LogicalType::DOUBLE),  // coefficients
	                    LogicalType::DOUBLE,                // mse
	                    LogicalType::LIST(LogicalType::DOUBLE),  // x_train_means
	                    LogicalType::LIST(LogicalType::DOUBLE),  // coefficient_std_errors
	                    LogicalType::DOUBLE,                // intercept_std_error
	                    LogicalType::BIGINT,                // df_residual
	                    LogicalType::ANY,                   // x_new (DOUBLE[] or DOUBLE[][])
	                    LogicalType::DOUBLE,                // confidence_level (optional)
	                    LogicalType::VARCHAR},              // interval_type (optional)
	                   ModelPredictExecute, ModelPredictBind);

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

	ANOFOX_DEBUG("anofox_statistics_model_predict function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
