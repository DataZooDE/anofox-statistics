#include "predict_scalar.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "../utils/statistical_distributions.hpp"
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * Helper function to compute prediction with intervals
 */
struct PredictionResult {
	double predicted;
	double ci_lower;
	double ci_upper;
	double std_error;
};

static PredictionResult ComputePrediction(double intercept, const vector<double> &coefficients, double mse,
                                          const vector<double> &x_train_means,
                                          const vector<double> &coefficient_std_errors, double intercept_std_error,
                                          idx_t df_residual, const vector<double> &x_new, double confidence_level,
                                          const string &interval_type) {

	PredictionResult result;

	// Compute prediction: ŷ = β₀ + β₁x₁ + β₂x₂ + ...
	result.predicted = intercept;
	for (idx_t i = 0; i < coefficients.size(); i++) {
		result.predicted += coefficients[i] * x_new[i];
	}

	// If no intervals requested or insufficient data
	if (interval_type == "none" || df_residual == 0 || std::isnan(mse)) {
		result.ci_lower = result.predicted;
		result.ci_upper = result.predicted;
		result.std_error = 0.0;
		return result;
	}

	// Compute approximate leverage: h ≈ 1/n + distance from training mean
	idx_t n_train = df_residual + (intercept_std_error > 0 ? 1 : 0) + coefficients.size();
	double h = 1.0 / static_cast<double>(n_train);

	for (idx_t j = 0; j < x_new.size(); j++) {
		if (!std::isnan(coefficient_std_errors[j]) && coefficient_std_errors[j] > 0) {
			double x_centered = x_new[j] - x_train_means[j];
			double se_beta = coefficient_std_errors[j];
			double xx_inv_jj = (se_beta * se_beta) / mse;
			h += x_centered * x_centered * xx_inv_jj;
		}
	}

	// Compute standard error
	if (interval_type == "confidence") {
		// SE for mean prediction
		result.std_error = std::sqrt(mse * h);
	} else {
		// SE for individual prediction
		result.std_error = std::sqrt(mse * (1.0 + h));
	}

	// Compute confidence/prediction interval
	double alpha = 1.0 - confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, static_cast<int>(df_residual));
	double margin = t_crit * result.std_error;

	result.ci_lower = result.predicted - margin;
	result.ci_upper = result.predicted + margin;

	return result;
}

/**
 * anofox_statistics_predict_simple(model_struct, x_new)
 * Returns LIST<DOUBLE> with just predicted values
 */
static void PredictSimpleFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);

	auto &model_vector = args.data[0];
	auto &x_new_vector = args.data[1];

	auto count = args.size();

	// Result is LIST<DOUBLE>
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &list_child = ListVector::GetEntry(result);
	ListVector::Reserve(result, count * 10); // Reserve some space

	idx_t list_offset = 0;

	UnifiedVectorFormat model_data;
	UnifiedVectorFormat x_new_data;
	model_vector.ToUnifiedFormat(count, model_data);
	x_new_vector.ToUnifiedFormat(count, x_new_data);

	auto &result_validity = FlatVector::Validity(result);
	auto pred_data = FlatVector::GetData<double>(list_child);

	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_data.sel->get_index(i);
		auto x_new_idx = x_new_data.sel->get_index(i);

		if (!model_data.validity.RowIsValid(model_idx) || !x_new_data.validity.RowIsValid(x_new_idx)) {
			result_validity.SetInvalid(i);
			list_entries[i] = list_entry_t {list_offset, 0};
			continue;
		}

		// Extract model struct fields
		auto &model_children = StructVector::GetEntries(model_vector);
		idx_t struct_idx = model_idx;

		// Get intercept (field 1)
		auto intercept_vec = model_children[1].get();
		auto intercept_data_ptr = FlatVector::GetData<double>(*intercept_vec);
		double intercept = intercept_data_ptr[struct_idx];

		// Get coefficients (field 0)
		auto coef_vec = model_children[0].get();
		auto coef_list_data = FlatVector::GetData<list_entry_t>(*coef_vec);
		auto &coef_child = ListVector::GetEntry(*coef_vec);
		auto coef_values = FlatVector::GetData<double>(coef_child);
		auto coef_entry = coef_list_data[struct_idx];

		vector<double> coefficients;
		for (idx_t j = 0; j < coef_entry.length; j++) {
			coefficients.push_back(coef_values[coef_entry.offset + j]);
		}

		// Get x_new (2D array)
		auto x_new_list_data = FlatVector::GetData<list_entry_t>(x_new_vector);
		auto &x_new_outer_child = ListVector::GetEntry(x_new_vector);
		auto x_new_outer_list_data = FlatVector::GetData<list_entry_t>(x_new_outer_child);
		auto &x_new_inner_child = ListVector::GetEntry(x_new_outer_child);
		auto x_new_values = FlatVector::GetData<double>(x_new_inner_child);

		auto x_new_entry = x_new_list_data[x_new_idx];

		// For each observation in x_new
		idx_t start_offset = list_offset;
		for (idx_t obs = 0; obs < x_new_entry.length; obs++) {
			auto inner_entry = x_new_outer_list_data[x_new_entry.offset + obs];

			// Extract this observation's features
			vector<double> x_obs;
			for (idx_t j = 0; j < inner_entry.length; j++) {
				x_obs.push_back(x_new_values[inner_entry.offset + j]);
			}

			// Compute prediction: ŷ = β₀ + β₁x₁ + β₂x₂ + ...
			double predicted = intercept;
			for (idx_t j = 0; j < coefficients.size() && j < x_obs.size(); j++) {
				predicted += coefficients[j] * x_obs[j];
			}

			pred_data[list_offset++] = predicted;
		}

		list_entries[i] = list_entry_t {start_offset, x_new_entry.length};
	}

	ListVector::SetListSize(list_child, list_offset);
}

/**
 * anofox_statistics_predict(model_struct, x_new, confidence_level, interval_type)
 * Returns LIST<STRUCT> with full prediction info
 */
static void PredictFullFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 4);

	auto &model_vector = args.data[0];
	auto &x_new_vector = args.data[1];
	auto &conf_level_vector = args.data[2];
	auto &interval_type_vector = args.data[3];

	auto count = args.size();

	// Result is LIST<STRUCT(predicted DOUBLE, ci_lower DOUBLE, ci_upper DOUBLE, std_error DOUBLE)>
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &list_child = ListVector::GetEntry(result);
	ListVector::Reserve(result, count * 10);

	auto &struct_entries = StructVector::GetEntries(list_child);
	auto pred_data = FlatVector::GetData<double>(*struct_entries[0]);
	auto ci_lower_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto ci_upper_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto se_data = FlatVector::GetData<double>(*struct_entries[3]);

	idx_t list_offset = 0;

	UnifiedVectorFormat model_data, x_new_data, conf_data, interval_data;
	model_vector.ToUnifiedFormat(count, model_data);
	x_new_vector.ToUnifiedFormat(count, x_new_data);
	conf_level_vector.ToUnifiedFormat(count, conf_data);
	interval_type_vector.ToUnifiedFormat(count, interval_data);

	auto conf_values = UnifiedVectorFormat::GetData<double>(conf_data);
	auto interval_values = UnifiedVectorFormat::GetData<string_t>(interval_data);

	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_data.sel->get_index(i);
		auto x_new_idx = x_new_data.sel->get_index(i);
		auto conf_idx = conf_data.sel->get_index(i);
		auto interval_idx = interval_data.sel->get_index(i);

		if (!model_data.validity.RowIsValid(model_idx) || !x_new_data.validity.RowIsValid(x_new_idx)) {
			result_validity.SetInvalid(i);
			list_entries[i] = list_entry_t {list_offset, 0};
			continue;
		}

		double confidence_level = conf_values[conf_idx];
		string interval_type = interval_values[interval_idx].GetString();

		// Extract model struct fields
		auto &model_children = StructVector::GetEntries(model_vector);
		idx_t struct_idx = model_idx;

		// Field indices: coefficients=0, intercept=1, r2=2, adj_r2=3, n_obs=4, mse=5,
		//                x_train_means=6, coefficient_std_errors=7, intercept_std_error=8, df_residual=9
		auto intercept_vec = model_children[1].get();
		auto mse_vec = model_children[5].get();
		auto intercept_se_vec = model_children[8].get();
		auto df_resid_vec = model_children[9].get();

		double intercept = FlatVector::GetData<double>(*intercept_vec)[struct_idx];
		double mse = FlatVector::GetData<double>(*mse_vec)[struct_idx];
		double intercept_se = FlatVector::GetData<double>(*intercept_se_vec)[struct_idx];
		idx_t df_residual = FlatVector::GetData<int64_t>(*df_resid_vec)[struct_idx];

		// Get coefficients
		auto coef_vec = model_children[0].get();
		auto coef_list_data = FlatVector::GetData<list_entry_t>(*coef_vec);
		auto &coef_child = ListVector::GetEntry(*coef_vec);
		auto coef_values = FlatVector::GetData<double>(coef_child);
		auto coef_entry = coef_list_data[struct_idx];

		vector<double> coefficients;
		for (idx_t j = 0; j < coef_entry.length; j++) {
			coefficients.push_back(coef_values[coef_entry.offset + j]);
		}

		// Get x_train_means
		auto x_means_vec = model_children[6].get();
		auto x_means_list_data = FlatVector::GetData<list_entry_t>(*x_means_vec);
		auto &x_means_child = ListVector::GetEntry(*x_means_vec);
		auto x_means_values = FlatVector::GetData<double>(x_means_child);
		auto x_means_entry = x_means_list_data[struct_idx];

		vector<double> x_train_means;
		for (idx_t j = 0; j < x_means_entry.length; j++) {
			x_train_means.push_back(x_means_values[x_means_entry.offset + j]);
		}

		// Get coefficient_std_errors
		auto coef_se_vec = model_children[7].get();
		auto coef_se_list_data = FlatVector::GetData<list_entry_t>(*coef_se_vec);
		auto &coef_se_child = ListVector::GetEntry(*coef_se_vec);
		auto coef_se_values = FlatVector::GetData<double>(coef_se_child);
		auto &coef_se_validity = FlatVector::Validity(coef_se_child);
		auto coef_se_entry = coef_se_list_data[struct_idx];

		vector<double> coefficient_std_errors;
		for (idx_t j = 0; j < coef_se_entry.length; j++) {
			idx_t idx = coef_se_entry.offset + j;
			if (coef_se_validity.RowIsValid(idx)) {
				coefficient_std_errors.push_back(coef_se_values[idx]);
			} else {
				coefficient_std_errors.push_back(std::numeric_limits<double>::quiet_NaN());
			}
		}

		// Get x_new (2D array)
		auto x_new_list_data = FlatVector::GetData<list_entry_t>(x_new_vector);
		auto &x_new_outer_child = ListVector::GetEntry(x_new_vector);
		auto x_new_outer_list_data = FlatVector::GetData<list_entry_t>(x_new_outer_child);
		auto &x_new_inner_child = ListVector::GetEntry(x_new_outer_child);
		auto x_new_values = FlatVector::GetData<double>(x_new_inner_child);

		auto x_new_entry = x_new_list_data[x_new_idx];

		// For each observation in x_new
		idx_t start_offset = list_offset;
		for (idx_t obs = 0; obs < x_new_entry.length; obs++) {
			auto inner_entry = x_new_outer_list_data[x_new_entry.offset + obs];

			// Extract this observation's features
			vector<double> x_obs;
			for (idx_t j = 0; j < inner_entry.length; j++) {
				x_obs.push_back(x_new_values[inner_entry.offset + j]);
			}

			// Compute prediction with intervals
			PredictionResult pred =
			    ComputePrediction(intercept, coefficients, mse, x_train_means, coefficient_std_errors, intercept_se,
			                      df_residual, x_obs, confidence_level, interval_type);

			pred_data[list_offset] = pred.predicted;
			ci_lower_data[list_offset] = pred.ci_lower;
			ci_upper_data[list_offset] = pred.ci_upper;
			se_data[list_offset] = pred.std_error;
			list_offset++;
		}

		list_entries[i] = list_entry_t {start_offset, x_new_entry.length};
	}

	ListVector::SetListSize(list_child, list_offset);
}

void PredictScalarFunctions::Register(ExtensionLoader &loader) {
	// anofox_statistics_predict_simple(model_struct, x_new) -> LIST<DOUBLE>
	ScalarFunction predict_simple("anofox_statistics_predict_simple",
	                              {LogicalType::ANY, LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
	                              LogicalType::LIST(LogicalType::DOUBLE), PredictSimpleFunction);
	loader.RegisterFunction(predict_simple);

	// anofox_statistics_predict(model_struct, x_new, conf_level, interval_type) -> LIST<STRUCT>
	child_list_t<LogicalType> pred_struct_fields;
	pred_struct_fields.push_back(make_pair("predicted", LogicalType::DOUBLE));
	pred_struct_fields.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
	pred_struct_fields.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
	pred_struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));

	ScalarFunction predict_full("anofox_statistics_predict",
	                            {LogicalType::ANY, LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
	                             LogicalType::DOUBLE, LogicalType::VARCHAR},
	                            LogicalType::LIST(LogicalType::STRUCT(pred_struct_fields)), PredictFullFunction);
	loader.RegisterFunction(predict_full);
}

} // namespace anofox_statistics
} // namespace duckdb
