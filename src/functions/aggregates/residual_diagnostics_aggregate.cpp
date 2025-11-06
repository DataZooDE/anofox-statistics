#include "residual_diagnostics_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

namespace duckdb {
namespace anofox_statistics {

/**
 * Residual Diagnostics Aggregate
 * Computes diagnostic statistics for residuals from grouped data
 *
 * Input: y_actual DOUBLE, y_predicted DOUBLE, options MAP
 * Options:
 *   - 'detailed': BOOLEAN (default FALSE) - return arrays vs summary
 *   - 'outlier_threshold': DOUBLE (default 2.5) - threshold for outlier detection
 *
 * Summary output: n_obs, n_outliers, max_abs_residual, mean_abs_residual, rmse
 * Detailed output: residuals[], std_residuals[], is_outlier[]
 */

struct ResidualDiagnosticsOptions {
	bool detailed = false;
	double outlier_threshold = 2.5;

	static ResidualDiagnosticsOptions ParseFromMap(const Value &options_map) {
		ResidualDiagnosticsOptions opts;
		if (options_map.IsNull() || options_map.type().id() != LogicalTypeId::MAP) {
			return opts;
		}

		auto &map_children = MapValue::GetChildren(options_map);
		for (idx_t i = 0; i < map_children.size(); i++) {
			auto &key_val = map_children[i];
			auto key_list = ListValue::GetChildren(key_val);
			if (key_list.size() != 2) continue;

			string key = key_list[0].ToString();
			auto &value = key_list[1];

			if (key == "detailed") {
				opts.detailed = value.GetValue<bool>();
			} else if (key == "outlier_threshold") {
				opts.outlier_threshold = value.GetValue<double>();
			}
		}
		return opts;
	}
};

struct ResidualDiagnosticsAggregateState {
	vector<double> residuals;
	ResidualDiagnosticsOptions options;
	bool options_initialized = false;

	void Reset() {
		residuals.clear();
		options = ResidualDiagnosticsOptions();
		options_initialized = false;
	}
};

static void ResidualDiagnosticsInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<ResidualDiagnosticsAggregateState *>(state_ptr);
	new (state) ResidualDiagnosticsAggregateState();
}

static void ResidualDiagnosticsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                       Vector &state_vector, idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<ResidualDiagnosticsAggregateState *>(state_data);

	auto &y_actual_vector = inputs[0];
	auto &y_predicted_vector = inputs[1];
	auto &options_vector = inputs[2];

	UnifiedVectorFormat y_actual_data, y_predicted_data, options_data;
	y_actual_vector.ToUnifiedFormat(count, y_actual_data);
	y_predicted_vector.ToUnifiedFormat(count, y_predicted_data);
	options_vector.ToUnifiedFormat(count, options_data);

	auto y_actual_ptr = UnifiedVectorFormat::GetData<double>(y_actual_data);
	auto y_predicted_ptr = UnifiedVectorFormat::GetData<double>(y_predicted_data);

	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_actual_idx = y_actual_data.sel->get_index(i);
		auto y_predicted_idx = y_predicted_data.sel->get_index(i);
		auto options_idx = options_data.sel->get_index(i);

		if (!y_actual_data.validity.RowIsValid(y_actual_idx) || !y_predicted_data.validity.RowIsValid(y_predicted_idx)) {
			continue;
		}

		// Initialize options from first row
		if (!state.options_initialized && options_data.validity.RowIsValid(options_idx)) {
			auto options_value = options_vector.GetValue(options_idx);
			state.options = ResidualDiagnosticsOptions::ParseFromMap(options_value);
			state.options_initialized = true;
		}

		double residual = y_actual_ptr[y_actual_idx] - y_predicted_ptr[y_predicted_idx];
		state.residuals.push_back(residual);
	}
}

static void ResidualDiagnosticsCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<ResidualDiagnosticsAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<ResidualDiagnosticsAggregateState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

		target_state.residuals.insert(target_state.residuals.end(), source_state.residuals.begin(),
		                              source_state.residuals.end());

		if (!target_state.options_initialized && source_state.options_initialized) {
			target_state.options = source_state.options;
			target_state.options_initialized = true;
		}
	}
}

static void ResidualDiagnosticsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                         idx_t count, idx_t offset) {

	auto states = FlatVector::GetData<ResidualDiagnosticsAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	auto &struct_entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.residuals.size();
		if (n == 0) {
			result_validity.SetInvalid(result_idx);
			continue;
		}

		// Compute statistics
		double mean_residual = 0.0;
		for (idx_t j = 0; j < n; j++) {
			mean_residual += state.residuals[j];
		}
		mean_residual /= static_cast<double>(n);

		double variance = 0.0;
		for (idx_t j = 0; j < n; j++) {
			double diff = state.residuals[j] - mean_residual;
			variance += diff * diff;
		}
		double sd = std::sqrt(variance / static_cast<double>(n - 1));

		// Compute standardized residuals and outliers
		vector<double> std_residuals(n);
		vector<bool> is_outlier(n);
		idx_t n_outliers = 0;
		double max_abs_residual = 0.0;
		double sum_abs_residual = 0.0;

		for (idx_t j = 0; j < n; j++) {
			double abs_residual = std::abs(state.residuals[j]);
			if (abs_residual > max_abs_residual) {
				max_abs_residual = abs_residual;
			}
			sum_abs_residual += abs_residual;

			if (sd > 1e-10) {
				std_residuals[j] = (state.residuals[j] - mean_residual) / sd;
				is_outlier[j] = std::abs(std_residuals[j]) > state.options.outlier_threshold;
				if (is_outlier[j]) {
					n_outliers++;
				}
			} else {
				std_residuals[j] = 0.0;
				is_outlier[j] = false;
			}
		}

		double mean_abs_residual = sum_abs_residual / static_cast<double>(n);
		double rmse = std::sqrt(variance * static_cast<double>(n - 1) / static_cast<double>(n));

		if (state.options.detailed) {
			// Detailed output: residuals[], std_residuals[], is_outlier[], summary stats
			auto &residuals_list = *struct_entries[0];
			auto &std_residuals_list = *struct_entries[1];
			auto &is_outlier_list = *struct_entries[2];
			auto n_obs_data = FlatVector::GetData<int64_t>(*struct_entries[3]);
			auto n_outliers_data = FlatVector::GetData<int64_t>(*struct_entries[4]);

			// Store arrays
			auto residuals_entries = FlatVector::GetData<list_entry_t>(residuals_list);
			auto &residuals_child = ListVector::GetEntry(residuals_list);
			ListVector::Reserve(residuals_list, (result_idx + 1) * n);
			auto residuals_child_data = FlatVector::GetData<double>(residuals_child);

			auto std_residuals_entries = FlatVector::GetData<list_entry_t>(std_residuals_list);
			auto &std_residuals_child = ListVector::GetEntry(std_residuals_list);
			ListVector::Reserve(std_residuals_list, (result_idx + 1) * n);
			auto std_residuals_child_data = FlatVector::GetData<double>(std_residuals_child);

			auto is_outlier_entries = FlatVector::GetData<list_entry_t>(is_outlier_list);
			auto &is_outlier_child = ListVector::GetEntry(is_outlier_list);
			ListVector::Reserve(is_outlier_list, (result_idx + 1) * n);
			auto is_outlier_child_data = FlatVector::GetData<bool>(is_outlier_child);

			idx_t list_offset = result_idx * n;
			for (idx_t j = 0; j < n; j++) {
				residuals_child_data[list_offset + j] = state.residuals[j];
				std_residuals_child_data[list_offset + j] = std_residuals[j];
				is_outlier_child_data[list_offset + j] = is_outlier[j];
			}

			residuals_entries[result_idx] = list_entry_t{list_offset, n};
			std_residuals_entries[result_idx] = list_entry_t{list_offset, n};
			is_outlier_entries[result_idx] = list_entry_t{list_offset, n};
			n_obs_data[result_idx] = static_cast<int64_t>(n);
			n_outliers_data[result_idx] = static_cast<int64_t>(n_outliers);

			ListVector::SetListSize(residuals_list, (result_idx + 1) * n);
			ListVector::SetListSize(std_residuals_list, (result_idx + 1) * n);
			ListVector::SetListSize(is_outlier_list, (result_idx + 1) * n);
		} else {
			// Summary output: n_obs, n_outliers, max_abs_residual, mean_abs_residual, rmse
			auto n_obs_data = FlatVector::GetData<int64_t>(*struct_entries[0]);
			auto n_outliers_data = FlatVector::GetData<int64_t>(*struct_entries[1]);
			auto max_abs_data = FlatVector::GetData<double>(*struct_entries[2]);
			auto mean_abs_data = FlatVector::GetData<double>(*struct_entries[3]);
			auto rmse_data = FlatVector::GetData<double>(*struct_entries[4]);

			n_obs_data[result_idx] = static_cast<int64_t>(n);
			n_outliers_data[result_idx] = static_cast<int64_t>(n_outliers);
			max_abs_data[result_idx] = max_abs_residual;
			mean_abs_data[result_idx] = mean_abs_residual;
			rmse_data[result_idx] = rmse;
		}

		ANOFOX_DEBUG("Residual diagnostics aggregate: n=" << n << ", outliers=" << n_outliers << ", rmse=" << rmse);
	}
}

void ResidualDiagnosticsAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering Residual Diagnostics aggregate function");

	// Summary version (default)
	child_list_t<LogicalType> summary_struct_fields;
	summary_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	summary_struct_fields.push_back(make_pair("n_outliers", LogicalType::BIGINT));
	summary_struct_fields.push_back(make_pair("max_abs_residual", LogicalType::DOUBLE));
	summary_struct_fields.push_back(make_pair("mean_abs_residual", LogicalType::DOUBLE));
	summary_struct_fields.push_back(make_pair("rmse", LogicalType::DOUBLE));

	// Detailed version (when detailed=true)
	child_list_t<LogicalType> detailed_struct_fields;
	detailed_struct_fields.push_back(make_pair("residuals", LogicalType::LIST(LogicalType::DOUBLE)));
	detailed_struct_fields.push_back(make_pair("std_residuals", LogicalType::LIST(LogicalType::DOUBLE)));
	detailed_struct_fields.push_back(make_pair("is_outlier", LogicalType::LIST(LogicalType::BOOLEAN)));
	detailed_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	detailed_struct_fields.push_back(make_pair("n_outliers", LogicalType::BIGINT));

	// Register with summary output type (user needs to use MAP{'detailed': true} for detailed)
	AggregateFunction anofox_statistics_residual_diagnostics_agg(
	    "anofox_statistics_residual_diagnostics_agg",
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY}, // y_actual, y_predicted, options
	    LogicalType::STRUCT(summary_struct_fields), AggregateFunction::StateSize<ResidualDiagnosticsAggregateState>,
	    ResidualDiagnosticsInitialize, ResidualDiagnosticsUpdate, ResidualDiagnosticsCombine,
	    ResidualDiagnosticsFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING);

	loader.RegisterFunction(anofox_statistics_residual_diagnostics_agg);

	ANOFOX_DEBUG("Residual Diagnostics aggregate function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
