#include "normality_test_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/statistical_distributions.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <vector>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * Normality Test Aggregate: anofox_statistics_normality_test_agg(residual DOUBLE, options MAP) -> STRUCT
 *
 * Accumulates residuals across rows in a GROUP BY,
 * then performs Jarque-Bera normality test on finalize.
 *
 * Options MAP:
 *   - 'alpha': DOUBLE (default 0.05) - significance level
 */

struct NormalityTestOptions {
	double alpha = 0.05;

	static NormalityTestOptions ParseFromMap(const Value &options_map) {
		NormalityTestOptions opts;
		if (options_map.IsNull() || options_map.type().id() != LogicalTypeId::MAP) {
			return opts;
		}

		auto &map_children = MapValue::GetChildren(options_map);
		for (idx_t i = 0; i < map_children.size(); i++) {
			auto &key_val = map_children[i];
			auto key_list = ListValue::GetChildren(key_val);
			if (key_list.size() != 2) {
				continue;
			}

			string key = key_list[0].ToString();
			auto &value = key_list[1];

			if (key == "alpha") {
				opts.alpha = value.GetValue<double>();
			}
		}
		return opts;
	}
};

struct NormalityTestAggregateState {
	vector<double> residuals;
	NormalityTestOptions options;
	bool options_initialized = false;

	void Reset() {
		residuals.clear();
		options = NormalityTestOptions();
		options_initialized = false;
	}
};

static void NormalityTestInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<NormalityTestAggregateState *>(state_ptr);
	new (state) NormalityTestAggregateState();
}

static void NormalityTestUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<NormalityTestAggregateState *>(state_data);

	auto &residual_vector = inputs[0];
	auto &options_vector = inputs[1];

	UnifiedVectorFormat residual_data, options_data;
	residual_vector.ToUnifiedFormat(count, residual_data);
	options_vector.ToUnifiedFormat(count, options_data);

	auto residual_ptr = UnifiedVectorFormat::GetData<double>(residual_data);

	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto residual_idx = residual_data.sel->get_index(i);
		auto options_idx = options_data.sel->get_index(i);

		if (!residual_data.validity.RowIsValid(residual_idx)) {
			continue;
		}

		// Initialize options from first row
		if (!state.options_initialized && options_data.validity.RowIsValid(options_idx)) {
			auto options_value = options_vector.GetValue(options_idx);
			state.options = NormalityTestOptions::ParseFromMap(options_value);
			state.options_initialized = true;
		}

		state.residuals.push_back(residual_ptr[residual_idx]);
	}
}

static void NormalityTestCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<NormalityTestAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<NormalityTestAggregateState *>(target);

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

static void NormalityTestFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {

	auto states = FlatVector::GetData<NormalityTestAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result: STRUCT(n_obs, skewness, kurtosis, jb_statistic, p_value, is_normal, conclusion)
	auto &struct_entries = StructVector::GetEntries(result);
	auto n_obs_data = FlatVector::GetData<int64_t>(*struct_entries[0]);
	auto skewness_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto kurtosis_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto jb_statistic_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto p_value_data = FlatVector::GetData<double>(*struct_entries[4]);
	auto is_normal_data = FlatVector::GetData<bool>(*struct_entries[5]);
	auto &conclusion_vector = *struct_entries[6];

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.residuals.size();
		if (n < 8) { // Need at least 8 observations for reliable JB test
			result_validity.SetInvalid(result_idx);
			continue;
		}

		// Compute moments
		double mean = 0.0;
		for (idx_t j = 0; j < n; j++) {
			mean += state.residuals[j];
		}
		mean /= static_cast<double>(n);

		double m2 = 0.0, m3 = 0.0, m4 = 0.0;
		for (idx_t j = 0; j < n; j++) {
			double diff = state.residuals[j] - mean;
			double diff2 = diff * diff;
			m2 += diff2;
			m3 += diff * diff2;
			m4 += diff2 * diff2;
		}
		m2 /= static_cast<double>(n);
		m3 /= static_cast<double>(n);
		m4 /= static_cast<double>(n);

		double variance = m2;
		double std_dev = std::sqrt(variance);

		// Sample skewness and kurtosis
		double skewness = m3 / (std_dev * std_dev * std_dev);
		double kurtosis = (m4 / (variance * variance)) - 3.0; // Excess kurtosis

		// Jarque-Bera test statistic: JB = n/6 * (S² + K²/4)
		double jb_statistic = static_cast<double>(n) / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0);

		// P-value from chi-squared distribution with df=2
		double p_value = ChiSquaredCDF::ComplementaryCDF(jb_statistic, 2.0);

		// Test result
		bool is_normal = p_value > state.options.alpha;
		string conclusion = is_normal ? "normal" : "non-normal";

		n_obs_data[result_idx] = static_cast<int64_t>(n);
		skewness_data[result_idx] = skewness;
		kurtosis_data[result_idx] = kurtosis;
		jb_statistic_data[result_idx] = jb_statistic;
		p_value_data[result_idx] = p_value;
		is_normal_data[result_idx] = is_normal;

		Value conclusion_val(conclusion);
		conclusion_vector.SetValue(result_idx, conclusion_val);

		ANOFOX_DEBUG("Normality test aggregate: n=" << n << ", JB=" << jb_statistic << ", p=" << p_value
		                                            << ", normal=" << is_normal);
	}
}

void NormalityTestAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering Normality Test aggregate function");

	child_list_t<LogicalType> normality_struct_fields;
	normality_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	normality_struct_fields.push_back(make_pair("skewness", LogicalType::DOUBLE));
	normality_struct_fields.push_back(make_pair("kurtosis", LogicalType::DOUBLE));
	normality_struct_fields.push_back(make_pair("jb_statistic", LogicalType::DOUBLE));
	normality_struct_fields.push_back(make_pair("p_value", LogicalType::DOUBLE));
	normality_struct_fields.push_back(make_pair("is_normal", LogicalType::BOOLEAN));
	normality_struct_fields.push_back(make_pair("conclusion", LogicalType::VARCHAR));

	AggregateFunction anofox_statistics_normality_test_agg(
	    "anofox_statistics_normality_test_agg", {LogicalType::DOUBLE, LogicalType::ANY}, // residual, options
	    LogicalType::STRUCT(normality_struct_fields), AggregateFunction::StateSize<NormalityTestAggregateState>,
	    NormalityTestInitialize, NormalityTestUpdate, NormalityTestCombine, NormalityTestFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING);

	loader.RegisterFunction(anofox_statistics_normality_test_agg);

	ANOFOX_DEBUG("Normality Test aggregate function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
