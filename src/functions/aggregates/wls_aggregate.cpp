#include "wls_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"
#include "../bridge/libanostat_wrapper.hpp"
#include "../bridge/type_converters.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * WLS Aggregate: anofox_stats_wls_agg(y DOUBLE, x DOUBLE[], weights DOUBLE) -> STRUCT
 *
 * Accumulates (y, x[], weight) tuples across rows in a GROUP BY,
 * then computes weighted least squares regression on finalize.
 */

struct WlsAggregateState {
	vector<double> y_values;
	vector<vector<double>> x_matrix; // Each row is one observation's features
	vector<double> weights;
	idx_t n_features = 0;
	RegressionOptions options;
	bool options_initialized = false;

	void Reset() {
		y_values.clear();
		x_matrix.clear();
		weights.clear();
		n_features = 0;
		options = RegressionOptions();
		options_initialized = false;
	}
};

static void WlsInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<WlsAggregateState *>(state_ptr);
	new (state) WlsAggregateState();
}

static void WlsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                      idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<WlsAggregateState *>(state_data);

	// Get y, x_array, weight, and options vectors
	auto &y_vector = inputs[0];
	auto &x_array_vector = inputs[1];
	auto &weight_vector = inputs[2];
	auto &options_vector = inputs[3];

	UnifiedVectorFormat y_data;
	UnifiedVectorFormat x_array_data;
	UnifiedVectorFormat weight_data;
	UnifiedVectorFormat options_data;
	y_vector.ToUnifiedFormat(count, y_data);
	x_array_vector.ToUnifiedFormat(count, x_array_data);
	weight_vector.ToUnifiedFormat(count, weight_data);
	options_vector.ToUnifiedFormat(count, options_data);

	auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);
	auto weight_ptr = UnifiedVectorFormat::GetData<double>(weight_data);

	// Process each row
	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_idx = y_data.sel->get_index(i);
		auto x_array_idx = x_array_data.sel->get_index(i);
		auto weight_idx = weight_data.sel->get_index(i);
		auto options_idx = options_data.sel->get_index(i);

		// Initialize options from first row
		if (!state.options_initialized && options_data.validity.RowIsValid(options_idx)) {
			auto options_value = options_vector.GetValue(options_idx);
			state.options = RegressionOptions::ParseFromMap(options_value);
			state.options_initialized = true;
		}

		// Skip if any input is NULL
		if (!y_data.validity.RowIsValid(y_idx) || !x_array_data.validity.RowIsValid(x_array_idx) ||
		    !weight_data.validity.RowIsValid(weight_idx)) {
			continue;
		}

		// Extract array elements from x
		auto x_array_entry = UnifiedVectorFormat::GetData<list_entry_t>(x_array_data)[x_array_idx];
		auto &x_child = ListVector::GetEntry(x_array_vector);

		UnifiedVectorFormat x_child_data;
		x_child.ToUnifiedFormat(ListVector::GetListSize(x_array_vector), x_child_data);
		auto x_child_ptr = UnifiedVectorFormat::GetData<double>(x_child_data);

		// Extract features from array
		vector<double> features;
		for (idx_t j = 0; j < x_array_entry.length; j++) {
			auto child_idx = x_child_data.sel->get_index(x_array_entry.offset + j);
			if (x_child_data.validity.RowIsValid(child_idx)) {
				features.push_back(x_child_ptr[child_idx]);
			} else {
				features.clear();
				break;
			}
		}

		if (features.empty()) {
			continue;
		}

		// Initialize n_features from first row
		if (state.n_features == 0) {
			state.n_features = features.size();
		} else if (features.size() != state.n_features) {
			continue;
		}

		// Add to state
		state.y_values.push_back(y_ptr[y_idx]);
		state.x_matrix.push_back(features);
		state.weights.push_back(weight_ptr[weight_idx]);
	}
}

static void WlsCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<WlsAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<WlsAggregateState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

		target_state.y_values.insert(target_state.y_values.end(), source_state.y_values.begin(),
		                             source_state.y_values.end());
		target_state.x_matrix.insert(target_state.x_matrix.end(), source_state.x_matrix.begin(),
		                             source_state.x_matrix.end());
		target_state.weights.insert(target_state.weights.end(), source_state.weights.begin(),
		                            source_state.weights.end());

		if (target_state.n_features == 0) {
			target_state.n_features = source_state.n_features;
		}
		if (!target_state.options_initialized && source_state.options_initialized) {
			target_state.options = source_state.options;
			target_state.options_initialized = true;
		}
	}
}

static void WlsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {

	auto states = FlatVector::GetData<WlsAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result: STRUCT(coefficients, intercept, r2, adj_r2, weighted_mse, n_obs,
	// mse, x_train_means, coefficient_std_errors, intercept_std_error, df_residual)
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0];
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto weighted_mse_data = FlatVector::GetData<double>(*struct_entries[4]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[5]);
	// Extended metadata
	auto mse_data = FlatVector::GetData<double>(*struct_entries[6]);
	auto &x_means_list = *struct_entries[7];
	auto &coef_se_list = *struct_entries[8];
	auto intercept_se_data = FlatVector::GetData<double>(*struct_entries[9]);
	auto df_resid_data = FlatVector::GetData<int64_t>(*struct_entries[10]);
	// New statistical metrics
	auto rse_data = FlatVector::GetData<double>(*struct_entries[11]);
	auto f_stat_data = FlatVector::GetData<double>(*struct_entries[12]);
	auto f_pval_data = FlatVector::GetData<double>(*struct_entries[13]);
	auto aic_data = FlatVector::GetData<double>(*struct_entries[14]);
	auto aicc_data = FlatVector::GetData<double>(*struct_entries[15]);
	auto bic_data = FlatVector::GetData<double>(*struct_entries[16]);
	auto loglik_data = FlatVector::GetData<double>(*struct_entries[17]);
	// Coefficient-level inference lists
	auto &coef_t_list = *struct_entries[18];
	auto &coef_p_list = *struct_entries[19];
	auto &coef_ci_lower_list = *struct_entries[20];
	auto &coef_ci_upper_list = *struct_entries[21];
	// Intercept-level inference
	auto intercept_t_data = FlatVector::GetData<double>(*struct_entries[22]);
	auto intercept_p_data = FlatVector::GetData<double>(*struct_entries[23]);
	auto intercept_ci_lower_data = FlatVector::GetData<double>(*struct_entries[24]);
	auto intercept_ci_upper_data = FlatVector::GetData<double>(*struct_entries[25]);

	auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
	auto &coef_child = ListVector::GetEntry(coef_list);
	ListVector::Reserve(coef_list, count * 10);

	// Prepare x_means list
	auto x_means_list_entries = FlatVector::GetData<list_entry_t>(x_means_list);
	auto &x_means_child = ListVector::GetEntry(x_means_list);
	ListVector::Reserve(x_means_list, count * 10);

	// Prepare coefficient_std_errors list
	auto coef_se_list_entries = FlatVector::GetData<list_entry_t>(coef_se_list);
	auto &coef_se_child = ListVector::GetEntry(coef_se_list);
	ListVector::Reserve(coef_se_list, count * 10);

	// Prepare coefficient-level inference lists
	auto coef_t_list_entries = FlatVector::GetData<list_entry_t>(coef_t_list);
	auto &coef_t_child = ListVector::GetEntry(coef_t_list);
	ListVector::Reserve(coef_t_list, count * 10);

	auto coef_p_list_entries = FlatVector::GetData<list_entry_t>(coef_p_list);
	auto &coef_p_child = ListVector::GetEntry(coef_p_list);
	ListVector::Reserve(coef_p_list, count * 10);

	auto coef_ci_lower_list_entries = FlatVector::GetData<list_entry_t>(coef_ci_lower_list);
	auto &coef_ci_lower_child = ListVector::GetEntry(coef_ci_lower_list);
	ListVector::Reserve(coef_ci_lower_list, count * 10);

	auto coef_ci_upper_list_entries = FlatVector::GetData<list_entry_t>(coef_ci_upper_list);
	auto &coef_ci_upper_child = ListVector::GetEntry(coef_ci_upper_list);
	ListVector::Reserve(coef_ci_upper_list, count * 10);

	idx_t list_offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.y_values.size();
		idx_t p = state.n_features;

		if (n < p + 1 || p == 0) {
			result_validity.SetInvalid(result_idx);
			list_entries[result_idx] = list_entry_t {list_offset, 0};
			continue;
		}

		// Build matrices
		Eigen::MatrixXd X(n, p);
		Eigen::VectorXd y(n);
		Eigen::VectorXd w(n);

		for (idx_t row = 0; row < n; row++) {
			y(row) = state.y_values[row];
			w(row) = state.weights[row];
			for (idx_t col = 0; col < p; col++) {
				X(row, col) = state.x_matrix[row][col];
			}
		}

		// Use libanostat WLSSolver (handles weighting and intercept automatically)
		auto lib_result =
		    bridge::LibanostatWrapper::FitWLS(state.y_values, state.x_matrix, state.weights, state.options, true);

		// Extract results
		double intercept = bridge::TypeConverters::ExtractIntercept(lib_result, state.options.intercept);
		auto feature_coefs_vec =
		    bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);
		Eigen::VectorXd feature_coefs =
		    Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());

		// Extract all fit statistics from libanostat (no recomputation)
		idx_t rank = lib_result.rank; // Already includes intercept!
		double r2 = lib_result.r_squared;
		double adj_r2 = lib_result.adj_r_squared;
		double mse = lib_result.mse;
		idx_t df_residual = lib_result.df_residual();

		// Compute x_means manually for metadata (for prediction API - this is data marshaling)
		// For WLS, we need weighted means
		Eigen::VectorXd w_eigen(n);
		for (idx_t i = 0; i < n; i++) {
			w_eigen(i) = state.weights[i];
		}
		double sum_weights = w_eigen.sum();
		Eigen::VectorXd x_means(p);
		for (idx_t j = 0; j < p; j++) {
			double weighted_sum = 0.0;
			for (idx_t i = 0; i < n; i++) {
				weighted_sum += state.weights[i] * state.x_matrix[i][j];
			}
			x_means(j) = weighted_sum / sum_weights;
		}

		// Store coefficients
		auto coef_data = FlatVector::GetData<double>(coef_child);
		auto &coef_validity = FlatVector::Validity(coef_child);
		for (idx_t j = 0; j < p; j++) {
			if (std::isnan(feature_coefs(j))) {
				coef_validity.SetInvalid(list_offset + j);
				coef_data[list_offset + j] = 0.0;
			} else {
				coef_data[list_offset + j] = feature_coefs(j);
			}
		}

		list_entries[result_idx] = list_entry_t {list_offset, p};

		// Extract extended metadata from libanostat
		// Intercept standard error
		double intercept_se = std::numeric_limits<double>::quiet_NaN();
		if (state.options.intercept && lib_result.has_std_errors && df_residual > 0) {
			// SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^-1 * x_mean))
			// Approximation: SE(intercept) ≈ sqrt(MSE / n) for centered data
			// Extract intercept std error (first element if intercept=true)
			if (lib_result.has_std_errors && lib_result.std_errors.size() > 0) {
				intercept_se = lib_result.std_errors(0);
			}
		}

		// 3. Store x_train_means in list (same offset as coefficients)
		auto x_means_data = FlatVector::GetData<double>(x_means_child);
		for (idx_t j = 0; j < p; j++) {
			x_means_data[list_offset + j] = x_means(j);
		}
		x_means_list_entries[result_idx] = list_entry_t {list_offset, p};

		// 4. Store coefficient_std_errors in list (same offset as coefficients)
		auto coef_se_data = FlatVector::GetData<double>(coef_se_child);
		auto &coef_se_validity = FlatVector::Validity(coef_se_child);
		// Extract feature std errors

		auto all_std_errors_vec = bridge::TypeConverters::ExtractStdErrors(lib_result);

		vector<double> feature_std_errors_vec;

		if (state.options.intercept && all_std_errors_vec.size() > 0) {

			feature_std_errors_vec.assign(all_std_errors_vec.begin() + 1, all_std_errors_vec.end());

		} else {

			feature_std_errors_vec = all_std_errors_vec;
		}

		Eigen::VectorXd feature_std_errors =
		    Eigen::Map<const Eigen::VectorXd>(feature_std_errors_vec.data(), feature_std_errors_vec.size());

		for (idx_t j = 0; j < p; j++) {
			if (lib_result.has_std_errors && !std::isnan(feature_std_errors(j))) {
				coef_se_data[list_offset + j] = feature_std_errors(j);
			} else {
				coef_se_validity.SetInvalid(list_offset + j);
				coef_se_data[list_offset + j] = 0.0;
			}
		}
		coef_se_list_entries[result_idx] = list_entry_t {list_offset, p};

		// Extract and store new statistical metrics
		double rse = bridge::TypeConverters::ExtractResidualStandardError(lib_result);
		double f_stat = bridge::TypeConverters::ExtractFStatistic(lib_result);
		double f_pval = bridge::TypeConverters::ExtractFStatisticPValue(lib_result);
		double aic = bridge::TypeConverters::ExtractAIC(lib_result);
		double aicc = bridge::TypeConverters::ExtractAICc(lib_result);
		double bic = bridge::TypeConverters::ExtractBIC(lib_result);
		double loglik = bridge::TypeConverters::ExtractLogLikelihood(lib_result);

		// Extract coefficient-level inference (feature coefficients only, excluding intercept)
		auto all_t_stats = bridge::TypeConverters::ExtractTStatistics(lib_result);
		auto all_p_vals = bridge::TypeConverters::ExtractPValues(lib_result);
		auto all_ci_lower = bridge::TypeConverters::ExtractCILower(lib_result);
		auto all_ci_upper = bridge::TypeConverters::ExtractCIUpper(lib_result);

		// Skip intercept (index 0) if present, extract feature coefficients
		vector<double> feature_t_stats, feature_p_vals, feature_ci_lower, feature_ci_upper;
		if (state.options.intercept && all_t_stats.size() > 0) {
			feature_t_stats.assign(all_t_stats.begin() + 1, all_t_stats.end());
			feature_p_vals.assign(all_p_vals.begin() + 1, all_p_vals.end());
			feature_ci_lower.assign(all_ci_lower.begin() + 1, all_ci_lower.end());
			feature_ci_upper.assign(all_ci_upper.begin() + 1, all_ci_upper.end());
		} else {
			feature_t_stats = all_t_stats;
			feature_p_vals = all_p_vals;
			feature_ci_lower = all_ci_lower;
			feature_ci_upper = all_ci_upper;
		}

		// Store coefficient-level inference in lists
		auto coef_t_data = FlatVector::GetData<double>(coef_t_child);
		auto &coef_t_validity = FlatVector::Validity(coef_t_child);
		auto coef_p_data = FlatVector::GetData<double>(coef_p_child);
		auto &coef_p_validity = FlatVector::Validity(coef_p_child);
		auto coef_ci_lower_data = FlatVector::GetData<double>(coef_ci_lower_child);
		auto &coef_ci_lower_validity = FlatVector::Validity(coef_ci_lower_child);
		auto coef_ci_upper_data = FlatVector::GetData<double>(coef_ci_upper_child);
		auto &coef_ci_upper_validity = FlatVector::Validity(coef_ci_upper_child);

		for (idx_t j = 0; j < p; j++) {
			// t-statistics
			if (j < feature_t_stats.size() && !std::isnan(feature_t_stats[j])) {
				coef_t_data[list_offset + j] = feature_t_stats[j];
			} else {
				coef_t_validity.SetInvalid(list_offset + j);
				coef_t_data[list_offset + j] = 0.0;
			}

			// p-values
			if (j < feature_p_vals.size() && !std::isnan(feature_p_vals[j])) {
				coef_p_data[list_offset + j] = feature_p_vals[j];
			} else {
				coef_p_validity.SetInvalid(list_offset + j);
				coef_p_data[list_offset + j] = 0.0;
			}

			// CI lower
			if (j < feature_ci_lower.size() && !std::isnan(feature_ci_lower[j])) {
				coef_ci_lower_data[list_offset + j] = feature_ci_lower[j];
			} else {
				coef_ci_lower_validity.SetInvalid(list_offset + j);
				coef_ci_lower_data[list_offset + j] = 0.0;
			}

			// CI upper
			if (j < feature_ci_upper.size() && !std::isnan(feature_ci_upper[j])) {
				coef_ci_upper_data[list_offset + j] = feature_ci_upper[j];
			} else {
				coef_ci_upper_validity.SetInvalid(list_offset + j);
				coef_ci_upper_data[list_offset + j] = 0.0;
			}
		}

		coef_t_list_entries[result_idx] = list_entry_t {list_offset, p};
		coef_p_list_entries[result_idx] = list_entry_t {list_offset, p};
		coef_ci_lower_list_entries[result_idx] = list_entry_t {list_offset, p};
		coef_ci_upper_list_entries[result_idx] = list_entry_t {list_offset, p};

		// Extract intercept-level inference
		double intercept_t = bridge::TypeConverters::ExtractInterceptTStatistic(lib_result);
		double intercept_p = bridge::TypeConverters::ExtractInterceptPValue(lib_result);
		double intercept_ci_lower = bridge::TypeConverters::ExtractInterceptCILower(lib_result);
		double intercept_ci_upper = bridge::TypeConverters::ExtractInterceptCIUpper(lib_result);

		// Increment offset for next result
		list_offset += p;

		// Fill all struct fields
		intercept_data[result_idx] = intercept;
		r2_data[result_idx] = r2;
		adj_r2_data[result_idx] = adj_r2;
		weighted_mse_data[result_idx] = mse; // For WLS, mse is already weighted
		n_data[result_idx] = n;
		mse_data[result_idx] = mse;
		intercept_se_data[result_idx] = intercept_se;
		df_resid_data[result_idx] = df_residual;
		// New statistical metrics
		rse_data[result_idx] = rse;
		f_stat_data[result_idx] = f_stat;
		f_pval_data[result_idx] = f_pval;
		aic_data[result_idx] = aic;
		aicc_data[result_idx] = aicc;
		bic_data[result_idx] = bic;
		loglik_data[result_idx] = loglik;
		// Intercept-level inference
		intercept_t_data[result_idx] = intercept_t;
		intercept_p_data[result_idx] = intercept_p;
		intercept_ci_lower_data[result_idx] = intercept_ci_lower;
		intercept_ci_upper_data[result_idx] = intercept_ci_upper;

		ANOFOX_DEBUG("WLS aggregate: n=" << n << ", p=" << p << ", r2=" << r2);
	}

	ListVector::SetListSize(coef_list, list_offset);
	ListVector::SetListSize(x_means_list, list_offset);
	ListVector::SetListSize(coef_se_list, list_offset);
	ListVector::SetListSize(coef_t_list, list_offset);
	ListVector::SetListSize(coef_p_list, list_offset);
	ListVector::SetListSize(coef_ci_lower_list, list_offset);
	ListVector::SetListSize(coef_ci_upper_list, list_offset);
}

// Destroy function not needed - std::vector handles cleanup automatically

/**
 * Window callback for WLS aggregate
 * Computes weighted least squares on the current window frame(s) for each row
 */
static void WlsWindow(AggregateInputData &aggr_input_data, const WindowPartitionInput &partition,
                      const_data_ptr_t g_state, data_ptr_t l_state, const SubFrames &subframes, Vector &result,
                      idx_t rid) {

	auto &result_validity = FlatVector::Validity(result);

	// Result: STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, weighted_mse DOUBLE, n_obs
	// BIGINT)
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0];
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto weighted_mse_data = FlatVector::GetData<double>(*struct_entries[4]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[5]);

	// Extract data for the entire partition
	vector<double> all_y;
	vector<vector<double>> all_x;
	vector<double> all_weights;
	RegressionOptions options;
	bool options_initialized = false;
	idx_t n_features = 0;

	// Read input data from the partition
	ColumnDataScanState scan_state;
	partition.inputs->InitializeScan(scan_state);

	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), partition.inputs->Types());

	while (partition.inputs->Scan(scan_state, chunk)) {
		auto &y_chunk = chunk.data[0];       // y
		auto &x_array_chunk = chunk.data[1]; // x array
		auto &weight_chunk = chunk.data[2];  // weight
		auto &options_chunk = chunk.data[3]; // options

		UnifiedVectorFormat y_data;
		UnifiedVectorFormat x_array_data;
		UnifiedVectorFormat weight_data;
		UnifiedVectorFormat options_data;
		y_chunk.ToUnifiedFormat(chunk.size(), y_data);
		x_array_chunk.ToUnifiedFormat(chunk.size(), x_array_data);
		weight_chunk.ToUnifiedFormat(chunk.size(), weight_data);
		options_chunk.ToUnifiedFormat(chunk.size(), options_data);

		auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);
		auto weight_ptr = UnifiedVectorFormat::GetData<double>(weight_data);

		for (idx_t i = 0; i < chunk.size(); i++) {
			auto y_idx = y_data.sel->get_index(i);
			auto x_array_idx = x_array_data.sel->get_index(i);
			auto weight_idx = weight_data.sel->get_index(i);
			auto options_idx = options_data.sel->get_index(i);

			// Initialize options from first valid row
			if (!options_initialized && options_data.validity.RowIsValid(options_idx)) {
				auto options_value = options_chunk.GetValue(options_idx);
				options = RegressionOptions::ParseFromMap(options_value);
				options_initialized = true;
			}

			// Skip if any is NULL
			if (!y_data.validity.RowIsValid(y_idx) || !x_array_data.validity.RowIsValid(x_array_idx) ||
			    !weight_data.validity.RowIsValid(weight_idx)) {
				all_y.push_back(std::numeric_limits<double>::quiet_NaN());
				all_x.push_back(vector<double>());
				all_weights.push_back(0.0);
				continue;
			}

			// Extract array elements
			auto x_array_entry = UnifiedVectorFormat::GetData<list_entry_t>(x_array_data)[x_array_idx];
			auto &x_child = ListVector::GetEntry(x_array_chunk);

			UnifiedVectorFormat x_child_data;
			x_child.ToUnifiedFormat(ListVector::GetListSize(x_array_chunk), x_child_data);
			auto x_child_ptr = UnifiedVectorFormat::GetData<double>(x_child_data);

			vector<double> features;
			for (idx_t j = 0; j < x_array_entry.length; j++) {
				auto child_idx = x_child_data.sel->get_index(x_array_entry.offset + j);
				if (x_child_data.validity.RowIsValid(child_idx)) {
					features.push_back(x_child_ptr[child_idx]);
				} else {
					features.clear();
					break;
				}
			}

			if (n_features == 0 && !features.empty()) {
				n_features = features.size();
			}

			all_y.push_back(y_ptr[y_idx]);
			all_x.push_back(features);
			all_weights.push_back(weight_ptr[weight_idx]);
		}
	}

	// Now compute WLS for the current window frame(s)
	vector<double> window_y;
	vector<vector<double>> window_x;
	vector<double> window_weights;

	// Accumulate all rows in the frame
	for (const auto &frame : subframes) {
		for (idx_t frame_idx = frame.start; frame_idx < frame.end; frame_idx++) {
			if (frame_idx < all_y.size() && !std::isnan(all_y[frame_idx]) && !all_x[frame_idx].empty()) {
				window_y.push_back(all_y[frame_idx]);
				window_x.push_back(all_x[frame_idx]);
				window_weights.push_back(all_weights[frame_idx]);
			}
		}
	}

	idx_t n = window_y.size();
	idx_t p = n_features;

	// Compute sum of weights
	double sum_weights = 0.0;
	for (idx_t i = 0; i < n; i++) {
		sum_weights += window_weights[i];
	}

	// Need at least p+1 observations for p features
	if (n < p + 1 || p == 0) {
		result_validity.SetInvalid(rid);
		auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
		list_entries[rid] = list_entry_t {0, 0};
		return;
	}

	// Build matrices
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);
	Eigen::VectorXd w(n);

	for (idx_t row = 0; row < n; row++) {
		y(row) = window_y[row];
		w(row) = window_weights[row];
		for (idx_t col = 0; col < p; col++) {
			X(row, col) = window_x[row][col];
		}
	}

	// Convert window data to DuckDB vectors for libanostat
	vector<double> y_vec(n);
	vector<double> w_vec(n);
	vector<vector<double>> x_vec(n, vector<double>(p));
	for (idx_t i = 0; i < n; i++) {
		y_vec[i] = window_y[i];
		w_vec[i] = window_weights[i];
		for (idx_t j = 0; j < p; j++) {
			x_vec[i][j] = window_x[i][j];
		}
	}

	// Use libanostat WLSSolver
	auto lib_result = bridge::LibanostatWrapper::FitWLS(y_vec, x_vec, w_vec, options, false);

	// Extract results
	double intercept = bridge::TypeConverters::ExtractIntercept(lib_result, options.intercept);
	auto feature_coefs_vec = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, options.intercept);
	Eigen::VectorXd feature_coefs =
	    Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());
	idx_t rank = lib_result.rank;

	// Compute predictions
	Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(n, intercept);
	for (idx_t j = 0; j < p; j++) {
		if (!std::isnan(feature_coefs(j))) {
			y_pred += feature_coefs(j) * X.col(j);
		}
	}

	// Compute weighted statistics
	Eigen::VectorXd residuals = y - y_pred;
	double ss_res = residuals.squaredNorm(); // Unweighted for MSE calculation
	double ss_res_weighted = (w.array() * residuals.array().square()).sum();

	double ss_tot_weighted;
	if (options.intercept) {
		double y_weighted_mean = (w.array() * y.array()).sum() / sum_weights;
		ss_tot_weighted = (w.array() * (y.array() - y_weighted_mean).square()).sum();
	} else {
		// No intercept: total sum of squares from zero
		ss_tot_weighted = (w.array() * y.array().square()).sum();
	}

	double r2 = (ss_tot_weighted > 1e-10) ? (1.0 - ss_res_weighted / ss_tot_weighted) : 0.0;

	// Adjusted R²: rank already includes intercept if fitted
	idx_t df_model = rank;
	double adj_r2 = 1.0 - (1.0 - r2) * static_cast<double>(n - 1) / static_cast<double>(n - df_model);
	double weighted_mse = ss_res_weighted / sum_weights;

	// Store coefficients in list
	auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
	auto &coef_child = ListVector::GetEntry(coef_list);
	auto coef_data = FlatVector::GetData<double>(coef_child);
	auto &coef_validity = FlatVector::Validity(coef_child);

	idx_t list_offset = rid * p; // Each result gets p coefficients
	ListVector::Reserve(coef_list, (rid + 1) * p);

	for (idx_t j = 0; j < p; j++) {
		if (std::isnan(feature_coefs(j))) {
			coef_validity.SetInvalid(list_offset + j);
			coef_data[list_offset + j] = 0.0;
		} else {
			coef_data[list_offset + j] = feature_coefs(j);
		}
	}

	list_entries[rid] = list_entry_t {list_offset, p};

	// Fill other struct fields
	intercept_data[rid] = intercept;
	r2_data[rid] = r2;
	adj_r2_data[rid] = adj_r2;
	weighted_mse_data[rid] = weighted_mse;
	n_data[rid] = n;

	ANOFOX_DEBUG("WLS window: n=" << n << ", p=" << p << ", r2=" << r2);
}

void WlsAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering WLS aggregate function");

	child_list_t<LogicalType> wls_struct_fields;
	wls_struct_fields.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	wls_struct_fields.push_back(make_pair("intercept", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("r2", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("adj_r2", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("weighted_mse", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	// Extended metadata for model_predict compatibility
	wls_struct_fields.push_back(make_pair("mse", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("x_train_means", LogicalType::LIST(LogicalType::DOUBLE)));
	wls_struct_fields.push_back(make_pair("coefficient_std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
	wls_struct_fields.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("df_residual", LogicalType::BIGINT));
	// New statistical metrics
	wls_struct_fields.push_back(make_pair("residual_standard_error", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("f_statistic_pvalue", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("aic", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("aicc", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("bic", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("log_likelihood", LogicalType::DOUBLE));
	// Coefficient-level inference
	wls_struct_fields.push_back(make_pair("coefficient_t_statistics", LogicalType::LIST(LogicalType::DOUBLE)));
	wls_struct_fields.push_back(make_pair("coefficient_p_values", LogicalType::LIST(LogicalType::DOUBLE)));
	wls_struct_fields.push_back(make_pair("coefficient_ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
	wls_struct_fields.push_back(make_pair("coefficient_ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
	// Intercept-level inference
	wls_struct_fields.push_back(make_pair("intercept_t_statistic", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("intercept_p_value", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("intercept_ci_lower", LogicalType::DOUBLE));
	wls_struct_fields.push_back(make_pair("intercept_ci_upper", LogicalType::DOUBLE));

	AggregateFunction anofox_stats_wls_fit_agg(
	    "anofox_stats_wls_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::STRUCT(wls_struct_fields), AggregateFunction::StateSize<WlsAggregateState>, WlsInitialize,
	    WlsUpdate, WlsCombine, WlsFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr,
	    nullptr, WlsWindow, nullptr, nullptr);
	loader.RegisterFunction(anofox_stats_wls_fit_agg);

	// Register alias without prefix
	AggregateFunction wls_fit_agg_alias(
	    "wls_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::STRUCT(wls_struct_fields), AggregateFunction::StateSize<WlsAggregateState>, WlsInitialize,
	    WlsUpdate, WlsCombine, WlsFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr,
	    nullptr, WlsWindow, nullptr, nullptr);
	loader.RegisterFunction(wls_fit_agg_alias);

	ANOFOX_DEBUG("WLS aggregate function registered successfully with alias wls_fit_agg");
}

} // namespace anofox_statistics
} // namespace duckdb
