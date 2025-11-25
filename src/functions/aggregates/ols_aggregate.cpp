#include "ols_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/options_parser.hpp"
#include "../bridge/libanostat_wrapper.hpp"
#include "../bridge/type_converters.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * State for OLS aggregate computation
 * Accumulates y and x values across rows, then computes OLS on finalize
 */
struct OlsAggregateState {
	vector<double> y_values;
	vector<double> x_values;

	void Reset() {
		y_values.clear();
		x_values.clear();
	}
};

/**
 * State for multi-variable OLS aggregate (array-based inputs)
 */
struct OlsArrayAggregateState {
	vector<double> y_values;
	vector<vector<double>> x_matrix; // Each row is one observation's features
	idx_t n_features = 0;
	RegressionOptions options;
	bool options_initialized = false;

	void Reset() {
		y_values.clear();
		x_matrix.clear();
		n_features = 0;
		options = RegressionOptions();
		options_initialized = false;
	}
};

/**
 * Initialize state (called once per group)
 */
static void OlsInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<OlsAggregateState *>(state_ptr);
	new (state) OlsAggregateState();
}

/**
 * Update state with new row (called for each row in group)
 */
static void OlsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                      idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<OlsAggregateState *>(state_data);

	// Get y and x vectors
	auto &y_vector = inputs[0];
	auto &x_vector = inputs[1];

	UnifiedVectorFormat y_data;
	UnifiedVectorFormat x_data;
	y_vector.ToUnifiedFormat(count, y_data);
	x_vector.ToUnifiedFormat(count, x_data);

	auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_ptr = UnifiedVectorFormat::GetData<double>(x_data);

	// Accumulate all non-NULL pairs
	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);

		// Skip if either value is NULL
		if (y_data.validity.RowIsValid(y_idx) && x_data.validity.RowIsValid(x_idx)) {
			state.y_values.push_back(y_ptr[y_idx]);
			state.x_values.push_back(x_ptr[x_idx]);
		}
	}
}

/**
 * Combine two states (for parallel aggregation)
 */
static void OlsCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<OlsAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<OlsAggregateState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

		// Merge source data into target
		target_state.y_values.insert(target_state.y_values.end(), source_state.y_values.begin(),
		                             source_state.y_values.end());
		target_state.x_values.insert(target_state.x_values.end(), source_state.x_values.begin(),
		                             source_state.x_values.end());
	}
}

/**
 * Finalize: Compute OLS coefficient and return result
 */
static void OlsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {

	auto states = FlatVector::GetData<OlsAggregateState *>(state_vector);
	auto result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.y_values.size();

		// Need at least 2 observations for regression
		if (n < 2) {
			result_validity.SetInvalid(result_idx);
			continue;
		}

		// Use libanostat OLSSolver for single variable OLS (NO intercept)
		// Convert to vector<vector<double>> format for X matrix (single column)
		vector<vector<double>> x_matrix;
		x_matrix.reserve(n);
		for (idx_t j = 0; j < n; j++) {
			x_matrix.push_back({state.x_values[j]});
		}

		// Fit using libanostat (no intercept)
		RegressionOptions opts;
		opts.intercept = false;

		try {
			auto lib_result = bridge::LibanostatWrapper::FitOLS(state.y_values, x_matrix, opts, false);

			// Extract coefficient (position 0 since no intercept)
			if (lib_result.is_aliased[0]) {
				result_validity.SetInvalid(result_idx);
				continue;
			}

			result_data[result_idx] = lib_result.coefficients[0];
			ANOFOX_DEBUG("OLS aggregate: n=" << n << ", coeff=" << lib_result.coefficients[0]);
		} catch (...) {
			result_validity.SetInvalid(result_idx);
		}
	}
}

/**
 * Destroy state (cleanup)
 */
static void OlsDestroy(Vector &state_vector, AggregateInputData &aggr_input_data, idx_t count) {
	auto states = FlatVector::GetData<OlsAggregateState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		states[i]->~OlsAggregateState();
	}
}

/**
 * Finalize for full fit struct: Compute OLS and return struct with all statistics
 */
static void OlsFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {

	auto states = FlatVector::GetData<OlsAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result is a STRUCT with fields: coefficient, intercept, r2, n_obs, std_error
	auto &struct_entries = StructVector::GetEntries(result);
	auto coef_data = FlatVector::GetData<double>(*struct_entries[0]);
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[3]);
	auto stderr_data = FlatVector::GetData<double>(*struct_entries[4]);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.y_values.size();

		// Need at least 2 observations for regression
		if (n < 2) {
			result_validity.SetInvalid(result_idx);
			continue;
		}

		// Use libanostat OLSSolver for single variable OLS WITH intercept
		// Convert to vector<vector<double>> format for X matrix (single column)
		vector<vector<double>> x_matrix;
		x_matrix.reserve(n);
		for (idx_t j = 0; j < n; j++) {
			x_matrix.push_back({state.x_values[j]});
		}

		// Fit using libanostat WITH intercept and standard errors
		RegressionOptions opts;
		opts.intercept = true;

		try {
			auto lib_result = bridge::LibanostatWrapper::FitOLS(state.y_values, x_matrix, opts, true);

			// Check if coefficient is aliased
			// With intercept: coefficients[0] = intercept, coefficients[1] = slope
			if (lib_result.is_aliased[1]) {
				result_validity.SetInvalid(result_idx);
				continue;
			}

			// Extract all statistics from libanostat result
			intercept_data[result_idx] = lib_result.coefficients[0];
			coef_data[result_idx] = lib_result.coefficients[1];
			r2_data[result_idx] = lib_result.r_squared;
			n_data[result_idx] = static_cast<int64_t>(n);
			stderr_data[result_idx] = lib_result.std_errors[1]; // SE of slope

			ANOFOX_DEBUG("OLS fit aggregate: n=" << n << ", coef=" << lib_result.coefficients[1]
			                                     << ", r2=" << lib_result.r_squared);
		} catch (...) {
			result_validity.SetInvalid(result_idx);
		}
	}
}

/**
 * Array aggregate functions for multi-variable OLS
 */

static void OlsArrayInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<OlsArrayAggregateState *>(state_ptr);
	new (state) OlsArrayAggregateState();
}

static void OlsArrayUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<OlsArrayAggregateState *>(state_data);

	// Get y, x_array, and options vectors
	auto &y_vector = inputs[0];
	auto &x_array_vector = inputs[1];
	auto &options_vector = inputs[2];

	UnifiedVectorFormat y_data;
	UnifiedVectorFormat x_array_data;
	UnifiedVectorFormat options_data;
	y_vector.ToUnifiedFormat(count, y_data);
	x_array_vector.ToUnifiedFormat(count, x_array_data);
	options_vector.ToUnifiedFormat(count, options_data);

	auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);

	// Process each row
	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_idx = y_data.sel->get_index(i);
		auto x_array_idx = x_array_data.sel->get_index(i);
		auto options_idx = options_data.sel->get_index(i);

		// Initialize options from first row
		if (!state.options_initialized && options_data.validity.RowIsValid(options_idx)) {
			auto options_value = options_vector.GetValue(options_idx);
			state.options = RegressionOptions::ParseFromMap(options_value);
			state.options_initialized = true;
		}

		// Skip if y or x_array is NULL
		if (!y_data.validity.RowIsValid(y_idx) || !x_array_data.validity.RowIsValid(x_array_idx)) {
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
				// NULL in array - skip this row
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
			// Feature count mismatch - skip this row
			continue;
		}

		// Add to state
		state.y_values.push_back(y_ptr[y_idx]);
		state.x_matrix.push_back(features);
	}
}

static void OlsArrayCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<OlsArrayAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<OlsArrayAggregateState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

		// Merge source data into target
		target_state.y_values.insert(target_state.y_values.end(), source_state.y_values.begin(),
		                             source_state.y_values.end());
		target_state.x_matrix.insert(target_state.x_matrix.end(), source_state.x_matrix.begin(),
		                             source_state.x_matrix.end());

		if (target_state.n_features == 0) {
			target_state.n_features = source_state.n_features;
		}
		if (!target_state.options_initialized && source_state.options_initialized) {
			target_state.options = source_state.options;
			target_state.options_initialized = true;
		}
	}
}

static void OlsArrayFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {

	auto states = FlatVector::GetData<OlsArrayAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result is a STRUCT with fields: coefficients, intercept, r2, adj_r2, n_obs,
	// mse, x_train_means, coefficient_std_errors, intercept_std_error, df_residual
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0]; // LIST of coefficients
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[4]);
	// Extended metadata
	auto mse_data = FlatVector::GetData<double>(*struct_entries[5]);
	auto &x_means_list = *struct_entries[6];
	auto &coef_se_list = *struct_entries[7];
	auto intercept_se_data = FlatVector::GetData<double>(*struct_entries[8]);
	auto df_resid_data = FlatVector::GetData<int64_t>(*struct_entries[9]);

	// Get list entry data and child vector for coefficients
	auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
	auto &coef_child = ListVector::GetEntry(coef_list);
	ListVector::Reserve(coef_list, count * 10); // Reserve space (estimate)

	// Prepare x_means list
	auto x_means_list_entries = FlatVector::GetData<list_entry_t>(x_means_list);
	auto &x_means_child = ListVector::GetEntry(x_means_list);
	ListVector::Reserve(x_means_list, count * 10);

	// Prepare coefficient_std_errors list
	auto coef_se_list_entries = FlatVector::GetData<list_entry_t>(coef_se_list);
	auto &coef_se_child = ListVector::GetEntry(coef_se_list);
	ListVector::Reserve(coef_se_list, count * 10);

	idx_t list_offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.y_values.size();
		idx_t p = state.n_features;

		// Need at least p+1 observations for p features
		if (n < p + 1 || p == 0) {
			result_validity.SetInvalid(result_idx);
			list_entries[result_idx] = list_entry_t {list_offset, 0};
			continue;
		}

		// Use libanostat OLSSolver (handles intercept and centering automatically)
		auto lib_result = bridge::LibanostatWrapper::FitOLS(state.y_values, state.x_matrix, state.options, true);

		// Extract results using TypeConverters
		double intercept = bridge::TypeConverters::ExtractIntercept(lib_result, state.options.intercept);
		auto feature_coefs_vec =
		    bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);
		Eigen::VectorXd feature_coefs =
		    Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());
		// Compute x_means manually (for prediction API - this is data marshaling, not statistics)
		Eigen::VectorXd x_means(p);
		for (idx_t j = 0; j < p; j++) {
			double sum = 0.0;
			for (idx_t i = 0; i < n; i++) {
				sum += state.x_matrix[i][j];
			}
			x_means(j) = sum / n;
		}

		// Extract all fit statistics from libanostat (no recomputation)
		idx_t rank = lib_result.rank; // Already includes intercept if fitted!
		double r2 = lib_result.r_squared;
		double adj_r2 = lib_result.adj_r_squared;

		// Store coefficients in child vector (NaN for aliased -> will be NULL)
		auto coef_data = FlatVector::GetData<double>(coef_child);
		auto &coef_validity = FlatVector::Validity(coef_child);
		for (idx_t j = 0; j < p; j++) {
			if (std::isnan(feature_coefs(j))) {
				// Aliased coefficient -> set as invalid (NULL)
				coef_validity.SetInvalid(list_offset + j);
				coef_data[list_offset + j] = 0.0; // Placeholder value
			} else {
				coef_data[list_offset + j] = feature_coefs(j);
			}
		}

		// Set list entry for coefficients
		list_entries[result_idx] = list_entry_t {list_offset, p};

		// Extract extended metadata from libanostat (no recomputation)
		// 1. MSE and df_residual
		double mse = lib_result.mse;
		idx_t df_residual = lib_result.df_residual();

		// 2. Intercept standard error from libanostat
		double intercept_se = std::numeric_limits<double>::quiet_NaN();
		if (state.options.intercept && lib_result.has_std_errors) {
			intercept_se = lib_result.std_errors[0]; // Intercept SE at position 0
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
		// Extract feature std errors (excluding intercept)
		auto all_std_errors_vec = bridge::TypeConverters::ExtractStdErrors(lib_result);
		vector<double> feature_std_errors_vec;
		if (state.options.intercept && all_std_errors_vec.size() > 0) {
			// Skip first element (intercept), extract rest
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

		// Increment offset for next result
		list_offset += p;

		// Fill all struct fields
		intercept_data[result_idx] = intercept;
		r2_data[result_idx] = r2;
		adj_r2_data[result_idx] = adj_r2;
		n_data[result_idx] = n;
		mse_data[result_idx] = mse;
		intercept_se_data[result_idx] = intercept_se;
		df_resid_data[result_idx] = df_residual;

		ANOFOX_DEBUG("OLS array aggregate: n=" << n << ", p=" << p << ", r2=" << r2 << ", mse=" << mse);
	}

	// Set final list sizes
	ListVector::SetListSize(coef_list, list_offset);
	ListVector::SetListSize(x_means_list, list_offset);
	ListVector::SetListSize(coef_se_list, list_offset);
}

// Destroy function not needed - std::vector handles cleanup automatically

/**
 * Window callback for OLS array aggregate
 * Computes OLS on the current window frame(s) for each row
 */
static void OlsArrayWindow(AggregateInputData &aggr_input_data, const WindowPartitionInput &partition,
                           const_data_ptr_t g_state, data_ptr_t l_state, const SubFrames &subframes, Vector &result,
                           idx_t rid) {

	auto &result_validity = FlatVector::Validity(result);

	// Result is a STRUCT with fields: coefficients (DOUBLE[]), intercept, r2, adj_r2, n_obs
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0];
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[4]);

	// Access input data from partition
	// The partition.inputs contains all input columns for this partition
	// We need to extract y, x_array, and options data

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

	idx_t row_idx = 0;
	while (partition.inputs->Scan(scan_state, chunk)) {
		// Access columns by their column IDs
		auto &y_chunk = chunk.data[0];       // First input is y
		auto &x_array_chunk = chunk.data[1]; // Second input is x array
		auto &options_chunk = chunk.data[2]; // Third input is options

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

			// Initialize options from first valid row
			if (!options_initialized && options_data.validity.RowIsValid(options_idx)) {
				auto options_value = options_chunk.GetValue(options_idx);
				options = RegressionOptions::ParseFromMap(options_value);
				options_initialized = true;
			}

			// Skip if y or x_array is NULL
			if (!y_data.validity.RowIsValid(y_idx) || !x_array_data.validity.RowIsValid(x_array_idx)) {
				all_y.push_back(std::numeric_limits<double>::quiet_NaN());
				all_x.push_back(vector<double>());
				row_idx++;
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
			row_idx++;
		}
	}

	// Now compute OLS for the current window frame(s)
	vector<double> window_y;
	vector<vector<double>> window_x;

	// Accumulate all rows in the frame
	for (const auto &frame : subframes) {
		for (idx_t frame_idx = frame.start; frame_idx < frame.end; frame_idx++) {
			if (frame_idx < all_y.size() && !std::isnan(all_y[frame_idx]) && !all_x[frame_idx].empty()) {
				window_y.push_back(all_y[frame_idx]);
				window_x.push_back(all_x[frame_idx]);
			}
		}
	}

	idx_t n = window_y.size();
	idx_t p = n_features;

	// Need at least p+1 observations for p features
	if (n < p + 1 || p == 0) {
		result_validity.SetInvalid(rid);
		auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
		list_entries[rid] = list_entry_t {0, 0};
		return;
	}

	// Build design matrix using Eigen
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);

	for (idx_t row = 0; row < n; row++) {
		y(row) = window_y[row];
		for (idx_t col = 0; col < p; col++) {
			X(row, col) = window_x[row][col];
		}
	}

	// Convert to DuckDB vectors for libanostat
	vector<double> y_vec(n);
	vector<vector<double>> x_vec(n, vector<double>(p));
	for (idx_t i = 0; i < n; i++) {
		y_vec[i] = window_y[i];
		for (idx_t j = 0; j < p; j++) {
			x_vec[i][j] = window_x[i][j];
		}
	}

	// Use libanostat OLSSolver
	auto lib_result = bridge::LibanostatWrapper::FitOLS(y_vec, x_vec, options, false);

	// Extract results
	double intercept = bridge::TypeConverters::ExtractIntercept(lib_result, options.intercept);
	auto feature_coefs_vec = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, options.intercept);
	Eigen::VectorXd feature_coefs =
	    Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());
	idx_t rank = lib_result.rank; // Already includes intercept!

	// Compute predictions
	Eigen::VectorXd y_pred(n);
	for (idx_t i = 0; i < n; i++) {
		y_pred(i) = intercept;
		for (idx_t j = 0; j < p; j++) {
			if (!std::isnan(lib_result.coefficients(j))) {
				y_pred(i) += lib_result.coefficients(j) * window_x[i][j];
			}
		}
	}

	// Compute R²
	Eigen::VectorXd y_eigen(n);
	for (idx_t i = 0; i < n; i++) {
		y_eigen(i) = window_y[i];
	}
	Eigen::VectorXd residuals = y_eigen - y_pred;
	double ss_res = residuals.squaredNorm();

	double ss_tot;
	if (options.intercept) {
		double mean_y = 0.0;
		for (idx_t i = 0; i < n; i++) {
			mean_y += window_y[i];
		}
		mean_y /= n;
		ss_tot = 0.0;
		for (idx_t i = 0; i < n; i++) {
			double diff = window_y[i] - mean_y;
			ss_tot += diff * diff;
		}
	} else {
		ss_tot = 0.0;
		for (idx_t i = 0; i < n; i++) {
			ss_tot += window_y[i] * window_y[i];
		}
	}

	double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

	// Adjusted R²: lib_result.rank already includes intercept
	idx_t df_model = rank;
	double adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);

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
	n_data[rid] = n;

	ANOFOX_DEBUG("OLS window: n=" << n << ", p=" << p << ", r2=" << r2);
}

void OlsAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering OLS aggregate functions");

	// 1. ols_coeff_agg(y DOUBLE, x DOUBLE) -> DOUBLE
	AggregateFunction ols_coeff_agg("ols_coeff_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                                AggregateFunction::StateSize<OlsAggregateState>, OlsInitialize, OlsUpdate,
	                                OlsCombine, OlsFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr,
	                                nullptr, OlsDestroy, nullptr, nullptr, nullptr, nullptr);
	loader.RegisterFunction(ols_coeff_agg);

	// 2. ols_fit_agg(y DOUBLE, x DOUBLE) -> STRUCT(coefficient DOUBLE, intercept DOUBLE, r2 DOUBLE, n_obs BIGINT,
	// std_error DOUBLE)
	child_list_t<LogicalType> fit_struct_fields;
	fit_struct_fields.push_back(make_pair("coefficient", LogicalType::DOUBLE));
	fit_struct_fields.push_back(make_pair("intercept", LogicalType::DOUBLE));
	fit_struct_fields.push_back(make_pair("r2", LogicalType::DOUBLE));
	fit_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	fit_struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));

	AggregateFunction ols_fit_agg(
	    "ols_fit_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::STRUCT(fit_struct_fields),
	    AggregateFunction::StateSize<OlsAggregateState>, OlsInitialize, OlsUpdate, OlsCombine, OlsFitFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, OlsDestroy, nullptr, nullptr, nullptr, nullptr);
	loader.RegisterFunction(ols_fit_agg);

	// 3. ols_fit_agg_array(y DOUBLE, x DOUBLE[]) -> STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2
	// DOUBLE, n_obs BIGINT, mse, x_train_means, coefficient_std_errors, intercept_std_error, df_residual)
	// Extended to always include metadata needed for model_predict
	child_list_t<LogicalType> array_fit_struct_fields;
	array_fit_struct_fields.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	array_fit_struct_fields.push_back(make_pair("intercept", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("r2", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("adj_r2", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	// Extended metadata for model_predict compatibility
	array_fit_struct_fields.push_back(make_pair("mse", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("x_train_means", LogicalType::LIST(LogicalType::DOUBLE)));
	array_fit_struct_fields.push_back(make_pair("coefficient_std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
	array_fit_struct_fields.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("df_residual", LogicalType::BIGINT));

	AggregateFunction ols_fit_agg_array(
	    "ols_fit_agg_array", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
	    LogicalType::STRUCT(array_fit_struct_fields), AggregateFunction::StateSize<OlsArrayAggregateState>,
	    OlsArrayInitialize, OlsArrayUpdate, OlsArrayCombine, OlsArrayFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
	loader.RegisterFunction(ols_fit_agg_array);

	// 4. anofox_statistics_ols_agg(y DOUBLE, x DOUBLE[], options MAP) -> STRUCT
	// This is the new unified API that matches table function signatures
	// Now supports window functions with OVER clause
	AggregateFunction anofox_statistics_ols_agg(
	    "anofox_statistics_ols_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(array_fit_struct_fields), AggregateFunction::StateSize<OlsArrayAggregateState>,
	    OlsArrayInitialize, OlsArrayUpdate, OlsArrayCombine, OlsArrayFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr, nullptr, OlsArrayWindow, nullptr,
	    nullptr);
	loader.RegisterFunction(anofox_statistics_ols_agg);

	ANOFOX_DEBUG("All OLS aggregate functions registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
