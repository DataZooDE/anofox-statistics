#include "ols_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"
#include "../utils/options_parser.hpp"

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

		// Simple OLS for single variable: β = Cov(x,y) / Var(x)
		// Or equivalently: β = (X'X)^(-1) X'y for single column X

		double sum_x = 0.0, sum_y = 0.0;
		for (idx_t j = 0; j < n; j++) {
			sum_x += state.x_values[j];
			sum_y += state.y_values[j];
		}

		double mean_x = sum_x / static_cast<double>(n);
		double mean_y = sum_y / static_cast<double>(n);

		double cov_xy = 0.0;
		double var_x = 0.0;

		for (idx_t j = 0; j < n; j++) {
			double dx = state.x_values[j] - mean_x;
			double dy = state.y_values[j] - mean_y;
			cov_xy += dx * dy;
			var_x += dx * dx;
		}

		// Avoid division by zero
		if (var_x < 1e-10) {
			result_validity.SetInvalid(result_idx);
			continue;
		}

		double coefficient = cov_xy / var_x;
		result_data[result_idx] = coefficient;

		ANOFOX_DEBUG("OLS aggregate: n=" << n << ", coeff=" << coefficient);
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

		// Compute OLS: y = intercept + coefficient * x
		double sum_x = 0.0, sum_y = 0.0;
		for (idx_t j = 0; j < n; j++) {
			sum_x += state.x_values[j];
			sum_y += state.y_values[j];
		}

		double mean_x = sum_x / static_cast<double>(n);
		double mean_y = sum_y / static_cast<double>(n);

		double cov_xy = 0.0;
		double var_x = 0.0;

		for (idx_t j = 0; j < n; j++) {
			double dx = state.x_values[j] - mean_x;
			double dy = state.y_values[j] - mean_y;
			cov_xy += dx * dy;
			var_x += dx * dx;
		}

		// Avoid division by zero
		if (var_x < 1e-10) {
			result_validity.SetInvalid(result_idx);
			continue;
		}

		double coefficient = cov_xy / var_x;
		double intercept = mean_y - coefficient * mean_x;

		// Compute R² and residual standard error
		double ss_tot = 0.0; // Total sum of squares
		double ss_res = 0.0; // Residual sum of squares

		for (idx_t j = 0; j < n; j++) {
			double y_pred = intercept + coefficient * state.x_values[j];
			double residual = state.y_values[j] - y_pred;
			ss_res += residual * residual;

			double deviation = state.y_values[j] - mean_y;
			ss_tot += deviation * deviation;
		}

		double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

		// Standard error of coefficient: sqrt(MSE / sum((x - mean_x)²))
		double mse = (n > 2) ? (ss_res / static_cast<double>(n - 2)) : 0.0;
		double std_error = std::sqrt(mse / var_x);

		// Fill struct fields
		coef_data[result_idx] = coefficient;
		intercept_data[result_idx] = intercept;
		r2_data[result_idx] = r2;
		n_data[result_idx] = n;
		stderr_data[result_idx] = std_error;

		ANOFOX_DEBUG("OLS fit aggregate: n=" << n << ", coef=" << coefficient << ", r2=" << r2);
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

		// Build design matrix using Eigen
		Eigen::MatrixXd X(n, p);
		Eigen::VectorXd y(n);

		for (idx_t row = 0; row < n; row++) {
			y(row) = state.y_values[row];
			for (idx_t col = 0; col < p; col++) {
				X(row, col) = state.x_matrix[row][col];
			}
		}

		// Handle intercept option
		double intercept = 0.0;
		RankDeficientOlsResult ols_result;

		// Store x_means for later use
		Eigen::VectorXd x_means;

		if (state.options.intercept) {
			// With intercept: center data, solve, compute intercept
			double mean_y = y.mean();
			x_means = X.colwise().mean();

			Eigen::VectorXd y_centered = y.array() - mean_y;
			Eigen::MatrixXd X_centered = X;
			for (idx_t j = 0; j < p; j++) {
				X_centered.col(j).array() -= x_means(j);
			}

			ols_result = RankDeficientOls::FitWithStdErrors(y_centered, X_centered);

			// Compute intercept (using only non-aliased features)
			double beta_dot_xmean = 0.0;
			for (idx_t j = 0; j < p; j++) {
				if (!ols_result.is_aliased[j]) {
					beta_dot_xmean += ols_result.coefficients[j] * x_means(j);
				}
			}
			intercept = mean_y - beta_dot_xmean;
		} else {
			// No intercept: solve directly on raw data
			ols_result = RankDeficientOls::FitWithStdErrors(y, X);
			intercept = 0.0;
			x_means = Eigen::VectorXd::Zero(p);
		}

		// Compute predictions using only non-aliased features
		Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(n, intercept);
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				y_pred += ols_result.coefficients[j] * X.col(j);
			}
		}

		// Compute R² and adjusted R² using effective rank
		Eigen::VectorXd residuals = y - y_pred;
		double ss_res = residuals.squaredNorm();

		double ss_tot;
		if (state.options.intercept) {
			double mean_y = y.mean();
			ss_tot = (y.array() - mean_y).square().sum();
		} else {
			// No intercept: total sum of squares from zero
			ss_tot = y.squaredNorm();
		}

		double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

		// Adjusted R²: rank already includes intercept if fitted
		idx_t df_model = ols_result.rank + (state.options.intercept ? 1 : 0);
		double adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);

		// Store coefficients in child vector (NaN for aliased -> will be NULL)
		auto coef_data = FlatVector::GetData<double>(coef_child);
		auto &coef_validity = FlatVector::Validity(coef_child);
		for (idx_t j = 0; j < p; j++) {
			if (std::isnan(ols_result.coefficients[j])) {
				// Aliased coefficient -> set as invalid (NULL)
				coef_validity.SetInvalid(list_offset + j);
				coef_data[list_offset + j] = 0.0; // Placeholder value
			} else {
				coef_data[list_offset + j] = ols_result.coefficients[j];
			}
		}

		// Set list entry for coefficients
		list_entries[result_idx] = list_entry_t {list_offset, p};

		// Compute extended metadata
		// 1. MSE
		idx_t df_residual = n - df_model;
		double mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

		// 2. Intercept standard error
		double intercept_se = std::numeric_limits<double>::quiet_NaN();
		if (state.options.intercept && ols_result.has_std_errors && df_residual > 0) {
			// SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^-1 * x_mean))
			// Approximation: SE(intercept) ≈ sqrt(MSE / n) for centered data
			intercept_se = std::sqrt(mse / n);
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
		for (idx_t j = 0; j < p; j++) {
			if (ols_result.has_std_errors && !std::isnan(ols_result.std_errors(j))) {
				coef_se_data[list_offset + j] = ols_result.std_errors(j);
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

	// Handle intercept option
	double intercept = 0.0;
	RankDeficientOlsResult ols_result;

	if (options.intercept) {
		// With intercept: center data, solve, compute intercept
		double mean_y = y.mean();
		Eigen::VectorXd x_means = X.colwise().mean();

		Eigen::VectorXd y_centered = y.array() - mean_y;
		Eigen::MatrixXd X_centered = X;
		for (idx_t j = 0; j < p; j++) {
			X_centered.col(j).array() -= x_means(j);
		}

		ols_result = RankDeficientOls::Fit(y_centered, X_centered);

		// Compute intercept (using only non-aliased features)
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			ANOFOX_DEBUG("OLS: j=" << j << ", is_aliased=" << ols_result.is_aliased[j]
			             << ", coef=" << ols_result.coefficients[j] << ", x_mean=" << x_means(j));
			if (!ols_result.is_aliased[j]) {
				beta_dot_xmean += ols_result.coefficients[j] * x_means(j);
			}
		}
		ANOFOX_DEBUG("OLS: mean_y=" << mean_y << ", beta_dot_xmean=" << beta_dot_xmean);
		intercept = mean_y - beta_dot_xmean;
		ANOFOX_DEBUG("OLS: intercept=" << intercept << ", rank=" << ols_result.rank);
	} else {
		// No intercept: solve directly on raw data
		ols_result = RankDeficientOls::Fit(y, X);
		intercept = 0.0;
	}

	// Compute predictions using only non-aliased features
	Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(n, intercept);
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			y_pred += ols_result.coefficients[j] * X.col(j);
		}
	}

	// Compute R² and adjusted R²
	Eigen::VectorXd residuals = y - y_pred;
	double ss_res = residuals.squaredNorm();

	double ss_tot;
	if (options.intercept) {
		double mean_y = y.mean();
		ss_tot = (y.array() - mean_y).square().sum();
	} else {
		// No intercept: total sum of squares from zero
		ss_tot = y.squaredNorm();
	}

	double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

	// Adjusted R²
	idx_t df_model = ols_result.rank + (options.intercept ? 1 : 0);
	double adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);

	// Store coefficients in list
	auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
	auto &coef_child = ListVector::GetEntry(coef_list);
	auto coef_data = FlatVector::GetData<double>(coef_child);
	auto &coef_validity = FlatVector::Validity(coef_child);

	idx_t list_offset = rid * p; // Each result gets p coefficients
	ListVector::Reserve(coef_list, (rid + 1) * p);

	for (idx_t j = 0; j < p; j++) {
		if (std::isnan(ols_result.coefficients[j])) {
			coef_validity.SetInvalid(list_offset + j);
			coef_data[list_offset + j] = 0.0;
		} else {
			coef_data[list_offset + j] = ols_result.coefficients[j];
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
