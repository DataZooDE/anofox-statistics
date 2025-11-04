#include "ols_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"

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

	void Reset() {
		y_values.clear();
		x_matrix.clear();
		n_features = 0;
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

		double mean_x = sum_x / n;
		double mean_y = sum_y / n;

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

		double mean_x = sum_x / n;
		double mean_y = sum_y / n;

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
		double mse = (n > 2) ? (ss_res / (n - 2)) : 0.0;
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

	// Get y and x_array vectors
	auto &y_vector = inputs[0];
	auto &x_array_vector = inputs[1];

	UnifiedVectorFormat y_data;
	UnifiedVectorFormat x_array_data;
	y_vector.ToUnifiedFormat(count, y_data);
	x_array_vector.ToUnifiedFormat(count, x_array_data);

	auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);

	// Process each row
	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_idx = y_data.sel->get_index(i);
		auto x_array_idx = x_array_data.sel->get_index(i);

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
	}
}

static void OlsArrayFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {

	auto states = FlatVector::GetData<OlsArrayAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result is a STRUCT with fields: coefficients (DOUBLE[]), intercept, r2, adj_r2, n_obs
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0]; // LIST of coefficients
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[4]);

	// Get list entry data and child vector
	auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
	auto &coef_child = ListVector::GetEntry(coef_list);
	ListVector::Reserve(coef_list, count * 10); // Reserve space (estimate)

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

		// Use rank-deficient OLS solver (handles constant/aliased features)
		auto ols_result = RankDeficientOls::Fit(y, X);

		// Compute intercept (using only non-aliased features)
		double mean_y = y.mean();
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				double x_mean = X.col(j).mean();
				beta_dot_xmean += ols_result.coefficients[j] * x_mean;
			}
		}
		double intercept = mean_y - beta_dot_xmean;

		// Compute predictions using only non-aliased features
		Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(n);
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				y_pred += ols_result.coefficients[j] * X.col(j);
			}
		}

		// Compute R² and adjusted R² using effective rank
		Eigen::VectorXd residuals = y - y_pred;
		double ss_res = residuals.squaredNorm();
		double ss_tot = (y.array() - mean_y).square().sum();

		double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
		double adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - ols_result.rank - 1);

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

		// Set list entry
		list_entries[result_idx] = list_entry_t {list_offset, p};
		list_offset += p;

		// Fill other struct fields
		intercept_data[result_idx] = intercept;
		r2_data[result_idx] = r2;
		adj_r2_data[result_idx] = adj_r2;
		n_data[result_idx] = n;

		ANOFOX_DEBUG("OLS array aggregate: n=" << n << ", p=" << p << ", r2=" << r2);
	}

	// Set final list size
	ListVector::SetListSize(coef_list, list_offset);
}

static void OlsArrayDestroy(Vector &state_vector, AggregateInputData &aggr_input_data, idx_t count) {
	auto states = FlatVector::GetData<OlsArrayAggregateState *>(state_vector);
	for (idx_t i = 0; i < count; i++) {
		states[i]->~OlsArrayAggregateState();
	}
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
	// DOUBLE, n_obs BIGINT)
	child_list_t<LogicalType> array_fit_struct_fields;
	array_fit_struct_fields.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	array_fit_struct_fields.push_back(make_pair("intercept", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("r2", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("adj_r2", LogicalType::DOUBLE));
	array_fit_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));

	AggregateFunction ols_fit_agg_array(
	    "ols_fit_agg_array", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
	    LogicalType::STRUCT(array_fit_struct_fields), AggregateFunction::StateSize<OlsArrayAggregateState>,
	    OlsArrayInitialize, OlsArrayUpdate, OlsArrayCombine, OlsArrayFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, OlsArrayDestroy, nullptr, nullptr, nullptr,
	    nullptr);
	loader.RegisterFunction(ols_fit_agg_array);

	ANOFOX_DEBUG("All OLS aggregate functions registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
