#include "wls_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"
#include "../utils/options_parser.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <Eigen/Dense>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * WLS Aggregate: anofox_statistics_wls_agg(y DOUBLE, x DOUBLE[], weights DOUBLE) -> STRUCT
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

		// Handle intercept option
		double intercept = 0.0;
		RankDeficientOlsResult ols_result;
		double sum_weights = w.sum();
		Eigen::VectorXd x_means; // Store for extended metadata

		if (state.options.intercept) {
			// With intercept: compute weighted means and center data
			double y_weighted_mean = (w.array() * y.array()).sum() / sum_weights;
			Eigen::VectorXd x_weighted_means = Eigen::VectorXd::Zero(p);
			for (idx_t j = 0; j < p; j++) {
				x_weighted_means(j) = (w.array() * X.col(j).array()).sum() / sum_weights;
			}
			x_means = x_weighted_means; // Store weighted means for later

			// Center data
			Eigen::MatrixXd X_centered = X;
			Eigen::VectorXd y_centered = y;
			for (idx_t row = 0; row < n; row++) {
				y_centered(row) = y(row) - y_weighted_mean;
				for (idx_t col = 0; col < p; col++) {
					X_centered(row, col) = X(row, col) - x_weighted_means(col);
				}
			}

			// Apply sqrt(W) transformation
			Eigen::VectorXd sqrt_w = w.array().sqrt();
			Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_centered;
			Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_centered;

			// Solve WLS on centered data
			ols_result = RankDeficientOls::FitWithStdErrors(y_weighted, X_weighted);

			// Compute intercept
			double beta_dot_xmean = 0.0;
			for (idx_t j = 0; j < p; j++) {
				if (!ols_result.is_aliased[j]) {
					beta_dot_xmean += ols_result.coefficients[j] * x_weighted_means(j);
				}
			}
			intercept = y_weighted_mean - beta_dot_xmean;
		} else {
			// No intercept: solve directly on weighted data
			Eigen::VectorXd sqrt_w = w.array().sqrt();
			Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X;
			Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y;

			ols_result = RankDeficientOls::FitWithStdErrors(y_weighted, X_weighted);
			intercept = 0.0;
			x_means = Eigen::VectorXd::Zero(p);
		}

		// Compute predictions
		Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(n, intercept);
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				y_pred += ols_result.coefficients[j] * X.col(j);
			}
		}

		// Compute weighted statistics
		Eigen::VectorXd residuals = y - y_pred;
		double ss_res = residuals.squaredNorm(); // Unweighted for MSE calculation
		double ss_res_weighted = (w.array() * residuals.array().square()).sum();

		double ss_tot_weighted;
		if (state.options.intercept) {
			double y_weighted_mean = (w.array() * y.array()).sum() / sum_weights;
			ss_tot_weighted = (w.array() * (y.array() - y_weighted_mean).square()).sum();
		} else {
			// No intercept: total sum of squares from zero
			ss_tot_weighted = (w.array() * y.array().square()).sum();
		}

		double r2 = (ss_tot_weighted > 1e-10) ? (1.0 - ss_res_weighted / ss_tot_weighted) : 0.0;

		// Adjusted R²: rank already includes intercept if fitted
		idx_t df_model = ols_result.rank;
		double adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);
		double weighted_mse = ss_res_weighted / sum_weights;

		// Store coefficients
		auto coef_data = FlatVector::GetData<double>(coef_child);
		auto &coef_validity = FlatVector::Validity(coef_child);
		for (idx_t j = 0; j < p; j++) {
			if (std::isnan(ols_result.coefficients[j])) {
				coef_validity.SetInvalid(list_offset + j);
				coef_data[list_offset + j] = 0.0;
			} else {
				coef_data[list_offset + j] = ols_result.coefficients[j];
			}
		}

		list_entries[result_idx] = list_entry_t {list_offset, p};

		// Compute extended metadata
		// 1. MSE (unweighted MSE for standard errors)
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
		weighted_mse_data[result_idx] = weighted_mse;
		n_data[result_idx] = n;
		mse_data[result_idx] = mse;
		intercept_se_data[result_idx] = intercept_se;
		df_resid_data[result_idx] = df_residual;

		ANOFOX_DEBUG("WLS aggregate: n=" << n << ", p=" << p << ", r2=" << r2);
	}

	ListVector::SetListSize(coef_list, list_offset);
	ListVector::SetListSize(x_means_list, list_offset);
	ListVector::SetListSize(coef_se_list, list_offset);
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

	// Handle intercept option
	double intercept = 0.0;
	RankDeficientOlsResult ols_result;
	double sum_weights = w.sum();

	if (options.intercept) {
		// With intercept: compute weighted means and center data
		double y_weighted_mean = (w.array() * y.array()).sum() / sum_weights;
		Eigen::VectorXd x_weighted_means = Eigen::VectorXd::Zero(p);
		for (idx_t j = 0; j < p; j++) {
			x_weighted_means(j) = (w.array() * X.col(j).array()).sum() / sum_weights;
		}

		// Center data
		Eigen::MatrixXd X_centered = X;
		Eigen::VectorXd y_centered = y;
		for (idx_t row = 0; row < n; row++) {
			y_centered(row) = y(row) - y_weighted_mean;
			for (idx_t col = 0; col < p; col++) {
				X_centered(row, col) = X(row, col) - x_weighted_means(col);
			}
		}

		// Apply sqrt(W) transformation
		Eigen::VectorXd sqrt_w = w.array().sqrt();
		Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_centered;
		Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_centered;

		// Solve WLS on centered data
		ols_result = RankDeficientOls::FitWithStdErrors(y_weighted, X_weighted);

		// Compute intercept
		double beta_dot_xmean = 0.0;
		for (idx_t j = 0; j < p; j++) {
			if (!ols_result.is_aliased[j]) {
				beta_dot_xmean += ols_result.coefficients[j] * x_weighted_means(j);
			}
		}
		intercept = y_weighted_mean - beta_dot_xmean;
	} else {
		// No intercept: solve directly on weighted data
		Eigen::VectorXd sqrt_w = w.array().sqrt();
		Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X;
		Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y;

		ols_result = RankDeficientOls::FitWithStdErrors(y_weighted, X_weighted);
		intercept = 0.0;
	}

	// Compute predictions
	Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(n, intercept);
	for (idx_t j = 0; j < p; j++) {
		if (!ols_result.is_aliased[j]) {
			y_pred += ols_result.coefficients[j] * X.col(j);
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
	idx_t df_model = ols_result.rank;
	double adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);
	double weighted_mse = ss_res_weighted / sum_weights;

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

	AggregateFunction anofox_statistics_wls_agg(
	    "anofox_statistics_wls_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::STRUCT(wls_struct_fields), AggregateFunction::StateSize<WlsAggregateState>, WlsInitialize,
	    WlsUpdate, WlsCombine, WlsFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr,
	    nullptr, WlsWindow, nullptr, nullptr);
	loader.RegisterFunction(anofox_statistics_wls_agg);

	ANOFOX_DEBUG("WLS aggregate function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
