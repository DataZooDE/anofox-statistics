#include "rls_aggregate.hpp"
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
 * RLS Aggregate: anofox_statistics_rls_agg(y DOUBLE, x DOUBLE[], forgetting_factor DOUBLE) -> STRUCT
 *
 * Accumulates (y, x[], forgetting_factor) tuples across rows in a GROUP BY,
 * then computes Recursive Least Squares on finalize.
 *
 * Note: forgetting_factor should be the same for all rows in a group.
 * We use the forgetting_factor from the first row.
 */

struct RlsAggregateState {
	vector<double> y_values;
	vector<vector<double>> x_matrix;
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

static void RlsInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<RlsAggregateState *>(state_ptr);
	new (state) RlsAggregateState();
}

static void RlsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                      idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<RlsAggregateState *>(state_data);

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

	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto y_idx = y_data.sel->get_index(i);
		auto x_array_idx = x_array_data.sel->get_index(i);
		auto options_idx = options_data.sel->get_index(i);

		if (!y_data.validity.RowIsValid(y_idx) || !x_array_data.validity.RowIsValid(x_array_idx)) {
			continue;
		}

		// Initialize options from first row
		if (!state.options_initialized && options_data.validity.RowIsValid(options_idx)) {
			auto options_value = options_vector.GetValue(options_idx);
			state.options = RegressionOptions::ParseFromMap(options_value);
			state.options_initialized = true;
		}

		auto x_array_entry = UnifiedVectorFormat::GetData<list_entry_t>(x_array_data)[x_array_idx];
		auto &x_child = ListVector::GetEntry(x_array_vector);

		UnifiedVectorFormat x_child_data;
		x_child.ToUnifiedFormat(ListVector::GetListSize(x_array_vector), x_child_data);
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

		if (features.empty()) {
			continue;
		}

		if (state.n_features == 0) {
			state.n_features = features.size();
		} else if (features.size() != state.n_features) {
			continue;
		}

		state.y_values.push_back(y_ptr[y_idx]);
		state.x_matrix.push_back(features);
	}
}

static void RlsCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<RlsAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<RlsAggregateState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

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

static void RlsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {

	auto states = FlatVector::GetData<RlsAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result: STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, forgetting_factor DOUBLE,
	// n_obs BIGINT)
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0];
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto ff_data = FlatVector::GetData<double>(*struct_entries[4]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[5]);
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

		// Use libanostat RLSSolver
		try {
			auto lib_result = bridge::LibanostatWrapper::FitRLS(state.y_values, state.x_matrix, state.options, false);

			// Extract intercept and feature coefficients from libanostat result
			double intercept = bridge::TypeConverters::ExtractIntercept(lib_result, state.options.intercept);
			auto feature_coefs_vec =
			    bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);

			// Compute x_means for prediction (needed for extended metadata)
			Eigen::VectorXd x_means;
			if (state.options.intercept) {
				Eigen::MatrixXd X(n, p);
				for (idx_t row = 0; row < n; row++) {
					for (idx_t col = 0; col < p; col++) {
						X(row, col) = state.x_matrix[row][col];
					}
				}
				x_means = X.colwise().mean();
			} else {
				x_means = Eigen::VectorXd::Zero(p);
			}

			// Extract all fit statistics from libanostat (no recomputation)
			idx_t rank = lib_result.rank;
			double r2 = lib_result.r_squared;
			double adj_r2 = lib_result.adj_r_squared;
			double mse = lib_result.mse;
			idx_t df_residual = lib_result.df_residual();

			// Store coefficients
			auto coef_data = FlatVector::GetData<double>(coef_child);
			for (idx_t j = 0; j < p; j++) {
				size_t coef_idx = state.options.intercept ? (j + 1) : j;
				if (lib_result.is_aliased[coef_idx]) {
					coef_data[list_offset + j] = std::numeric_limits<double>::quiet_NaN();
				} else {
					coef_data[list_offset + j] = lib_result.coefficients[coef_idx];
				}
			}

			list_entries[result_idx] = list_entry_t {list_offset, p};

			// RLS is a biased estimator (like Ridge/Elastic Net), so standard errors are not well-defined
			double intercept_se = std::numeric_limits<double>::quiet_NaN();

			// Store x_train_means
			auto x_means_data = FlatVector::GetData<double>(x_means_child);
			for (idx_t j = 0; j < p; j++) {
				x_means_data[list_offset + j] = x_means(j);
			}
			x_means_list_entries[result_idx] = list_entry_t {list_offset, p};

			// Store coefficient_std_errors (NULL for RLS - standard errors not well-defined)
			auto coef_se_data = FlatVector::GetData<double>(coef_se_child);
			auto &coef_se_validity = FlatVector::Validity(coef_se_child);
			for (idx_t j = 0; j < p; j++) {
				coef_se_validity.SetInvalid(list_offset + j);
				coef_se_data[list_offset + j] = 0.0;
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
			ff_data[result_idx] = state.options.forgetting_factor;
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

			ANOFOX_DEBUG("RLS aggregate: n=" << n << ", p=" << p << ", ff=" << state.options.forgetting_factor
			                                 << ", r2=" << r2);
		} catch (...) {
			result_validity.SetInvalid(result_idx);
			list_entries[result_idx] = list_entry_t {list_offset, 0};
		}
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
 * Window callback for RLS aggregate
 * Computes Recursive Least Squares with sequential updates on the current window frame(s) for each row
 */
static void RlsWindow(AggregateInputData &aggr_input_data, const WindowPartitionInput &partition,
                      const_data_ptr_t g_state, data_ptr_t l_state, const SubFrames &subframes, Vector &result,
                      idx_t rid) {

	auto &result_validity = FlatVector::Validity(result);

	// Result: STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, forgetting_factor DOUBLE, n_obs
	// BIGINT)
	auto &struct_entries = StructVector::GetEntries(result);
	auto &coef_list = *struct_entries[0];
	auto intercept_data = FlatVector::GetData<double>(*struct_entries[1]);
	auto r2_data = FlatVector::GetData<double>(*struct_entries[2]);
	auto adj_r2_data = FlatVector::GetData<double>(*struct_entries[3]);
	auto ff_data = FlatVector::GetData<double>(*struct_entries[4]);
	auto n_data = FlatVector::GetData<int64_t>(*struct_entries[5]);

	// Extract data for the entire partition
	vector<double> all_y;
	vector<vector<double>> all_x;
	RegressionOptions options;
	bool options_initialized = false;
	idx_t n_features = 0;

	ColumnDataScanState scan_state;
	partition.inputs->InitializeScan(scan_state);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), partition.inputs->Types());

	while (partition.inputs->Scan(scan_state, chunk)) {
		auto &y_chunk = chunk.data[0];
		auto &x_array_chunk = chunk.data[1];
		auto &options_chunk = chunk.data[2];

		UnifiedVectorFormat y_data, x_array_data, options_data;
		y_chunk.ToUnifiedFormat(chunk.size(), y_data);
		x_array_chunk.ToUnifiedFormat(chunk.size(), x_array_data);
		options_chunk.ToUnifiedFormat(chunk.size(), options_data);

		auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);

		for (idx_t i = 0; i < chunk.size(); i++) {
			auto y_idx = y_data.sel->get_index(i);
			auto x_array_idx = x_array_data.sel->get_index(i);
			auto options_idx = options_data.sel->get_index(i);

			if (!options_initialized && options_data.validity.RowIsValid(options_idx)) {
				auto options_value = options_chunk.GetValue(options_idx);
				options = RegressionOptions::ParseFromMap(options_value);
				options_initialized = true;
			}

			if (!y_data.validity.RowIsValid(y_idx) || !x_array_data.validity.RowIsValid(x_array_idx)) {
				all_y.push_back(std::numeric_limits<double>::quiet_NaN());
				all_x.push_back(vector<double>());
				continue;
			}

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
		}
	}

	// Extract window data
	vector<double> window_y;
	vector<vector<double>> window_x;

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

	if (n < p + 1 || p == 0) {
		result_validity.SetInvalid(rid);
		auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
		list_entries[rid] = list_entry_t {0, 0};
		return;
	}

	// Build matrices
	Eigen::MatrixXd X(n, p);
	Eigen::VectorXd y(n);
	for (idx_t row = 0; row < n; row++) {
		y(row) = window_y[row];
		for (idx_t col = 0; col < p; col++) {
			X(row, col) = window_x[row][col];
		}
	}

	// Handle intercept option and perform RLS
	double intercept = 0.0;
	Eigen::VectorXd beta;
	Eigen::VectorXd x_means;

	if (options.intercept) {
		// With intercept: center data first, then do RLS
		double y_mean = y.mean();
		x_means = X.colwise().mean();

		Eigen::VectorXd y_centered = y.array() - y_mean;
		Eigen::MatrixXd X_centered = X;
		for (idx_t j = 0; j < p; j++) {
			X_centered.col(j).array() -= x_means(j);
		}

		// RLS: Sequential updates on centered data
		beta = Eigen::VectorXd::Zero(p);
		double initial_p = 1000.0;
		Eigen::MatrixXd P = Eigen::MatrixXd::Identity(p, p) * initial_p;

		// Sequential updates
		for (idx_t t = 0; t < n; t++) {
			Eigen::VectorXd x_t = X_centered.row(t).transpose();
			double y_t = y_centered(t);

			// Prediction error
			double y_pred = x_t.dot(beta);
			double error = y_t - y_pred;

			// Kalman gain
			double denominator = options.forgetting_factor + x_t.dot(P * x_t);
			Eigen::VectorXd K = P * x_t / denominator;

			// Update
			beta = beta + K * error;
			P = (P - K * x_t.transpose() * P) / options.forgetting_factor;
		}

		// Compute intercept from centered coefficients
		double beta_dot_xmean = beta.dot(x_means);
		intercept = y_mean - beta_dot_xmean;
	} else {
		// No intercept: RLS on raw data
		beta = Eigen::VectorXd::Zero(p);
		double initial_p = 1000.0;
		Eigen::MatrixXd P = Eigen::MatrixXd::Identity(p, p) * initial_p;

		// Sequential updates
		for (idx_t t = 0; t < n; t++) {
			Eigen::VectorXd x_t = X.row(t).transpose();
			double y_t = y(t);

			// Prediction error
			double y_pred = x_t.dot(beta);
			double error = y_t - y_pred;

			// Kalman gain
			double denominator = options.forgetting_factor + x_t.dot(P * x_t);
			Eigen::VectorXd K = P * x_t / denominator;

			// Update
			beta = beta + K * error;
			P = (P - K * x_t.transpose() * P) / options.forgetting_factor;
		}

		intercept = 0.0;
		x_means = Eigen::VectorXd::Zero(p);
	}

	// Compute predictions
	Eigen::VectorXd y_pred = X * beta;
	y_pred.array() += intercept;

	// Compute statistics
	Eigen::VectorXd residuals = y - y_pred;
	double ss_res = residuals.squaredNorm();
	double ss_tot = options.intercept ? (y.array() - y.mean()).square().sum() : y.squaredNorm();
	double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
	idx_t df_model = p + (options.intercept ? 1 : 0);
	double adj_r2 = 1.0 - (1.0 - r2) * static_cast<double>(n - 1) / static_cast<double>(n - df_model);

	// Store coefficients
	auto list_entries = FlatVector::GetData<list_entry_t>(coef_list);
	auto &coef_child = ListVector::GetEntry(coef_list);
	auto coef_data = FlatVector::GetData<double>(coef_child);

	idx_t list_offset = rid * p;
	ListVector::Reserve(coef_list, (rid + 1) * p);

	for (idx_t j = 0; j < p; j++) {
		coef_data[list_offset + j] = beta(j);
	}

	list_entries[rid] = list_entry_t {list_offset, p};
	intercept_data[rid] = intercept;
	r2_data[rid] = r2;
	adj_r2_data[rid] = adj_r2;
	ff_data[rid] = options.forgetting_factor;
	n_data[rid] = n;

	ANOFOX_DEBUG("RLS window: n=" << n << ", p=" << p << ", ff=" << options.forgetting_factor << ", r2=" << r2);
}

void RlsAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering RLS aggregate function");

	child_list_t<LogicalType> rls_struct_fields;
	rls_struct_fields.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	rls_struct_fields.push_back(make_pair("intercept", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("r2", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("adj_r2", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("forgetting_factor", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("n_obs", LogicalType::BIGINT));
	rls_struct_fields.push_back(make_pair("mse", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("x_train_means", LogicalType::LIST(LogicalType::DOUBLE)));
	rls_struct_fields.push_back(make_pair("coefficient_std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
	rls_struct_fields.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("df_residual", LogicalType::BIGINT));
	// New statistical metrics
	rls_struct_fields.push_back(make_pair("residual_standard_error", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("f_statistic_pvalue", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("aic", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("aicc", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("bic", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("log_likelihood", LogicalType::DOUBLE));
	// Coefficient-level inference
	rls_struct_fields.push_back(make_pair("coefficient_t_statistics", LogicalType::LIST(LogicalType::DOUBLE)));
	rls_struct_fields.push_back(make_pair("coefficient_p_values", LogicalType::LIST(LogicalType::DOUBLE)));
	rls_struct_fields.push_back(make_pair("coefficient_ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
	rls_struct_fields.push_back(make_pair("coefficient_ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
	// Intercept-level inference
	rls_struct_fields.push_back(make_pair("intercept_t_statistic", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("intercept_p_value", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("intercept_ci_lower", LogicalType::DOUBLE));
	rls_struct_fields.push_back(make_pair("intercept_ci_upper", LogicalType::DOUBLE));

	AggregateFunction anofox_statistics_rls_agg(
	    "anofox_statistics_rls_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::STRUCT(rls_struct_fields), AggregateFunction::StateSize<RlsAggregateState>, RlsInitialize,
	    RlsUpdate, RlsCombine, RlsFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, nullptr, nullptr,
	    nullptr, RlsWindow, nullptr, nullptr);
	loader.RegisterFunction(anofox_statistics_rls_agg);

	ANOFOX_DEBUG("RLS aggregate function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
