#include "vif_aggregate.hpp"
#include "../utils/tracing.hpp"
#include "../utils/rank_deficient_ols.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * VIF Aggregate: anofox_stats_vif_agg(x DOUBLE[]) -> STRUCT
 *
 * Accumulates feature vectors across rows in a GROUP BY,
 * then computes Variance Inflation Factor for each feature to detect multicollinearity.
 *
 * VIF_j = 1 / (1 - R²_j)
 * where R²_j is from regressing feature j against all other features.
 */

struct VifAggregateState {
	vector<vector<double>> x_matrix; // Each row is an observation, columns are features
	idx_t n_features = 0;

	void Reset() {
		x_matrix.clear();
		n_features = 0;
	}
};

static void VifInitialize(const AggregateFunction &function, data_ptr_t state_ptr) {
	auto state = reinterpret_cast<VifAggregateState *>(state_ptr);
	new (state) VifAggregateState();
}

static void VifUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                      idx_t count) {

	UnifiedVectorFormat state_data;
	state_vector.ToUnifiedFormat(count, state_data);
	auto states = UnifiedVectorFormat::GetData<VifAggregateState *>(state_data);

	auto &x_array_vector = inputs[0];

	UnifiedVectorFormat x_array_data;
	x_array_vector.ToUnifiedFormat(count, x_array_data);

	for (idx_t i = 0; i < count; i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto &state = *states[state_idx];

		auto x_array_idx = x_array_data.sel->get_index(i);

		if (!x_array_data.validity.RowIsValid(x_array_idx)) {
			continue;
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
			continue; // Skip mismatched dimensions
		}

		state.x_matrix.push_back(features);
	}
}

static void VifCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	auto source_ptr = FlatVector::GetData<VifAggregateState *>(source);
	auto target_ptr = FlatVector::GetData<VifAggregateState *>(target);

	for (idx_t i = 0; i < count; i++) {
		auto &source_state = *source_ptr[i];
		auto &target_state = *target_ptr[i];

		target_state.x_matrix.insert(target_state.x_matrix.end(), source_state.x_matrix.begin(),
		                             source_state.x_matrix.end());

		if (target_state.n_features == 0) {
			target_state.n_features = source_state.n_features;
		}
	}
}

static string GetSeverityLabel(double vif) {
	if (std::isnan(vif) || std::isinf(vif)) {
		return "perfect"; // Perfect multicollinearity
	} else if (vif < 5.0) {
		return "low";
	} else if (vif < 10.0) {
		return "moderate";
	} else {
		return "high";
	}
}

static void VifFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {

	auto states = FlatVector::GetData<VifAggregateState *>(state_vector);
	auto &result_validity = FlatVector::Validity(result);

	// Result: STRUCT(variable_id BIGINT[], variable_name VARCHAR[], vif DOUBLE[], severity VARCHAR[])
	auto &struct_entries = StructVector::GetEntries(result);
	auto &variable_id_list = *struct_entries[0];
	auto &variable_name_list = *struct_entries[1];
	auto &vif_list = *struct_entries[2];
	auto &severity_list = *struct_entries[3];

	auto variable_id_entries = FlatVector::GetData<list_entry_t>(variable_id_list);
	auto &variable_id_child = ListVector::GetEntry(variable_id_list);
	auto variable_id_child_data = FlatVector::GetData<int64_t>(variable_id_child);

	auto variable_name_entries = FlatVector::GetData<list_entry_t>(variable_name_list);
	auto &variable_name_child = ListVector::GetEntry(variable_name_list);

	auto vif_entries = FlatVector::GetData<list_entry_t>(vif_list);
	auto &vif_child = ListVector::GetEntry(vif_list);
	auto vif_child_data = FlatVector::GetData<double>(vif_child);
	auto &vif_child_validity = FlatVector::Validity(vif_child);

	auto severity_entries = FlatVector::GetData<list_entry_t>(severity_list);
	auto &severity_child = ListVector::GetEntry(severity_list);

	idx_t list_offset = 0;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[i];
		idx_t result_idx = offset + i;

		idx_t n = state.x_matrix.size();
		idx_t p = state.n_features;

		if (n < p + 2 || p == 0) {
			result_validity.SetInvalid(result_idx);
			variable_id_entries[result_idx] = list_entry_t {list_offset, 0};
			variable_name_entries[result_idx] = list_entry_t {list_offset, 0};
			vif_entries[result_idx] = list_entry_t {list_offset, 0};
			severity_entries[result_idx] = list_entry_t {list_offset, 0};
			continue;
		}

		// Build feature matrix X
		Eigen::MatrixXd X(n, p);
		for (idx_t row = 0; row < n; row++) {
			for (idx_t col = 0; col < p; col++) {
				X(row, col) = state.x_matrix[row][col];
			}
		}

		// Compute VIF for each feature
		ListVector::Reserve(variable_id_list, list_offset + p);
		ListVector::Reserve(vif_list, list_offset + p);

		for (idx_t j = 0; j < p; j++) {
			// Variable ID (1-indexed)
			variable_id_child_data[list_offset + j] = static_cast<int64_t>(j + 1);

			// Variable name
			string var_name = "x" + std::to_string(j + 1);
			Value var_name_val(var_name);
			ListVector::PushBack(variable_name_child, var_name_val);

			// Compute VIF: regress X_j against all other X features
			// Extract feature j as response
			Eigen::VectorXd y_j = X.col(j);

			// Build design matrix with all features except j
			Eigen::MatrixXd X_other(n, p - 1);
			idx_t col_idx = 0;
			for (idx_t k = 0; k < p; k++) {
				if (k != j) {
					X_other.col(col_idx++) = X.col(k);
				}
			}

			// Fit OLS: y_j ~ X_other (with intercept)
			double y_mean = y_j.mean();
			Eigen::VectorXd x_means = X_other.colwise().mean();
			Eigen::VectorXd y_centered = y_j.array() - y_mean;
			Eigen::MatrixXd X_centered(n, p - 1);
			for (idx_t row = 0; row < n; row++) {
				for (idx_t col = 0; col < p - 1; col++) {
					X_centered(row, col) = X_other(row, col) - x_means(col);
				}
			}

			auto fit_result = RankDeficientOls::Fit(y_centered, X_centered);
			double r2_j = fit_result.r_squared;

			// VIF = 1 / (1 - R²)
			double vif;
			string severity;
			if (r2_j >= 0.9999) { // Near-perfect multicollinearity
				vif_child_validity.SetInvalid(list_offset + j);
				vif_child_data[list_offset + j] = 0.0; // Placeholder
				severity = "perfect";
			} else {
				vif = 1.0 / (1.0 - r2_j);
				vif_child_data[list_offset + j] = vif;
				severity = GetSeverityLabel(vif);
			}

			// Severity label
			Value severity_val(severity);
			ListVector::PushBack(severity_child, severity_val);
		}

		variable_id_entries[result_idx] = list_entry_t {list_offset, p};
		variable_name_entries[result_idx] = list_entry_t {list_offset, p};
		vif_entries[result_idx] = list_entry_t {list_offset, p};
		severity_entries[result_idx] = list_entry_t {list_offset, p};
		list_offset += p;

		ANOFOX_DEBUG("VIF aggregate: n=" << n << ", p=" << p);
	}

	ListVector::SetListSize(variable_id_list, list_offset);
	ListVector::SetListSize(variable_name_list, list_offset);
	ListVector::SetListSize(vif_list, list_offset);
	ListVector::SetListSize(severity_list, list_offset);
}

void VifAggregateFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_stats_vif_agg (with alias vif_agg)");

	child_list_t<LogicalType> vif_struct_fields;
	vif_struct_fields.push_back(make_pair("variable_id", LogicalType::LIST(LogicalType::BIGINT)));
	vif_struct_fields.push_back(make_pair("variable_name", LogicalType::LIST(LogicalType::VARCHAR)));
	vif_struct_fields.push_back(make_pair("vif", LogicalType::LIST(LogicalType::DOUBLE)));
	vif_struct_fields.push_back(make_pair("severity", LogicalType::LIST(LogicalType::VARCHAR)));

	AggregateFunction anofox_stats_vif_agg(
	    "anofox_stats_vif_agg", {LogicalType::LIST(LogicalType::DOUBLE)}, // x features
	    LogicalType::STRUCT(vif_struct_fields), AggregateFunction::StateSize<VifAggregateState>, VifInitialize,
	    VifUpdate, VifCombine, VifFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING);

	loader.RegisterFunction(anofox_stats_vif_agg);

	// Register alias
	AggregateFunction vif_agg_alias(
	    "vif_agg", {LogicalType::LIST(LogicalType::DOUBLE)}, // x features
	    LogicalType::STRUCT(vif_struct_fields), AggregateFunction::StateSize<VifAggregateState>, VifInitialize,
	    VifUpdate, VifCombine, VifFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING);

	loader.RegisterFunction(vif_agg_alias);

	ANOFOX_DEBUG("anofox_stats_vif_agg registered successfully with alias vif_agg");
}

} // namespace anofox_statistics
} // namespace duckdb
