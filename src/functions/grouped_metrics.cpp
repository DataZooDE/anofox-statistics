#include "grouped_metrics.hpp"
#include "../bridge/type_converter.hpp"
#include "../bridge/memory_manager.hpp"
#include "../utils/validation.hpp"
#include "../utils/tracing.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

#include <vector>
#include <map>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Bind data structure for grouped metrics function
 */
struct GroupedMetricsBindData : public FunctionData {
	// Input parameters
	std::vector<std::string> group_col_names;
	std::string y_actual_col_name;
	std::string y_pred_col_name;

	// Result structure for each group
	struct GroupResult {
		idx_t group_id;
		std::vector<std::string> group_key;
		double r_squared;
		double mse;
		double rmse;
		double mae;
		idx_t n_obs;
	};

	std::vector<GroupResult> results;
	idx_t result_index;

	GroupedMetricsBindData() : result_index(0) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<GroupedMetricsBindData>();
		result->group_col_names = group_col_names;
		result->y_actual_col_name = y_actual_col_name;
		result->y_pred_col_name = y_pred_col_name;
		result->results = results;
		result->result_index = result_index;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

unique_ptr<FunctionData> GroupedMetricsFunction::GroupedMetricsBind(ClientContext &context,
                                                                    TableFunctionBindInput &input) {
	ANOFOX_INFO("Starting grouped metrics bind phase");

	auto result = make_uniq<GroupedMetricsBindData>();

	// Parse input parameters
	if (input.inputs.size() != 4) {
		throw InvalidInputException("anofox_grouped_metrics requires exactly 4 parameters: "
		                            "data, group_cols, y_actual_col, y_pred_col");
	}

	// Get the input table expression
	auto &table_expr = input.inputs[0];

	// Get group column names
	if (input.inputs[1]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("group_cols must be a constant array");
	}
	auto &group_cols_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[1]);
	if (group_cols_const.value.type() != LogicalType::LIST) {
		throw InvalidInputException("group_cols must be an array of strings");
	}

	auto &group_cols_list = ListVector::GetEntry(group_cols_const.value);
	for (idx_t i = 0; i < ListVector::GetListSize(group_cols_const.value); i++) {
		if (FlatVector::IsNull(group_cols_list, i)) {
			throw InvalidInputException("group_cols contains NULL values");
		}
		auto col_str = FlatVector::GetData<string_t>(group_cols_list)[i];
		result->group_col_names.push_back(col_str.GetString());
	}

	if (result->group_col_names.empty()) {
		throw InvalidInputException("group_cols must not be empty");
	}

	// Get Y actual column name
	if (input.inputs[2]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("y_actual_col must be a constant string");
	}
	auto &y_actual_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[2]);
	if (y_actual_const.value.type() != LogicalType::VARCHAR) {
		throw InvalidInputException("y_actual_col must be a string");
	}
	result->y_actual_col_name = y_actual_const.value.ToString();

	// Get Y predicted column name
	if (input.inputs[3]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("y_pred_col must be a constant string");
	}
	auto &y_pred_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[3]);
	if (y_pred_const.value.type() != LogicalType::VARCHAR) {
		throw InvalidInputException("y_pred_col must be a string");
	}
	result->y_pred_col_name = y_pred_const.value.ToString();

	ANOFOX_DEBUG("Grouped metrics parameters: group_cols=" + std::to_string(result->group_col_names.size()) +
	             ", y_actual=" + result->y_actual_col_name + ", y_pred=" + result->y_pred_col_name);

	ANOFOX_INFO("Grouped metrics bind phase completed");

	return std::move(result);
}

void GroupedMetricsFunction::GroupedMetricsExecute(ClientContext &context, TableFunctionInput &data,
                                                   DataChunk &output) {
	ANOFOX_DEBUG("Grouped metrics execute phase");

	auto &bind_data = dynamic_cast<GroupedMetricsBindData &>(*data.bind_data);

	// If we've returned all results, we're done
	if (bind_data.result_index >= bind_data.results.size()) {
		return;
	}

	// Stream results one group at a time
	idx_t output_idx = 0;
	while (output_idx < 2048 && bind_data.result_index < bind_data.results.size()) {
		auto &group_result = bind_data.results[bind_data.result_index];

		// Column 0: group_id
		auto &group_id_vec = output.data[0];
		group_id_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(group_id_vec)[output_idx] = group_result.group_id;

		// Column 1: group_key (VARCHAR[])
		auto &group_key_vec = output.data[1];
		group_key_vec.SetVectorType(VectorType::FLAT_VECTOR);
		auto group_key_data = FlatVector::GetData<list_entry_t>(group_key_vec);
		group_key_data[output_idx].offset = output_idx;
		group_key_data[output_idx].length = group_result.group_key.size();

		// Column 2: r_squared
		auto &r2_vec = output.data[2];
		r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(r2_vec)[output_idx] = group_result.r_squared;

		// Column 3: mse
		auto &mse_vec = output.data[3];
		mse_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(mse_vec)[output_idx] = group_result.mse;

		// Column 4: rmse
		auto &rmse_vec = output.data[4];
		rmse_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(rmse_vec)[output_idx] = group_result.rmse;

		// Column 5: mae
		auto &mae_vec = output.data[5];
		mae_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(mae_vec)[output_idx] = group_result.mae;

		// Column 6: n_obs
		auto &n_obs_vec = output.data[6];
		n_obs_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(n_obs_vec)[output_idx] = group_result.n_obs;

		bind_data.result_index++;
		output_idx++;
	}

	output.SetCardinality(output_idx);

	ANOFOX_DEBUG("Grouped metrics execute completed");
}

void GroupedMetricsFunction::Register(ExtensionLoader &loader) {
	ANOFOX_INFO("Registering anofox_grouped_metrics table function");

	auto grouped_metrics_func =
	    make_uniq<TableFunction>("anofox_grouped_metrics",
	                             std::vector<LogicalType> {LogicalType::TABLE, LogicalType::LIST(LogicalType::VARCHAR),
	                                                       LogicalType::VARCHAR, LogicalType::VARCHAR},
	                             GroupedMetricsExecute, GroupedMetricsBind);

	grouped_metrics_func->bind_replace_projection = true;

	loader.RegisterTableFunction(std::move(grouped_metrics_func));

	ANOFOX_INFO("anofox_grouped_metrics registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
