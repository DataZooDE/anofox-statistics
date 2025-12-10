#include "grouped_ridge_fit.hpp"
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

// #include <anofox/ridge.hpp>  // Library integration in Phase 2
#include <vector>
#include <map>
#include <cmath>
#include <Eigen/Dense>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Bind data structure for grouped ridge fit function
 */
struct GroupedRidgeFitBindData : public FunctionData {
	// Input parameters
	std::vector<std::string> group_col_names;
	std::vector<std::string> x_col_names;
	std::string y_col_name;
	double lambda;
	bool add_intercept;

	// Result structure for each group
	struct GroupResult {
		idx_t group_id;
		std::vector<std::string> group_key;
		Eigen::VectorXd coefficients;
		double intercept;
		double r2;
		double adj_r2;
		double mse;
		double rmse;
		double lambda;
		idx_t n_obs;
		idx_t n_features;
	};

	std::vector<GroupResult> results;
	idx_t result_index;

	GroupedRidgeFitBindData() : lambda(0.0), add_intercept(true), result_index(0) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<GroupedRidgeFitBindData>();
		result->group_col_names = group_col_names;
		result->x_col_names = x_col_names;
		result->y_col_name = y_col_name;
		result->lambda = lambda;
		result->add_intercept = add_intercept;
		result->results = results;
		result->result_index = result_index;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

unique_ptr<FunctionData> GroupedRidgeFitFunction::GroupedRidgeFitBind(ClientContext &context,
                                                                      TableFunctionBindInput &input) {
	ANOFOX_INFO("Starting grouped ridge fit bind phase");

	auto result = make_uniq<GroupedRidgeFitBindData>();

	// Parse input parameters
	if (input.inputs.size() < 5 || input.inputs.size() > 6) {
		throw InvalidInputException("anofox_grouped_ridge_fit requires 5-6 parameters: "
		                            "data, group_cols, x_cols, y_col, lambda, [add_intercept]");
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

	// Get X column names
	if (input.inputs[2]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("x_cols must be a constant array");
	}
	auto &x_cols_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[2]);
	if (x_cols_const.value.type() != LogicalType::LIST) {
		throw InvalidInputException("x_cols must be an array of strings");
	}

	auto &x_cols_list = ListVector::GetEntry(x_cols_const.value);
	for (idx_t i = 0; i < ListVector::GetListSize(x_cols_const.value); i++) {
		if (FlatVector::IsNull(x_cols_list, i)) {
			throw InvalidInputException("x_cols contains NULL values");
		}
		auto col_str = FlatVector::GetData<string_t>(x_cols_list)[i];
		result->x_col_names.push_back(col_str.GetString());
	}

	if (result->x_col_names.empty()) {
		throw InvalidInputException("x_cols must not be empty");
	}

	// Get Y column name
	if (input.inputs[3]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("y_col must be a constant string");
	}
	auto &y_col_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[3]);
	if (y_col_const.value.type() != LogicalType::VARCHAR) {
		throw InvalidInputException("y_col must be a string");
	}
	result->y_col_name = y_col_const.value.ToString();

	// Get lambda (regularization strength)
	if (input.inputs[4]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("lambda must be a constant");
	}
	auto &lambda_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[4]);
	if (lambda_const.value.type() != LogicalType::DOUBLE) {
		throw InvalidInputException("lambda must be a DOUBLE");
	}
	result->lambda = lambda_const.value.GetValue<double>();
	if (result->lambda < 0.0) {
		throw InvalidInputException("lambda must be >= 0");
	}

	// Get add_intercept flag
	if (input.inputs.size() > 5) {
		if (input.inputs[5]->type == ExpressionType::VALUE_CONSTANT) {
			auto &intercept_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[5]);
			result->add_intercept = intercept_const.value.GetValue<bool>();
		}
	}

	ANOFOX_DEBUG("Grouped ridge parameters: group_cols=" + std::to_string(result->group_col_names.size()) +
	             ", x_cols=" + std::to_string(result->x_col_names.size()) +
	             ", lambda=" + std::to_string(result->lambda) + ", intercept=" + std::to_string(result->add_intercept));

	ANOFOX_INFO("Grouped ridge fit bind phase completed");

	return std::move(result);
}

void GroupedRidgeFitFunction::GroupedRidgeFitExecute(ClientContext &context, TableFunctionInput &data,
                                                     DataChunk &output) {
	ANOFOX_DEBUG("Grouped ridge fit execute phase");

	auto &bind_data = dynamic_cast<GroupedRidgeFitBindData &>(*data.bind_data);

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

		// Column 2: coefficients (DOUBLE[])
		auto &coeff_vec = output.data[2];
		coeff_vec.SetVectorType(VectorType::FLAT_VECTOR);
		auto coeff_data = FlatVector::GetData<list_entry_t>(coeff_vec);
		coeff_data[output_idx].offset = output_idx;
		coeff_data[output_idx].length = group_result.n_features;

		// Column 3: intercept
		auto &intercept_vec = output.data[3];
		intercept_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(intercept_vec)[output_idx] = group_result.intercept;

		// Column 4: r2
		auto &r2_vec = output.data[4];
		r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(r2_vec)[output_idx] = group_result.r2;

		// Column 5: adj_r2
		auto &adj_r2_vec = output.data[5];
		adj_r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(adj_r2_vec)[output_idx] = group_result.adj_r2;

		// Column 6: mse
		auto &mse_vec = output.data[6];
		mse_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(mse_vec)[output_idx] = group_result.mse;

		// Column 7: rmse
		auto &rmse_vec = output.data[7];
		rmse_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(rmse_vec)[output_idx] = group_result.rmse;

		// Column 8: lambda
		auto &lambda_vec = output.data[8];
		lambda_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(lambda_vec)[output_idx] = group_result.lambda;

		// Column 9: n_obs
		auto &n_obs_vec = output.data[9];
		n_obs_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(n_obs_vec)[output_idx] = group_result.n_obs;

		// Column 10: n_features
		auto &n_feat_vec = output.data[10];
		n_feat_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(n_feat_vec)[output_idx] = group_result.n_features;

		bind_data.result_index++;
		output_idx++;
	}

	output.SetCardinality(output_idx);

	ANOFOX_DEBUG("Grouped ridge fit execute completed");
}

void GroupedRidgeFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_INFO("Registering anofox_grouped_ridge_fit table function");

	auto grouped_ridge_func = make_uniq<TableFunction>(
	    "anofox_grouped_ridge_fit",
	    std::vector<LogicalType> {LogicalType::TABLE, LogicalType::LIST(LogicalType::VARCHAR),
	                              LogicalType::LIST(LogicalType::VARCHAR), LogicalType::VARCHAR, LogicalType::DOUBLE,
	                              LogicalType::BOOLEAN},
	    GroupedRidgeFitExecute, GroupedRidgeFitBind);

	grouped_ridge_func->bind_replace_projection = true;

	loader.RegisterTableFunction(std::move(grouped_ridge_func));

	ANOFOX_INFO("anofox_grouped_ridge_fit registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
