#include "expanding_ols_fit.hpp"
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

// #include <anofox/ols.hpp>  // Library integration in Phase 2
#include <vector>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Bind data structure for expanding OLS fit function
 */
struct ExpandingOlsFitBindData : public FunctionData {
	// Input parameters
	std::vector<std::string> x_col_names;
	std::string y_col_name;
	idx_t min_obs;
	bool add_intercept;
	std::string partition_col; // Optional
	std::string order_col;     // Optional

	// Data collected from input
	Eigen::MatrixXd all_x_data;
	Eigen::VectorXd all_y_data;

	// Results for each expanding window
	struct WindowResult {
		idx_t window_id;
		Eigen::VectorXd coefficients;
		double intercept;
		double r_squared;
		double adj_r_squared;
		double mse;
		double rmse;
		idx_t n_obs;
		idx_t n_features;
		idx_t window_start_idx;
		idx_t window_end_idx;
	};

	std::vector<WindowResult> results;
	idx_t result_index;

	ExpandingOlsFitBindData() : min_obs(30), add_intercept(true), result_index(0) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ExpandingOlsFitBindData>();
		result->x_col_names = x_col_names;
		result->y_col_name = y_col_name;
		result->min_obs = min_obs;
		result->add_intercept = add_intercept;
		result->partition_col = partition_col;
		result->order_col = order_col;
		result->all_x_data = all_x_data;
		result->all_y_data = all_y_data;
		result->results = results;
		result->result_index = result_index;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

unique_ptr<FunctionData> ExpandingOlsFitFunction::ExpandingOlsFitBind(ClientContext &context,
                                                                      TableFunctionBindInput &input) {
	ANOFOX_INFO("Starting expanding OLS fit bind phase");

	auto result = make_uniq<ExpandingOlsFitBindData>();

	// Parse input parameters
	if (input.inputs.size() < 4 || input.inputs.size() > 7) {
		throw InvalidInputException("anofox_statistics_expanding_ols_fit requires 4-7 parameters: "
		                            "data, x_cols, y_col, min_obs, [partition_col], [order_col], [add_intercept]");
	}

	// Get the input table expression
	auto &table_expr = input.inputs[0];

	// Get X column names
	if (input.inputs[1]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("x_cols must be a constant array");
	}
	auto &x_cols_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[1]);
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

	// Get Y column name
	if (input.inputs[2]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("y_col must be a constant string");
	}
	auto &y_col_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[2]);
	if (y_col_const.value.type() != LogicalType::VARCHAR) {
		throw InvalidInputException("y_col must be a string");
	}
	result->y_col_name = y_col_const.value.ToString();

	// Get minimum observations
	if (input.inputs[3]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("min_obs must be a constant");
	}
	auto &min_obs_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[3]);
	if (min_obs_const.value.type() != LogicalType::BIGINT) {
		throw InvalidInputException("min_obs must be an INTEGER");
	}
	result->min_obs = min_obs_const.value.GetValue<int64_t>();
	if (result->min_obs < 2) {
		throw InvalidInputException("min_obs must be >= 2");
	}

	// Get optional partition column
	if (input.inputs.size() > 4) {
		if (input.inputs[4]->type == ExpressionType::VALUE_CONSTANT) {
			auto &part_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[4]);
			if (part_const.value.type() == LogicalType::VARCHAR) {
				result->partition_col = part_const.value.ToString();
			}
		}
	}

	// Get optional order column
	if (input.inputs.size() > 5) {
		if (input.inputs[5]->type == ExpressionType::VALUE_CONSTANT) {
			auto &order_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[5]);
			if (order_const.value.type() == LogicalType::VARCHAR) {
				result->order_col = order_const.value.ToString();
			}
		}
	}

	// Get add_intercept flag
	if (input.inputs.size() > 6) {
		if (input.inputs[6]->type == ExpressionType::VALUE_CONSTANT) {
			auto &intercept_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[6]);
			result->add_intercept = intercept_const.value.GetValue<bool>();
		}
	}

	ANOFOX_DEBUG("Expanding OLS parameters: min_obs=" + std::to_string(result->min_obs) + ", x_cols=" +
	             std::to_string(result->x_col_names.size()) + ", intercept=" + std::to_string(result->add_intercept));

	ANOFOX_INFO("Expanding OLS fit bind phase completed");

	return std::move(result);
}

void ExpandingOlsFitFunction::ExpandingOlsFitExecute(ClientContext &context, TableFunctionInput &data,
                                                     DataChunk &output) {
	ANOFOX_DEBUG("Expanding OLS fit execute phase");

	auto &bind_data = dynamic_cast<ExpandingOlsFitBindData &>(*data.bind_data);

	// If we've returned all results, we're done
	if (bind_data.result_index >= bind_data.results.size()) {
		return;
	}

	// Stream results one window at a time
	idx_t output_idx = 0;
	while (output_idx < 2048 && bind_data.result_index < bind_data.results.size()) {
		auto &window_result = bind_data.results[bind_data.result_index];

		// Column 0: window_id
		auto &window_id_vec = output.data[0];
		window_id_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(window_id_vec)[output_idx] = window_result.window_id;

		// Column 1: coefficients (DOUBLE[])
		auto &coeff_vec = output.data[1];
		coeff_vec.SetVectorType(VectorType::FLAT_VECTOR);
		auto coeff_data = FlatVector::GetData<list_entry_t>(coeff_vec);
		coeff_data[output_idx].offset = output_idx;
		coeff_data[output_idx].length = window_result.n_features;

		// Column 2: intercept
		auto &intercept_vec = output.data[2];
		intercept_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(intercept_vec)[output_idx] = window_result.intercept;

		// Column 3: r_squared
		auto &r2_vec = output.data[3];
		r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(r2_vec)[output_idx] = window_result.r_squared;

		// Column 4: adj_r_squared
		auto &adj_r2_vec = output.data[4];
		adj_r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(adj_r2_vec)[output_idx] = window_result.adj_r_squared;

		// Column 5: mse
		auto &mse_vec = output.data[5];
		mse_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(mse_vec)[output_idx] = window_result.mse;

		// Column 6: rmse
		auto &rmse_vec = output.data[6];
		rmse_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<double>(rmse_vec)[output_idx] = window_result.rmse;

		// Column 7: n_obs
		auto &n_obs_vec = output.data[7];
		n_obs_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(n_obs_vec)[output_idx] = window_result.n_obs;

		// Column 8: n_features
		auto &n_feat_vec = output.data[8];
		n_feat_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(n_feat_vec)[output_idx] = window_result.n_features;

		// Column 9: window_start_idx
		auto &start_idx_vec = output.data[9];
		start_idx_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(start_idx_vec)[output_idx] = window_result.window_start_idx;

		// Column 10: window_end_idx
		auto &end_idx_vec = output.data[10];
		end_idx_vec.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::GetData<int64_t>(end_idx_vec)[output_idx] = window_result.window_end_idx;

		bind_data.result_index++;
		output_idx++;
	}

	output.SetCardinality(output_idx);

	ANOFOX_DEBUG("Expanding OLS fit execute completed");
}

void ExpandingOlsFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_INFO("Registering anofox_statistics_expanding_ols_fit table function");

	auto expanding_ols_func = make_uniq<TableFunction>(
	    "anofox_statistics_expanding_ols_fit",
	    std::vector<LogicalType> {LogicalType::TABLE, LogicalType::LIST(LogicalType::VARCHAR), LogicalType::VARCHAR,
	                              LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                              LogicalType::BOOLEAN},
	    ExpandingOlsFitExecute, ExpandingOlsFitBind);

	expanding_ols_func->bind_replace_projection = true;

	loader.RegisterTableFunction(std::move(expanding_ols_func));

	ANOFOX_INFO("anofox_statistics_expanding_ols_fit registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
