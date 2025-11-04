#include "ols_fit.hpp"
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
 * @brief Bind data structure for OLS fit function
 *
 * Stores data collected during bind phase that will be used in execute phase
 */
struct OlsFitBindData : public FunctionData {
	// Input parameters
	std::vector<std::string> x_col_names;
	std::string y_col_name;
	bool add_intercept;
	std::string method;

	// Data collected from input
	Eigen::MatrixXd x_data;
	Eigen::VectorXd y_data;
	idx_t n_obs;
	idx_t n_features;

	// Results from fitting
	Eigen::VectorXd coefficients;
	double intercept;
	double r_squared;
	double adj_r_squared;
	double mse;
	double rmse;

	// For result streaming
	bool result_returned = false;

	OlsFitBindData() = default;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<OlsFitBindData>();
		result->x_col_names = x_col_names;
		result->y_col_name = y_col_name;
		result->add_intercept = add_intercept;
		result->method = method;
		result->x_data = x_data;
		result->y_data = y_data;
		result->n_obs = n_obs;
		result->n_features = n_features;
		result->coefficients = coefficients;
		result->intercept = intercept;
		result->r_squared = r_squared;
		result->adj_r_squared = adj_r_squared;
		result->mse = mse;
		result->rmse = rmse;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false; // Table functions don't cache results
	}
};

unique_ptr<FunctionData> OlsFitFunction::OlsFitBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	ANOFOX_INFO("Starting OLS fit bind phase");

	auto result = make_uniq<OlsFitBindData>();

	// Parse input parameters
	if (input.inputs.size() < 3 || input.inputs.size() > 5) {
		throw InvalidInputException("anofox_statistics_ols_fit requires 3-5 parameters: "
		                            "data, x_cols, y_col, [add_intercept], [method]");
	}

	// Get the input table expression
	auto &table_expr = input.inputs[0];

	// Get X column names (v1.4.1 API)
	if (input.inputs[1].type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("x_cols must be an array of strings");
	}

	auto x_cols_list = ListValue::GetChildren(input.inputs[1]);
	for (const auto &val : x_cols_list) {
		if (val.type().id() != LogicalTypeId::VARCHAR) {
			throw InvalidInputException("x_cols must be an array of column names (strings)");
		}
		result->x_col_names.push_back(val.GetValue<string>());
	}

	if (result->x_col_names.empty()) {
		throw InvalidInputException("x_cols must contain at least one column name");
	}

	ANOFOX_DEBUG("X columns: " << result->x_col_names.size());

	// Get Y column name (v1.4.1 API)
	if (input.inputs[2].type().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("y_col must be a column name (string)");
	}
	result->y_col_name = input.inputs[2].GetValue<string>();

	// Get optional add_intercept parameter (default true)
	result->add_intercept = true;
	if (input.inputs.size() > 3) {
		if (input.inputs[3].type() != LogicalType::BOOLEAN) {
			throw InvalidInputException("add_intercept must be a boolean");
		}
		result->add_intercept = input.inputs[3].GetValue<bool>();
	}

	// Get optional method parameter (default 'auto') - v1.4.1 API
	result->method = "auto";
	if (input.inputs.size() > 4) {
		if (input.inputs[4].type().id() != LogicalTypeId::VARCHAR) {
			throw InvalidInputException("method must be a string");
		}
		result->method = input.inputs[4].GetValue<string>();
	}

	ANOFOX_DEBUG("Y column: " << result->y_col_name);
	ANOFOX_DEBUG("Add intercept: " << (result->add_intercept ? "true" : "false"));
	ANOFOX_DEBUG("Method: " << result->method);

	// TODO(v1.4.1): The old table_expr->Execute() API no longer exists in v1.4.1
	// Need to reimplement table data collection using v1.4.1 patterns
	// Possible approaches:
	// 1. Use TableFunctionBindInput::input_table_types/names with a different data access pattern
	// 2. Change from TABLE input to a different table function pattern
	// 3. Implement as aggregate function with custom state
	throw InvalidInputException("anofox_statistics_ols_fit is not yet updated for DuckDB v1.4.1 API. "
	                            "The table input processing needs architectural refactoring.");

	/*
	// OLD v1.3 CODE - DOES NOT WORK IN v1.4.1:
	// Execute the table function to collect all data
	auto table_func_result = table_expr->Execute(context);

	// Collect all data into matrices
	idx_t total_rows = 0;
	std::vector<Eigen::MatrixXd> x_chunks;
	std::vector<Eigen::VectorXd> y_chunks;

	ANOFOX_DEBUG("Collecting data from input table...");

	while (table_func_result) {
	    auto& chunk = table_func_result;

	    if (chunk->size() > 0) {
	        // Find column indices
	        ValidationUtils::ValidateRequiredColumns(chunk, result->x_col_names);
	        auto x_indices = ValidationUtils::FindColumnsByNames(chunk, result->x_col_names);
	        auto y_index = ValidationUtils::FindColumnByName(chunk, result->y_col_name);

	        // Validate numeric types
	        for (auto idx : x_indices) {
	            ValidationUtils::ValidateColumn(chunk->data[idx], result->x_col_names[idx]);
	        }
	        ValidationUtils::ValidateColumn(chunk->data[y_index], result->y_col_name);

	        // Extract data
	        auto x_chunk = TypeConverter::ExtractDoubleMatrix(chunk, x_indices);
	        auto y_chunk = TypeConverter::ExtractDoubleColumn(chunk, y_index);

	        x_chunks.push_back(x_chunk);
	        y_chunks.push_back(y_chunk);

	        total_rows += chunk->size();
	    }

	    table_func_result = table_func_result->Next();
	}

	if (total_rows == 0) {
	    throw InvalidInputException("Input table is empty");
	}

	ANOFOX_DEBUG("Collected " << total_rows << " rows from input table");

	// Combine chunks into single matrix
	result->n_obs = total_rows;
	result->n_features = result->x_col_names.size();

	auto x_buffer = MemoryManager::CreateMatrixBuffer(total_rows, result->n_features);
	auto y_buffer = MemoryManager::CreateVectorBuffer(total_rows);

	idx_t row_offset = 0;
	for (size_t i = 0; i < x_chunks.size(); i++) {
	    auto& x_chunk = x_chunks[i];
	    auto& y_chunk = y_chunks[i];
	    idx_t chunk_rows = x_chunk.rows();

	    x_buffer.Get().block(row_offset, 0, chunk_rows, result->n_features) = x_chunk;
	    y_buffer.Get().segment(row_offset, chunk_rows) = y_chunk;

	    row_offset += chunk_rows;
	}

	result->x_data = x_buffer.Get();
	result->y_data = y_buffer.Get();

	ANOFOX_DEBUG("Performing OLS regression with " << result->n_features << " features");

	// Perform OLS fitting using AnofoxStatistics library
	::anofox_statistics::OLS ols;
	ols.SetAddIntercept(result->add_intercept);

	// Fit the model
	ANOFOX_TIMING_START();
	auto fit_result = ols.Fit(result->x_data, result->y_data);
	ANOFOX_TIMING_END("OLS fitting");

	if (!fit_result) {
	    throw InvalidInputException("OLS fitting failed: " + fit_result.GetError());
	}

	// Extract results
	auto model = fit_result.GetValue();
	result->coefficients = model.GetCoefficients();
	result->intercept = model.GetIntercept();
	result->r_squared = model.GetRSquared();
	result->adj_r_squared = model.GetAdjustedRSquared();
	result->mse = model.GetMSE();
	result->rmse = std::sqrt(result->mse);

	ANOFOX_INFO("OLS regression completed successfully");
	ANOFOX_DEBUG("R² = " << result->r_squared << ", RMSE = " << result->rmse);
	*/

	// Set return types and names (v1.4.1 API)
	names = {"coefficients", "intercept", "r_squared", "adj_r_squared", "mse", "rmse", "n_obs", "n_features"};

	return_types = {LogicalType::LIST(LogicalType::DOUBLE),
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT};

	return std::move(result);
}

void OlsFitFunction::OlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	ANOFOX_TRACE("OLS fit execute phase");

	// v1.4.1 API: bind_data is const, use const_cast
	auto &bind_data = const_cast<OlsFitBindData &>(dynamic_cast<const OlsFitBindData &>(*data.bind_data));

	// Return results only once
	if (bind_data.result_returned) {
		return; // No more data
	}

	bind_data.result_returned = true;

	// Build output chunk with single row
	output.SetCardinality(1);

	// Column 0: coefficients (DOUBLE[]) - v1.4.1 API
	vector<Value> coeffs_values;
	for (idx_t i = 0; i < bind_data.coefficients.size(); i++) {
		coeffs_values.push_back(Value(bind_data.coefficients[i]));
	}
	auto coeffs_list = Value::LIST(LogicalType::DOUBLE, coeffs_values);
	output.data[0].SetValue(0, coeffs_list);

	// Columns 1-7: scalar values - v1.4.1 API
	output.data[1].SetValue(0, Value(bind_data.intercept));
	output.data[2].SetValue(0, Value(bind_data.r_squared));
	output.data[3].SetValue(0, Value(bind_data.adj_r_squared));
	output.data[4].SetValue(0, Value(bind_data.mse));
	output.data[5].SetValue(0, Value(bind_data.rmse));
	output.data[6].SetValue(0, Value::BIGINT(bind_data.n_obs));
	output.data[7].SetValue(0, Value::BIGINT(bind_data.n_features));
}

void OlsFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering anofox_statistics_ols_fit table function");

	// Input parameters: TABLE, VARCHAR[], VARCHAR, [BOOLEAN], [VARCHAR]
	vector<LogicalType> arguments = {LogicalType::TABLE, LogicalType::LIST(LogicalType::VARCHAR), LogicalType::VARCHAR,
	                                 LogicalType::BOOLEAN, LogicalType::VARCHAR};

	// Create table function (v1.4.1 API)
	TableFunction function("anofox_statistics_ols_fit", arguments, OlsFitExecute, OlsFitBind);

	// Register
	loader.RegisterFunction(function);

	ANOFOX_DEBUG("anofox_statistics_ols_fit table function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
