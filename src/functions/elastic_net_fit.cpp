#include "elastic_net_fit.hpp"
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

// #include <anofox/elastic_net.hpp>  // Library integration in Phase 2
#include <vector>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Bind data structure for ElasticNet fit function
 */
struct ElasticNetFitBindData : public FunctionData {
	// Input parameters
	std::vector<std::string> x_col_names;
	std::string y_col_name;
	double alpha;
	double lambda;
	bool add_intercept;

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
	idx_t n_nonzero;

	// For result streaming
	bool result_returned = false;

	ElasticNetFitBindData() : alpha(0.5), lambda(0.01), add_intercept(true), n_nonzero(0) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ElasticNetFitBindData>();
		result->x_col_names = x_col_names;
		result->y_col_name = y_col_name;
		result->alpha = alpha;
		result->lambda = lambda;
		result->add_intercept = add_intercept;
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
		result->n_nonzero = n_nonzero;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

unique_ptr<FunctionData> ElasticNetFitFunction::ElasticNetFitBind(ClientContext &context,
                                                                  TableFunctionBindInput &input) {
	ANOFOX_INFO("Starting ElasticNet fit bind phase");

	auto result = make_uniq<ElasticNetFitBindData>();

	// Parse input parameters
	if (input.inputs.size() < 4 || input.inputs.size() > 6) {
		throw InvalidInputException("anofox_statistics_elastic_net_fit requires 4-6 parameters: "
		                            "data, x_cols, y_col, alpha, lambda, [add_intercept]");
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

	// Get alpha (L1 weight)
	if (input.inputs[3]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("alpha must be a constant numeric value");
	}
	auto &alpha_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[3]);
	if (alpha_const.value.type() != LogicalType::DOUBLE) {
		throw InvalidInputException("alpha must be a DOUBLE");
	}
	result->alpha = alpha_const.value.GetValue<double>();
	if (result->alpha < 0.0 || result->alpha > 1.0) {
		throw InvalidInputException("alpha must be in range [0, 1]");
	}

	// Get lambda (regularization strength)
	if (input.inputs[4]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidInputException("lambda must be a constant numeric value");
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
		if (input.inputs[5]->type != ExpressionType::VALUE_CONSTANT) {
			throw InvalidInputException("add_intercept must be a constant boolean");
		}
		auto &intercept_const = dynamic_cast<BoundConstantExpression &>(*input.inputs[5]);
		result->add_intercept = intercept_const.value.GetValue<bool>();
	}

	ANOFOX_DEBUG("ElasticNet parameters: alpha=" + std::to_string(result->alpha) +
	             ", lambda=" + std::to_string(result->lambda) + ", intercept=" + std::to_string(result->add_intercept));

	// Validate column names and collect data would happen here
	// Similar to OLS fit implementation

	result->n_obs = 0;
	result->n_features = result->x_col_names.size();

	ANOFOX_INFO("ElasticNet fit bind phase completed");

	return std::move(result);
}

void ElasticNetFitFunction::ElasticNetFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	ANOFOX_DEBUG("ElasticNet fit execute phase");

	auto &bind_data = dynamic_cast<ElasticNetFitBindData &>(*data.bind_data);

	if (bind_data.result_returned) {
		return;
	}

	// Set output columns
	output.SetCardinality(1);

	// Column 0: coefficients (DOUBLE[])
	auto &coeff_vec = output.data[0];
	coeff_vec.SetVectorType(VectorType::FLAT_VECTOR);
	auto coeff_data = FlatVector::GetData<list_entry_t>(coeff_vec);
	coeff_data[0].offset = 0;
	coeff_data[0].length = bind_data.n_features;

	// Column 1: intercept (DOUBLE)
	auto &intercept_vec = output.data[1];
	intercept_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(intercept_vec)[0] = bind_data.intercept;

	// Column 2: r_squared (DOUBLE)
	auto &r2_vec = output.data[2];
	r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(r2_vec)[0] = bind_data.r_squared;

	// Column 3: adj_r_squared (DOUBLE)
	auto &adj_r2_vec = output.data[3];
	adj_r2_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(adj_r2_vec)[0] = bind_data.adj_r_squared;

	// Column 4: mse (DOUBLE)
	auto &mse_vec = output.data[4];
	mse_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(mse_vec)[0] = bind_data.mse;

	// Column 5: rmse (DOUBLE)
	auto &rmse_vec = output.data[5];
	rmse_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(rmse_vec)[0] = bind_data.rmse;

	// Column 6: n_obs (BIGINT)
	auto &n_obs_vec = output.data[6];
	n_obs_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<int64_t>(n_obs_vec)[0] = static_cast<int64_t>(bind_data.n_obs);

	// Column 7: n_features (BIGINT)
	auto &n_feat_vec = output.data[7];
	n_feat_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<int64_t>(n_feat_vec)[0] = static_cast<int64_t>(bind_data.n_features);

	// Column 8: alpha (DOUBLE)
	auto &alpha_vec = output.data[8];
	alpha_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(alpha_vec)[0] = bind_data.alpha;

	// Column 9: lambda (DOUBLE)
	auto &lambda_vec = output.data[9];
	lambda_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<double>(lambda_vec)[0] = bind_data.lambda;

	// Column 10: n_nonzero (BIGINT)
	auto &nonzero_vec = output.data[10];
	nonzero_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::GetData<int64_t>(nonzero_vec)[0] = static_cast<int64_t>(bind_data.n_nonzero);

	bind_data.result_returned = true;

	ANOFOX_DEBUG("ElasticNet fit execute completed");
}

void ElasticNetFitFunction::Register(ExtensionLoader &loader) {
	ANOFOX_INFO("Registering anofox_statistics_elastic_net_fit table function");

	auto elastic_net_func = make_uniq<TableFunction>(
	    "anofox_statistics_elastic_net_fit",
	    std::vector<LogicalType> {LogicalType::TABLE, LogicalType::LIST(LogicalType::VARCHAR), LogicalType::VARCHAR,
	                              LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN},
	    ElasticNetFitExecute, ElasticNetFitBind);

	elastic_net_func->bind_replace_projection = true;

	loader.RegisterTableFunction(std::move(elastic_net_func));

	ANOFOX_INFO("anofox_statistics_elastic_net_fit registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
