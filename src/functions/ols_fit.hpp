#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Ordinary Least Squares (OLS) table function
 *
 * Performs linear regression fitting on input data.
 *
 * Signature:
 *   SELECT * FROM anofox_ols_fit(
 *       data := <table>,
 *       x_cols := [<col_name>, ...],
 *       y_col := <col_name>,
 *       add_intercept := true,
 *       method := 'auto'
 *   )
 *
 * Returns:
 *   - coefficients: DOUBLE[] - Model coefficients for X variables
 *   - intercept: DOUBLE - Model intercept (if add_intercept=true)
 *   - r_squared: DOUBLE - R² goodness of fit metric
 *   - adj_r_squared: DOUBLE - Adjusted R² accounting for degrees of freedom
 *   - mse: DOUBLE - Mean Squared Error
 *   - rmse: DOUBLE - Root Mean Squared Error
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features (excluding intercept)
 *
 * Examples:
 *   -- Simple OLS on table
 *   SELECT * FROM anofox_ols_fit(
 *       (SELECT x1, x2, y FROM my_data),
 *       ['x1', 'x2'],
 *       'y'
 *   );
 *
 *   -- Without intercept
 *   SELECT * FROM anofox_ols_fit(
 *       (SELECT x1, x2, y FROM my_data),
 *       ['x1', 'x2'],
 *       'y',
 *       false
 *   );
 */
class OlsFitFunction {
public:
	/**
	 * @brief Register the anofox_ols_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs (v1.4.1 API)
	static unique_ptr<FunctionData> OlsFitBind(ClientContext &context, TableFunctionBindInput &input,
	                                           vector<LogicalType> &return_types, vector<string> &names);

	// Execute phase - stream results back
	static void OlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
