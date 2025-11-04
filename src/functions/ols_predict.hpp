#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS scalar prediction function
 *
 * Computes predictions for new data using OLS model coefficients.
 *
 * Signature:
 *   SELECT anofox_ols_predict(
 *       coefficients := DOUBLE[],
 *       intercept := DOUBLE,
 *       x1 := DOUBLE,
 *       x2 := DOUBLE,
 *       ...
 *   ) -> DOUBLE
 *
 * Or with array input:
 *   SELECT anofox_ols_predict(
 *       coefficients := [0.5, -0.3],
 *       intercept := 0.1,
 *       x_values := [1.0, 2.0]
 *   ) -> DOUBLE
 *
 * Examples:
 *   -- Predict with individual columns
 *   SELECT x1, x2, anofox_ols_predict([0.5, -0.3], 0.1, x1, x2) as predicted_y
 *   FROM new_data;
 *
 *   -- Predict with arrays
 *   SELECT anofox_ols_predict(
 *       (SELECT coefficients FROM my_model),
 *       (SELECT intercept FROM my_model),
 *       [1.0, 2.0]
 *   ) as prediction;
 */
class OlsPredictFunction {
public:
	/**
	 * @brief Register the anofox_ols_predict scalar function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	/**
	 * @brief Execute prediction with individual parameters
	 * Variadic form: anofox_ols_predict(coeffs, intercept, x1, x2, ...)
	 */
	static void OlsPredictVariadic(DataChunk &args, ExpressionState &state, Vector &result);

	/**
	 * @brief Execute prediction with array of coefficients
	 * Array form: anofox_ols_predict(coeffs, intercept, x_array)
	 */
	static void OlsPredictArray(DataChunk &args, ExpressionState &state, Vector &result);
};

} // namespace anofox_statistics
} // namespace duckdb
