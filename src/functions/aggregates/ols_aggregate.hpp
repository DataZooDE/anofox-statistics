#pragma once

#include "duckdb.hpp"
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS Aggregate Function - Compute OLS regression per group
 *
 * Aggregate function that computes OLS regression coefficients for grouped data.
 * Works with GROUP BY to perform regression analysis on each group separately.
 *
 * Signature:
 *   SELECT group_col, ols_coeff_agg(y, x) as coeff FROM data GROUP BY group_col;
 *
 * Examples:
 *   -- Single variable regression per group
 *   SELECT
 *       category,
 *       ols_coeff_agg(sales, price) as price_coeff
 *   FROM sales_data
 *   GROUP BY category;
 *
 *   -- Calculate beta for each stock
 *   SELECT
 *       stock_id,
 *       ols_coeff_agg(stock_return, market_return)[1] as beta
 *   FROM returns
 *   GROUP BY stock_id;
 */
class OlsAggregateFunction {
public:
	/**
	 * @brief Register OLS aggregate functions
	 *
	 * Registers:
	 * - ols_coeff_agg(y, x) -> DOUBLE - Single coefficient
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
