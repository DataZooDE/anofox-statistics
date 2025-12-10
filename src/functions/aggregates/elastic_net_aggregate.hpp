#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * Elastic Net Aggregate Function
 *
 * Computes Elastic Net regression (L1 + L2 regularization) on grouped data.
 * Supports window functions with OVER clause for rolling/expanding windows.
 *
 * Usage:
 *   SELECT category,
 *          anofox_statistics_elastic_net_agg(y, [x1, x2], MAP{'intercept': true, 'alpha': 0.5, 'lambda': 0.01})
 *   FROM data
 *   GROUP BY category;
 *
 * Window function usage:
 *   SELECT anofox_statistics_elastic_net_agg(y, [x1, x2], MAP{'alpha': 0.5, 'lambda': 0.01})
 *          OVER (ORDER BY time ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
 *   FROM time_series;
 */
struct ElasticNetAggregateFunction {
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
