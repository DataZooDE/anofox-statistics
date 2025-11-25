#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Elastic Net Inference - Statistical inference for Elastic Net regression
 *
 * Returns comprehensive inference for Elastic Net regression including:
 * - Coefficient estimates (sparse, some may be zero)
 * - Standard errors (debiased for non-zero coefficients)
 * - t-statistics
 * - p-values (two-tailed)
 * - Confidence intervals
 * - Significance flags
 *
 * Note: Elastic Net produces sparse solutions. Zero coefficients are included
 * with NULL statistics. Non-zero coefficients use debiased inference.
 *
 * Usage:
 *   SELECT * FROM anofox_statistics_elastic_net_inference(
 *       y := [1.0, 2.0, 3.0, 4.0]::DOUBLE[],
 *       x := [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]::DOUBLE[][],
 *       options := MAP{'alpha': 0.5, 'lambda': 0.1, 'confidence_level': 0.95, 'intercept': true}
 *   );
 */
class ElasticNetInferenceFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
