#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief VIF - Variance Inflation Factor for multicollinearity detection
 *
 * Computes VIF for each predictor variable:
 * VIF_j = 1 / (1 - R²_j)
 *
 * where R²_j is obtained by regressing x_j on all other x variables.
 *
 * Interpretation:
 * - VIF = 1: No correlation with other variables
 * - VIF < 5: Low multicollinearity (acceptable)
 * - VIF 5-10: Moderate multicollinearity (warning)
 * - VIF > 10: High multicollinearity (problematic)
 *
 * Usage:
 *   SELECT * FROM vif(
 *       x := [[1.0, 2.0, 3.0],
 *             [2.0, 4.1, 6.2],
 *             [3.0, 5.9, 9.1]]::DOUBLE[][]
 *   );
 *
 * Returns one row per variable with:
 * - variable_id: Variable index (1-indexed)
 * - variable_name: Variable name (x1, x2, ...)
 * - vif: VIF value
 * - severity: 'low', 'moderate', or 'high'
 */
class VifFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
