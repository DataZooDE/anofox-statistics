#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Normality Test - Test if residuals follow normal distribution
 *
 * Implements Jarque-Bera test based on skewness and kurtosis:
 * JB = n/6 * (S² + (K-3)²/4)
 *
 * where:
 * - S = skewness
 * - K = kurtosis
 * - Under H0 (normality), JB ~ χ²(2)
 *
 * Usage:
 *   SELECT * FROM normality_test(
 *       residuals := [0.1, -0.2, 0.15, -0.05, 0.0]::DOUBLE[],
 *       alpha := 0.05
 *   );
 *
 * Returns single row with:
 * - n_obs: Number of observations
 * - skewness: Sample skewness
 * - kurtosis: Sample kurtosis
 * - jb_statistic: Jarque-Bera test statistic
 * - p_value: P-value (approximate)
 * - is_normal: TRUE if we cannot reject normality
 * - conclusion: 'normal' or 'non-normal'
 */
class NormalityTestFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
