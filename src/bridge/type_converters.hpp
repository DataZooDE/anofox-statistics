#pragma once

#include "duckdb.hpp"
#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include <Eigen/Dense>
#include <vector>
#include <limits>

namespace duckdb {
namespace anofox_statistics {
namespace bridge {

/**
 * Type Converters: Bridge between DuckDB and libanostat (Eigen) types
 *
 * This bridge layer provides bidirectional conversion between:
 * - DuckDB types (idx_t, vector<double>, etc.) ↔ Eigen types (VectorXd, MatrixXd)
 * - DuckDB types ↔ libanostat types (size_t, std::vector)
 *
 * Design principle: Keep conversions simple and explicit.
 * Zero-copy where possible, but correctness takes priority.
 */
class TypeConverters {
public:
	// ========================================================================
	// DuckDB → Eigen Conversions
	// ========================================================================

	/**
	 * Convert DuckDB vector<double> to Eigen::VectorXd
	 *
	 * @param data DuckDB vector of doubles
	 * @return Eigen::VectorXd copy of the data
	 */
	static Eigen::VectorXd ToEigenVector(const vector<double> &data) {
		Eigen::VectorXd result(static_cast<Eigen::Index>(data.size()));
		for (size_t i = 0; i < data.size(); i++) {
			result(static_cast<Eigen::Index>(i)) = data[i];
		}
		return result;
	}

	/**
	 * Convert DuckDB 2D vector (column-major) to Eigen::MatrixXd
	 *
	 * @param data Column-major 2D array: data[j][i] = X(i,j)
	 *             where j is feature index, i is observation index
	 * @return Eigen::MatrixXd with n rows (observations) and p columns (features)
	 */
	static Eigen::MatrixXd ToEigenMatrix(const vector<vector<double>> &data) {
		if (data.empty()) {
			return Eigen::MatrixXd(0, 0);
		}

		const size_t p = data.size();      // number of features
		const size_t n = data[0].size();   // number of observations

		// Validate all columns have same length
		for (size_t j = 0; j < p; j++) {
			if (data[j].size() != n) {
				throw InvalidInputException("All feature columns must have same length");
			}
		}

		Eigen::MatrixXd result(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p));
		for (size_t j = 0; j < p; j++) {
			for (size_t i = 0; i < n; i++) {
				result(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = data[j][i];
			}
		}
		return result;
	}

	/**
	 * Convert DuckDB vector<double> to Eigen::VectorXd (weights)
	 *
	 * Same as ToEigenVector but with semantic clarity for weights
	 */
	static Eigen::VectorXd ToEigenWeights(const vector<double> &weights) {
		return ToEigenVector(weights);
	}

	// ========================================================================
	// Eigen → DuckDB Conversions
	// ========================================================================

	/**
	 * Convert Eigen::VectorXd to DuckDB vector<double>
	 *
	 * @param vec Eigen vector
	 * @return DuckDB vector<double> copy
	 */
	static vector<double> FromEigenVector(const Eigen::VectorXd &vec) {
		vector<double> result;
		result.reserve(static_cast<size_t>(vec.size()));
		for (Eigen::Index i = 0; i < vec.size(); i++) {
			result.push_back(vec(i));
		}
		return result;
	}

	/**
	 * Convert Eigen::VectorXd to DuckDB vector<bool> (for is_aliased)
	 *
	 * @param aliased Boolean vector from libanostat
	 * @return DuckDB vector<bool>
	 */
	static vector<bool> FromStdVectorBool(const std::vector<bool> &aliased) {
		vector<bool> result;
		result.reserve(aliased.size());
		for (bool val : aliased) {
			result.push_back(val);
		}
		return result;
	}

	// ========================================================================
	// libanostat::RegressionResult → DuckDB Types
	// ========================================================================

	/**
	 * Extract coefficients from libanostat result
	 * Converts NaN (aliased coefficients) to appropriate representation
	 *
	 * @param result Regression result from libanostat
	 * @return Vector of coefficient values (NaN for aliased)
	 */
	static vector<double> ExtractCoefficients(const libanostat::core::RegressionResult &result) {
		return FromEigenVector(result.coefficients);
	}

	/**
	 * Extract feature coefficients (excluding intercept) from libanostat result
	 * When intercept=true, libanostat includes intercept in coefficients array.
	 * This function extracts only the feature coefficients in original feature order.
	 *
	 * @param result Regression result from libanostat
	 * @param intercept Whether intercept was included in the model
	 * @return Vector of feature coefficients (excluding intercept, in original order, NaN for aliased)
	 */
	static vector<double> ExtractFeatureCoefficients(const libanostat::core::RegressionResult &result,
	                                                  bool intercept) {
		if (!intercept) {
			// No intercept, return all coefficients
			return ExtractCoefficients(result);
		}

		// When intercept=true, original column 0 is intercept, columns 1..p_user are features
		// We need to extract coefficients for original columns 1..p_user in order
		// permutation_indices maps: pivoted_position -> original_column_index
		// We need inverse: original_column_index -> coefficient_value
		
		size_t n_params = result.coefficients.size();
		size_t n_features = n_params - 1; // Excluding intercept
		
		// Build map: original_column_index -> coefficient_value
		vector<double> coeff_map(n_params, std::numeric_limits<double>::quiet_NaN());
		for (size_t i = 0; i < n_params; i++) {
			size_t orig_col = result.permutation_indices[i];
			if (orig_col < n_params) {
				coeff_map[orig_col] = result.coefficients(static_cast<Eigen::Index>(i));
			}
		}
		
		// Extract feature coefficients in original order (columns 1..n_features)
		vector<double> feature_coeffs;
		feature_coeffs.reserve(n_features);
		for (size_t j = 1; j <= n_features; j++) {
			feature_coeffs.push_back(coeff_map[j]);
		}
		
		return feature_coeffs;
	}

	/**
	 * Extract intercept from libanostat result
	 * When intercept=true, libanostat includes intercept in coefficients array at position 0 (after pivoting).
	 *
	 * @param result Regression result from libanostat
	 * @param intercept Whether intercept was included in the model
	 * @return Intercept value, or 0.0 if intercept=false
	 */
	static double ExtractIntercept(const libanostat::core::RegressionResult &result, bool intercept) {
		if (!intercept) {
			return 0.0;
		}

		// Find intercept position using permutation_indices
		// Original column 0 is the intercept, find where it ended up after pivoting
		for (size_t i = 0; i < result.permutation_indices.size(); i++) {
			if (result.permutation_indices[i] == 0) {
				// Found intercept position
				return result.coefficients(static_cast<Eigen::Index>(i));
			}
		}

		// Intercept not found (should not happen if intercept=true)
		return std::numeric_limits<double>::quiet_NaN();
	}

	/**
	 * Extract standard errors from libanostat result
	 *
	 * @param result Regression result from libanostat
	 * @return Vector of standard errors (NaN if not available)
	 */
	static vector<double> ExtractStdErrors(const libanostat::core::RegressionResult &result) {
		if (!result.has_std_errors) {
			// Return vector of NaN if std errors not computed
			vector<double> nan_vec(static_cast<size_t>(result.coefficients.size()),
			                       std::numeric_limits<double>::quiet_NaN());
			return nan_vec;
		}
		return FromEigenVector(result.std_errors);
	}

	/**
	 * Extract residuals from libanostat result
	 */
	static vector<double> ExtractResiduals(const libanostat::core::RegressionResult &result) {
		return FromEigenVector(result.residuals);
	}

	/**
	 * Extract aliased flag vector from libanostat result
	 */
	static vector<bool> ExtractIsAliased(const libanostat::core::RegressionResult &result) {
		return FromStdVectorBool(result.is_aliased);
	}

	// ========================================================================
	// DuckDB → libanostat::RegressionOptions Conversion
	// ========================================================================

	/**
	 * Convert DuckDB RegressionOptions to libanostat RegressionOptions
	 *
	 * This is a direct mapping since the types are designed to be compatible.
	 *
	 * @param duckdb_opts DuckDB regression options
	 * @return libanostat regression options
	 */
	static libanostat::core::RegressionOptions ToLibanostatOptions(
	    const RegressionOptions &duckdb_opts) {

		libanostat::core::RegressionOptions result;

		result.intercept = duckdb_opts.intercept;
		result.lambda = duckdb_opts.lambda;
		result.alpha = duckdb_opts.alpha;
		result.forgetting_factor = duckdb_opts.forgetting_factor;
		result.confidence_level = duckdb_opts.confidence_level;
		result.tolerance = duckdb_opts.tolerance;
		result.max_iterations = static_cast<size_t>(duckdb_opts.max_iterations);
		result.full_output = duckdb_opts.full_output;

		return result;
	}

	// ========================================================================
	// Index Type Conversions
	// ========================================================================

	/**
	 * Convert DuckDB idx_t to libanostat size_t
	 *
	 * Direct cast, included for clarity and future-proofing
	 */
	static size_t ToSizeT(idx_t val) {
		return static_cast<size_t>(val);
	}

	/**
	 * Convert libanostat size_t to DuckDB idx_t
	 */
	static idx_t ToIdxT(size_t val) {
		return static_cast<idx_t>(val);
	}

	// ========================================================================
	// Fit Statistics Extraction Helpers
	// ========================================================================

	/**
	 * Extract R² from result
	 */
	static double ExtractRSquared(const libanostat::core::RegressionResult &result) {
		return result.r_squared;
	}

	/**
	 * Extract adjusted R² from result
	 */
	static double ExtractAdjRSquared(const libanostat::core::RegressionResult &result) {
		return result.adj_r_squared;
	}

	/**
	 * Extract MSE from result
	 */
	static double ExtractMSE(const libanostat::core::RegressionResult &result) {
		return result.mse;
	}

	/**
	 * Extract RMSE from result
	 */
	static double ExtractRMSE(const libanostat::core::RegressionResult &result) {
		return result.rmse;
	}

	/**
	 * Extract rank from result
	 */
	static idx_t ExtractRank(const libanostat::core::RegressionResult &result) {
		return ToIdxT(result.rank);
	}

	/**
	 * Compute degrees of freedom (model) from result
	 */
	static idx_t ComputeDFModel(const libanostat::core::RegressionResult &result, bool intercept) {
		return ToIdxT(result.rank + (intercept ? 1 : 0));
	}

	/**
	 * Compute degrees of freedom (residual) from result
	 * 
	 * Note: When intercept=true, result.rank already includes the intercept column
	 * (since X_work has intercept as first column), so we don't add 1 for intercept.
	 */
	static idx_t ComputeDFResidual(const libanostat::core::RegressionResult &result,
	                                size_t n_obs, bool intercept) {
		// result.rank already includes intercept if intercept=true (it's the rank of X_work)
		// So df_model = result.rank (no need to add 1 for intercept)
		size_t df_model = result.rank;
		if (n_obs > df_model) {
			return ToIdxT(n_obs - df_model);
		}
		return 0;
	}
};

} // namespace bridge
} // namespace anofox_statistics
} // namespace duckdb
