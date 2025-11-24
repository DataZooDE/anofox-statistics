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

		const size_t p = data.size();    // number of features
		const size_t n = data[0].size(); // number of observations

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
	 * After R-compatible fix: coefficients are stored as:
	 * - coefficients[0] = intercept (if intercept=true)
	 * - coefficients[1..p_user] = feature coefficients in ORIGINAL order (already unpermuted)
	 *
	 * @param result Regression result from libanostat
	 * @param intercept Whether intercept was included in the model
	 * @return Vector of feature coefficients (excluding intercept, in original order, NaN for aliased)
	 */
	static vector<double> ExtractFeatureCoefficients(const libanostat::core::RegressionResult &result, bool intercept) {
		if (!intercept) {
			// No intercept, return all coefficients
			return ExtractCoefficients(result);
		}

		// After R-compatible fix: coefficients[1..p_user] are already in original feature order
		// No need to use permutation_indices, just extract positions 1..p_user
		size_t n_params = result.coefficients.size();
		size_t n_features = n_params - 1; // Excluding intercept at position 0

		vector<double> feature_coeffs;
		feature_coeffs.reserve(n_features);
		for (size_t j = 1; j <= n_features; j++) {
			feature_coeffs.push_back(result.coefficients(static_cast<Eigen::Index>(j)));
		}

		return feature_coeffs;
	}

	/**
	 * Extract intercept from libanostat result
	 * After the R-compatible fix, when intercept=true, the intercept is ALWAYS at position 0.
	 *
	 * @param result Regression result from libanostat
	 * @param intercept Whether intercept was included in the model
	 * @return Intercept value, or 0.0 if intercept=false
	 */
	static double ExtractIntercept(const libanostat::core::RegressionResult &result, bool intercept) {
		if (!intercept) {
			return 0.0;
		}

		// After R-compatible fix: intercept is ALWAYS at position 0, never aliased
		return result.coefficients(0);
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
	static libanostat::core::RegressionOptions ToLibanostatOptions(const RegressionOptions &duckdb_opts) {

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
	 *
	 * After R-compatible fix in ba2334b: result.rank now includes intercept if present.
	 * df_model = result.rank
	 */
	static idx_t ComputeDFModel(const libanostat::core::RegressionResult &result, bool intercept) {
		return ToIdxT(result.rank);
	}

	/**
	 * Compute degrees of freedom (residual) from result
	 *
	 * After R-compatible fix in ba2334b: result.rank now includes BOTH features AND intercept.
	 * df_residual = n_obs - result.rank
	 */
	static idx_t ComputeDFResidual(const libanostat::core::RegressionResult &result, size_t n_obs, bool intercept) {
		// df_model = rank (already includes intercept if present)
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
