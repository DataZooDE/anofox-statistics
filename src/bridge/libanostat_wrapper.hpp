#pragma once

#include "type_converters.hpp"
#include "libanostat/solvers/ols_solver.hpp"
#include "libanostat/solvers/ridge_solver.hpp"
#include "libanostat/solvers/elastic_net_solver.hpp"
#include "libanostat/solvers/wls_solver.hpp"
#include "libanostat/solvers/rls_solver.hpp"

namespace duckdb {
namespace anofox_statistics {
namespace bridge {

/**
 * LibanostatWrapper: High-level interface for calling libanostat from DuckDB
 *
 * This wrapper provides a clean interface that:
 * 1. Accepts DuckDB types (vector<double>, vector<vector<double>>, etc.)
 * 2. Converts to Eigen types using TypeConverters
 * 3. Calls appropriate libanostat solver
 * 4. Converts results back to DuckDB types
 *
 * Design: Static methods for stateless operations
 */
class LibanostatWrapper {
public:
	// ========================================================================
	// OLS (Ordinary Least Squares)
	// ========================================================================

	/**
	 * Fit OLS regression
	 *
	 * @param y_data Response vector (DuckDB format)
	 * @param x_data Design matrix (DuckDB format, row-major by default)
	 * @param options Regression options (DuckDB format)
	 * @param compute_std_errors If true, compute standard errors
	 * @param row_major If true (default), x_data is row-major; if false, column-major
	 * @return libanostat RegressionResult (can be converted back using TypeConverters)
	 */
	static libanostat::core::RegressionResult FitOLS(const vector<double> &y_data, const vector<vector<double>> &x_data,
	                                                 const RegressionOptions &options,
	                                                 bool compute_std_errors = false, bool row_major = true) {

		// Convert DuckDB types to Eigen
		auto y = TypeConverters::ToEigenVector(y_data);
		auto X = TypeConverters::ToEigenMatrix(x_data, row_major);
		auto lib_opts = TypeConverters::ToLibanostatOptions(options);

		// Call libanostat solver
		if (compute_std_errors) {
			return libanostat::solvers::OLSSolver::FitWithStdErrors(y, X, lib_opts);
		} else {
			return libanostat::solvers::OLSSolver::Fit(y, X, lib_opts);
		}
	}

	// ========================================================================
	// Ridge Regression
	// ========================================================================

	/**
	 * Fit Ridge regression with L2 regularization
	 *
	 * @param y_data Response vector
	 * @param x_data Design matrix (row-major by default)
	 * @param options Regression options (must include lambda parameter)
	 * @param compute_std_errors If true, compute approximate standard errors
	 * @param row_major If true (default), x_data is row-major; if false, column-major
	 * @return libanostat RegressionResult
	 */
	static libanostat::core::RegressionResult FitRidge(const vector<double> &y_data,
	                                                   const vector<vector<double>> &x_data,
	                                                   const RegressionOptions &options,
	                                                   bool compute_std_errors = false, bool row_major = true) {

		auto y = TypeConverters::ToEigenVector(y_data);
		auto X = TypeConverters::ToEigenMatrix(x_data, row_major);
		auto lib_opts = TypeConverters::ToLibanostatOptions(options);

		if (compute_std_errors) {
			return libanostat::solvers::RidgeSolver::FitWithStdErrors(y, X, lib_opts);
		} else {
			return libanostat::solvers::RidgeSolver::Fit(y, X, lib_opts);
		}
	}

	// ========================================================================
	// Elastic Net (L1 + L2 regularization)
	// ========================================================================

	/**
	 * Fit Elastic Net regression with L1+L2 penalties
	 *
	 * @param y_data Response vector
	 * @param x_data Design matrix (row-major by default)
	 * @param options Regression options (must include lambda and alpha)
	 * @param compute_std_errors If true, return NaN for std_errors (bootstrap recommended)
	 * @param row_major If true (default), x_data is row-major; if false, column-major
	 * @return libanostat RegressionResult
	 */
	static libanostat::core::RegressionResult FitElasticNet(const vector<double> &y_data,
	                                                        const vector<vector<double>> &x_data,
	                                                        const RegressionOptions &options,
	                                                        bool compute_std_errors = false, bool row_major = true) {

		auto y = TypeConverters::ToEigenVector(y_data);
		auto X = TypeConverters::ToEigenMatrix(x_data, row_major);
		auto lib_opts = TypeConverters::ToLibanostatOptions(options);

		if (compute_std_errors) {
			return libanostat::solvers::ElasticNetSolver::FitWithStdErrors(y, X, lib_opts);
		} else {
			return libanostat::solvers::ElasticNetSolver::Fit(y, X, lib_opts);
		}
	}

	// ========================================================================
	// WLS (Weighted Least Squares)
	// ========================================================================

	/**
	 * Fit Weighted Least Squares regression
	 *
	 * @param y_data Response vector
	 * @param x_data Design matrix (row-major by default)
	 * @param weights Observation weights (all must be > 0)
	 * @param options Regression options
	 * @param compute_std_errors If true, compute standard errors
	 * @param row_major If true (default), x_data is row-major; if false, column-major
	 * @return libanostat RegressionResult
	 */
	static libanostat::core::RegressionResult FitWLS(const vector<double> &y_data, const vector<vector<double>> &x_data,
	                                                 const vector<double> &weights, const RegressionOptions &options,
	                                                 bool compute_std_errors = false, bool row_major = true) {

		auto y = TypeConverters::ToEigenVector(y_data);
		auto X = TypeConverters::ToEigenMatrix(x_data, row_major);
		auto w = TypeConverters::ToEigenWeights(weights);
		auto lib_opts = TypeConverters::ToLibanostatOptions(options);

		if (compute_std_errors) {
			return libanostat::solvers::WLSSolver::FitWithStdErrors(y, X, w, lib_opts);
		} else {
			return libanostat::solvers::WLSSolver::Fit(y, X, w, lib_opts);
		}
	}

	// ========================================================================
	// RLS (Recursive Least Squares - Online Learning)
	// ========================================================================

	/**
	 * Fit Recursive Least Squares regression
	 *
	 * @param y_data Response vector
	 * @param x_data Design matrix (row-major by default)
	 * @param options Regression options (must include forgetting_factor)
	 * @param compute_std_errors If true, compute approximate standard errors
	 * @param row_major If true (default), x_data is row-major; if false, column-major
	 * @return libanostat RegressionResult
	 */
	static libanostat::core::RegressionResult FitRLS(const vector<double> &y_data, const vector<vector<double>> &x_data,
	                                                 const RegressionOptions &options,
	                                                 bool compute_std_errors = false, bool row_major = true) {

		auto y = TypeConverters::ToEigenVector(y_data);
		auto X = TypeConverters::ToEigenMatrix(x_data, row_major);
		auto lib_opts = TypeConverters::ToLibanostatOptions(options);

		if (compute_std_errors) {
			return libanostat::solvers::RLSSolver::FitWithStdErrors(y, X, lib_opts);
		} else {
			return libanostat::solvers::RLSSolver::Fit(y, X, lib_opts);
		}
	}

	// ========================================================================
	// Convenience Methods for Common Operations
	// ========================================================================

	/**
	 * Fit any regression model based on options
	 *
	 * This factory method chooses the appropriate solver based on options:
	 * - lambda=0, alpha=0 → OLS
	 * - lambda>0, alpha=0 → Ridge
	 * - lambda>0, alpha>0 → Elastic Net
	 *
	 * @param y_data Response vector
	 * @param x_data Design matrix (row-major by default)
	 * @param options Regression options (determines which solver to use)
	 * @param compute_std_errors If true, compute standard errors
	 * @param row_major If true (default), x_data is row-major; if false, column-major
	 * @return libanostat RegressionResult
	 */
	static libanostat::core::RegressionResult FitAuto(const vector<double> &y_data,
	                                                  const vector<vector<double>> &x_data,
	                                                  const RegressionOptions &options,
	                                                  bool compute_std_errors = false, bool row_major = true) {

		// Determine which solver to use based on parameters
		if (options.lambda <= 1e-10) {
			// OLS (no regularization)
			return FitOLS(y_data, x_data, options, compute_std_errors, row_major);
		} else if (options.alpha <= 1e-10) {
			// Ridge (L2 only)
			return FitRidge(y_data, x_data, options, compute_std_errors, row_major);
		} else {
			// Elastic Net (L1 + L2)
			return FitElasticNet(y_data, x_data, options, compute_std_errors, row_major);
		}
	}

	/**
	 * Extract all fit statistics as a struct for easy DuckDB output
	 */
	struct FitStatistics {
		double r_squared;
		double adj_r_squared;
		double mse;
		double rmse;
		idx_t rank;
		idx_t df_model;
		idx_t df_residual;
	};

	/**
	 * Compute all fit statistics from libanostat result
	 *
	 * @param result Regression result from libanostat
	 * @param n_obs Number of observations
	 * @param intercept Whether intercept was included
	 * @return FitStatistics struct
	 */
	static FitStatistics ComputeFitStatistics(const libanostat::core::RegressionResult &result, size_t n_obs,
	                                          bool intercept) {

		FitStatistics stats;
		stats.r_squared = TypeConverters::ExtractRSquared(result);
		stats.adj_r_squared = TypeConverters::ExtractAdjRSquared(result);
		stats.mse = TypeConverters::ExtractMSE(result);
		stats.rmse = TypeConverters::ExtractRMSE(result);
		stats.rank = TypeConverters::ExtractRank(result);
		stats.df_model = TypeConverters::ComputeDFModel(result, intercept);
		stats.df_residual = TypeConverters::ComputeDFResidual(result, n_obs, intercept);

		return stats;
	}
};

} // namespace bridge
} // namespace anofox_statistics
} // namespace duckdb
