#pragma once

#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include <Eigen/Dense>
#include <string>

namespace libanostat {
namespace solvers {

/**
 * IRegressor: Abstract interface for all regression solvers
 *
 * This interface defines the contract that all regression solvers must implement.
 * It enables polymorphic use of different regression algorithms and provides a
 * unified API for:
 * - Testing: Mock implementations for unit tests
 * - Factory patterns: Runtime solver selection
 * - Plugin systems: External solver implementations
 * - Documentation: Clear contract of what a solver must provide
 *
 * Design Philosophy:
 * - Header-only interface (no virtual table overhead in actual implementations)
 * - All solvers implement Fit() and FitWithStdErrors() methods
 * - Consistent return type (RegressionResult)
 * - Options pattern for extensibility (RegressionOptions)
 *
 * Note on Implementation:
 * While this interface uses virtual methods for polymorphic use, the actual
 * solver classes (OLSSolver, RidgeSolver, etc.) use static methods for
 * performance. This interface is primarily for:
 * 1. Documentation of the solver contract
 * 2. Testing and mocking
 * 3. Runtime solver selection (when needed)
 *
 * For performance-critical code, use the concrete solver classes directly.
 */
class IRegressor {
public:
	virtual ~IRegressor() = default;

	/**
	 * Get the name of this regression solver
	 *
	 * @return Solver name (e.g., "OLS", "Ridge", "ElasticNet", "WLS", "RLS")
	 */
	virtual std::string GetName() const = 0;

	/**
	 * Get the solver type identifier
	 *
	 * @return Solver type (e.g., "ols", "ridge", "elastic_net", "wls", "rls")
	 */
	virtual std::string GetType() const = 0;

	/**
	 * Fit the regression model
	 *
	 * This is the core method that all regression solvers must implement.
	 * It computes the regression coefficients and basic fit statistics.
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p, column-major)
	 * @param options Regression options (intercept, lambda, alpha, etc.)
	 * @return RegressionResult with coefficients and fit statistics
	 *
	 * @throws std::invalid_argument if inputs are invalid (dimension mismatch, etc.)
	 * @throws std::runtime_error if numerical computation fails
	 */
	virtual core::RegressionResult Fit(
	    const Eigen::VectorXd &y,
	    const Eigen::MatrixXd &X,
	    const core::RegressionOptions &options) const = 0;

	/**
	 * Fit the regression model with standard errors
	 *
	 * In addition to coefficients and fit statistics, this method also computes
	 * standard errors for the coefficients, enabling statistical inference
	 * (t-statistics, p-values, confidence intervals).
	 *
	 * Note: For some solvers (e.g., Elastic Net with L1 penalty), standard errors
	 * may be approximate or unavailable (returned as NaN). In such cases,
	 * bootstrap methods are recommended for uncertainty quantification.
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p, column-major)
	 * @param options Regression options
	 * @return RegressionResult with coefficients, fit statistics, and std_errors
	 *
	 * @throws std::invalid_argument if inputs are invalid
	 * @throws std::runtime_error if numerical computation fails
	 */
	virtual core::RegressionResult FitWithStdErrors(
	    const Eigen::VectorXd &y,
	    const Eigen::MatrixXd &X,
	    const core::RegressionOptions &options) const = 0;

	/**
	 * Check if this solver supports a given set of options
	 *
	 * This method validates whether the solver can handle the provided options.
	 * For example:
	 * - OLS requires lambda=0 and alpha=0 (no regularization)
	 * - Ridge requires lambda>0 and alpha=0 (L2 only)
	 * - ElasticNet requires lambda>0 and alpha in (0,1] (L1+L2)
	 * - RLS requires forgetting_factor in (0,1]
	 *
	 * @param options Regression options to validate
	 * @return true if this solver supports the options, false otherwise
	 */
	virtual bool SupportsOptions(const core::RegressionOptions &options) const = 0;

	/**
	 * Get a description of what this solver does
	 *
	 * @return Human-readable description of the algorithm
	 */
	virtual std::string GetDescription() const = 0;

	/**
	 * Check if this solver supports standard error computation
	 *
	 * Some solvers (e.g., Elastic Net) may not provide accurate standard errors
	 * due to the nature of the algorithm (L1 penalty is not differentiable).
	 *
	 * @return true if FitWithStdErrors() provides reliable standard errors
	 */
	virtual bool SupportsStdErrors() const = 0;

	/**
	 * Check if this solver can handle rank-deficient design matrices
	 *
	 * Solvers like OLS use QR decomposition with column pivoting to detect
	 * and handle rank deficiency. Others may require full-rank matrices.
	 *
	 * @return true if solver can handle rank-deficient X
	 */
	virtual bool SupportsRankDeficiency() const = 0;

	/**
	 * Get the recommended use case for this solver
	 *
	 * @return Description of when to use this solver (e.g., "Simple linear regression",
	 *         "Multicollinearity with L2 penalty", "Feature selection with L1+L2")
	 */
	virtual std::string GetUseCase() const = 0;
};

/**
 * RegressorAdapter: Template adapter for static solver classes
 *
 * This template wraps static solver classes (OLSSolver, RidgeSolver, etc.) to
 * make them conform to the IRegressor interface. This enables polymorphic use
 * of solvers without modifying the original solver classes.
 *
 * Example usage:
 * ```cpp
 * std::unique_ptr<IRegressor> solver = std::make_unique<RegressorAdapter<OLSSolver>>(
 *     "OLS", "ols", "Ordinary Least Squares", "Simple linear regression");
 *
 * auto result = solver->Fit(y, X, options);
 * ```
 *
 * Template parameter TSolver must provide:
 * - static RegressionResult Fit(y, X, options)
 * - static RegressionResult FitWithStdErrors(y, X, options)
 */
template <typename TSolver>
class RegressorAdapter : public IRegressor {
private:
	std::string name_;
	std::string type_;
	std::string description_;
	std::string use_case_;
	bool supports_std_errors_;
	bool supports_rank_deficiency_;

public:
	/**
	 * Constructor
	 *
	 * @param name Solver name (e.g., "OLS")
	 * @param type Solver type identifier (e.g., "ols")
	 * @param description What the solver does
	 * @param use_case When to use this solver
	 * @param supports_std_errors Whether standard errors are reliable
	 * @param supports_rank_deficiency Whether solver handles rank deficiency
	 */
	RegressorAdapter(
	    const std::string &name,
	    const std::string &type,
	    const std::string &description,
	    const std::string &use_case,
	    bool supports_std_errors = true,
	    bool supports_rank_deficiency = true)
	    : name_(name), type_(type), description_(description), use_case_(use_case),
	      supports_std_errors_(supports_std_errors),
	      supports_rank_deficiency_(supports_rank_deficiency) {
	}

	std::string GetName() const override {
		return name_;
	}

	std::string GetType() const override {
		return type_;
	}

	std::string GetDescription() const override {
		return description_;
	}

	std::string GetUseCase() const override {
		return use_case_;
	}

	bool SupportsStdErrors() const override {
		return supports_std_errors_;
	}

	bool SupportsRankDeficiency() const override {
		return supports_rank_deficiency_;
	}

	core::RegressionResult Fit(
	    const Eigen::VectorXd &y,
	    const Eigen::MatrixXd &X,
	    const core::RegressionOptions &options) const override {
		return TSolver::Fit(y, X, options);
	}

	core::RegressionResult FitWithStdErrors(
	    const Eigen::VectorXd &y,
	    const Eigen::MatrixXd &X,
	    const core::RegressionOptions &options) const override {
		return TSolver::FitWithStdErrors(y, X, options);
	}

	bool SupportsOptions(const core::RegressionOptions &options) const override {
		// Default implementation: all options are supported
		// Derived classes can override for specific validation
		return true;
	}
};

} // namespace solvers
} // namespace libanostat
