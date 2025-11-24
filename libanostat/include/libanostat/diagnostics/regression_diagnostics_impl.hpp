#pragma once

#include "regression_diagnostics.hpp"
#include <stdexcept>

namespace libanostat {
namespace diagnostics {

// Implementation of RegressionDiagnostics methods

inline Eigen::MatrixXd RegressionDiagnostics::ExtractValidColumns(
    const Eigen::MatrixXd &X,
    const std::vector<bool> &is_aliased) {

	size_t n = X.rows();
	size_t p = X.cols();

	// Count non-aliased features
	size_t n_valid = 0;
	for (size_t j = 0; j < p; j++) {
		if (!is_aliased[j]) {
			n_valid++;
		}
	}

	if (n_valid == 0) {
		return Eigen::MatrixXd(n, 0);
	}

	// Extract non-aliased columns
	Eigen::MatrixXd X_valid(n, n_valid);
	size_t valid_idx = 0;
	for (size_t j = 0; j < p; j++) {
		if (!is_aliased[j]) {
			X_valid.col(valid_idx) = X.col(j);
			valid_idx++;
		}
	}

	return X_valid;
}

inline Eigen::VectorXd RegressionDiagnostics::ComputeLeverage(
    const Eigen::MatrixXd &X,
    const std::vector<bool> &is_aliased,
    bool intercept) {

	size_t n = X.rows();

	// Extract non-aliased columns
	Eigen::MatrixXd X_valid = ExtractValidColumns(X, is_aliased);

	if (X_valid.cols() == 0) {
		// No valid features
		if (intercept) {
			// Leverage for intercept-only model: h_i = 1/n for all i
			return Eigen::VectorXd::Constant(n, 1.0 / n);
		} else {
			// No features at all -> undefined leverage
			return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
		}
	}

	// If intercept=true, X is centered, we need to augment with 1s column
	Eigen::MatrixXd X_aug;
	if (intercept) {
		X_aug.resize(n, X_valid.cols() + 1);
		X_aug.col(0) = Eigen::VectorXd::Ones(n);
		X_aug.rightCols(X_valid.cols()) = X_valid;
	} else {
		X_aug = X_valid;
	}

	// Compute H = X(X'X)^{-1}X'
	// For efficiency, compute leverage as diagonal of H
	// h_i = x_i'(X'X)^{-1}x_i

	try {
		Eigen::MatrixXd XtX = X_aug.transpose() * X_aug;
		Eigen::MatrixXd XtX_inv = XtX.inverse();

		Eigen::VectorXd leverage(n);
		for (size_t i = 0; i < n; i++) {
			Eigen::VectorXd x_i = X_aug.row(i);
			leverage(i) = x_i.transpose() * XtX_inv * x_i;
		}

		return leverage;

	} catch (...) {
		// If computation fails, return NaN
		return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
	}
}

inline Eigen::VectorXd RegressionDiagnostics::ComputeStandardizedResiduals(
    const Eigen::VectorXd &residuals,
    const Eigen::VectorXd &leverage,
    double mse) {

	size_t n = residuals.size();
	Eigen::VectorXd std_residuals(n);

	for (size_t i = 0; i < n; i++) {
		if (std::isnan(residuals(i)) || std::isnan(leverage(i)) || std::isnan(mse)) {
			std_residuals(i) = std::numeric_limits<double>::quiet_NaN();
		} else {
			double denominator = std::sqrt(mse * (1.0 - leverage(i)));
			if (denominator <= 0.0 || leverage(i) >= 1.0) {
				std_residuals(i) = std::numeric_limits<double>::quiet_NaN();
			} else {
				std_residuals(i) = residuals(i) / denominator;
			}
		}
	}

	return std_residuals;
}

inline Eigen::VectorXd RegressionDiagnostics::ComputeCooksDistance(
    const Eigen::VectorXd &standardized_residuals,
    const Eigen::VectorXd &leverage,
    size_t n_params) {

	size_t n = standardized_residuals.size();
	Eigen::VectorXd cooks_d(n);

	for (size_t i = 0; i < n; i++) {
		if (std::isnan(standardized_residuals(i)) || std::isnan(leverage(i))) {
			cooks_d(i) = std::numeric_limits<double>::quiet_NaN();
		} else {
			double r_std_sq = standardized_residuals(i) * standardized_residuals(i);
			double h_i = leverage(i);

			if (h_i >= 1.0 || n_params == 0) {
				cooks_d(i) = std::numeric_limits<double>::quiet_NaN();
			} else {
				// D_i = (r_i^std)² * h_i / (p * (1 - h_i))
				cooks_d(i) = (r_std_sq * h_i) / (n_params * (1.0 - h_i));
			}
		}
	}

	return cooks_d;
}

inline Eigen::VectorXd RegressionDiagnostics::ComputeDFFITS(
    const Eigen::VectorXd &standardized_residuals,
    const Eigen::VectorXd &leverage) {

	size_t n = standardized_residuals.size();
	Eigen::VectorXd dffits(n);

	for (size_t i = 0; i < n; i++) {
		if (std::isnan(standardized_residuals(i)) || std::isnan(leverage(i))) {
			dffits(i) = std::numeric_limits<double>::quiet_NaN();
		} else {
			double h_i = leverage(i);

			if (h_i >= 1.0) {
				dffits(i) = std::numeric_limits<double>::quiet_NaN();
			} else {
				// DFFITS_i = r_i^std * sqrt(h_i / (1 - h_i))
				dffits(i) = standardized_residuals(i) * std::sqrt(h_i / (1.0 - h_i));
			}
		}
	}

	return dffits;
}

inline RegressionDiagnostics::DiagnosticsResult RegressionDiagnostics::ComputeAllDiagnostics(
    const core::RegressionResult &result,
    const Eigen::MatrixXd &X,
    bool intercept) {

	DiagnosticsResult diag;

	try {
		// Compute leverage values
		diag.leverage = ComputeLeverage(X, result.is_aliased, intercept);

		// Compute standardized residuals
		diag.standardized_residuals = ComputeStandardizedResiduals(
		    result.residuals, diag.leverage, result.mse);

		// Compute Cook's distance: rank now includes intercept if fitted
		size_t n_params = result.rank;
		diag.cooks_distance = ComputeCooksDistance(
		    diag.standardized_residuals, diag.leverage, n_params);

		// Compute DFFITS
		diag.dffits = ComputeDFFITS(diag.standardized_residuals, diag.leverage);

		diag.has_diagnostics = true;

	} catch (...) {
		// If any computation fails, return empty result
		diag.has_diagnostics = false;
	}

	return diag;
}

} // namespace diagnostics
} // namespace libanostat
