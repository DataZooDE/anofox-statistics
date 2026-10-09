#pragma once

#include "duckdb.hpp"
#include "anofox_stats_ffi.h"

#include <cmath>
#include <limits>
#include <vector>

namespace duckdb {

//! Leverage-aware prediction intervals for one fitted group or window frame.
//!
//! The interval of a new row x0 is yhat +/- t * s * sqrt(1 + x0' M x0), where M is
//! built once from the training design (see anofox_interval_matrix): (X'X)^-1 for
//! OLS, (X'WX)^-1 with weights, and the ridge sandwich A X'X A for ridge. For the
//! robust and sparse estimators the OLS leverage over the training rows (active
//! columns only, for elastic net) is used as an approximation. The previous formula,
//! s * sqrt(1 + 1/n), is only right at the centroid of x and was too narrow when
//! extrapolating.
//!
//! If M cannot be formed (e.g. a singular design), predictions are still returned
//! but the bounds are NaN, which callers write as NULL.
class LeverageIntervals {
public:
	LeverageIntervals(const vector<AnofoxDataArray> &x_train, const double *coefficients, idx_t coefficients_len,
	                  double intercept, bool fit_intercept, double residual_std_error, idx_t n_observations,
	                  const AnofoxDataArray *weights = nullptr, double ridge_lambda = 0.0)
	    : coefficients(coefficients), coefficients_len(coefficients_len),
	      intercept(fit_intercept ? intercept : std::numeric_limits<double>::quiet_NaN()),
	      residual_std_error(residual_std_error), n_observations(n_observations) {
		n_params = fit_intercept ? 1 : 0;
		for (idx_t j = 0; j < coefficients_len; j++) {
			if (std::isfinite(coefficients[j]) && coefficients[j] != 0.0) {
				n_params++;
			}
		}
		AnofoxError error;
		size_t dim_out = 0;
		if (!anofox_interval_matrix(x_train.data(), x_train.size(), coefficients, coefficients_len, fit_intercept,
		                            weights, ridge_lambda, &matrix, &dim_out, &error)) {
			matrix = nullptr;
		}
		dim = dim_out;
	}

	~LeverageIntervals() {
		anofox_free_interval_matrix(matrix);
	}

	LeverageIntervals(const LeverageIntervals &) = delete;
	LeverageIntervals &operator=(const LeverageIntervals &) = delete;

	//! Predict x_new. Returns false only if no point prediction is possible.
	bool Predict(const double *x_new, idx_t x_len, double confidence_level, AnofoxPredictionResult &out) const {
		if (residual_std_error == 0.0) {
			// Exact fit: the interval has zero width.
			out.yhat = PointPrediction(x_new, x_len);
			out.yhat_lower = out.yhat;
			out.yhat_upper = out.yhat;
			return true;
		}
		if (matrix) {
			AnofoxError error;
			if (anofox_predict_with_interval_matrix(coefficients, coefficients_len, intercept, x_new, x_len, matrix,
			                                        dim, n_observations, n_params, residual_std_error,
			                                        confidence_level, 0, &out, &error)) {
				return true;
			}
		}
		// No leverage matrix: point prediction only.
		out.yhat = PointPrediction(x_new, x_len);
		out.yhat_lower = std::numeric_limits<double>::quiet_NaN();
		out.yhat_upper = std::numeric_limits<double>::quiet_NaN();
		return true;
	}

private:
	double PointPrediction(const double *x_new, idx_t x_len) const {
		double yhat = std::isfinite(intercept) ? intercept : 0.0;
		for (idx_t j = 0; j < coefficients_len && j < x_len; j++) {
			if (std::isfinite(coefficients[j])) {
				yhat += coefficients[j] * x_new[j];
			}
		}
		return yhat;
	}

	const double *coefficients;
	idx_t coefficients_len;
	double intercept;
	double residual_std_error;
	idx_t n_observations;
	idx_t n_params = 0;
	double *matrix = nullptr;
	idx_t dim = 0;
};

//! Write a bound into a DOUBLE vector, as NULL when it is not finite.
inline void WriteIntervalBound(Vector &vec, idx_t idx, double value) {
	if (std::isfinite(value)) {
		FlatVector::GetData<double>(vec)[idx] = value;
	} else {
		FlatVector::SetNull(vec, idx, true);
	}
}

} // namespace duckdb
