#pragma once

// Profile-likelihood confidence intervals of the GLM fit aggregates (#310).
//
// With the `profile_ci := true` option, a GLM fit replaces its Wald intervals
// (conf_low / conf_high, the ci_lower / ci_upper aliases and the intercept's
// interval) by profile-likelihood intervals, R's confint(glm) (MASS::confint.glm
// before R 4.4). anofox-regression (inference::profile_confint) profiles the same
// rows, columns and offset the fit used; a bound the profile does not reach is NULL
// (R's NA). The option implies compute_inference and needs an unpenalized fit.

#include <cmath>

#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#include "anofox_stats_ffi.h"
#include "error_dispatch.hpp"
#include "map_options_parser.hpp"

namespace duckdb {

//! Bind-time check of `profile_ci`: true implies inference, and a penalty
//! (glm_lambda > 0 or a prior) would make the profile of the likelihood meaningless.
inline bool ProfileCiOption(const RegressionMapOptions &opts, const char *fn) {
	if (!opts.profile_ci.value_or(false)) {
		return false;
	}
	if (opts.glm_lambda.value_or(0.0) > 0.0 || opts.HasPriors()) {
		throw InvalidInputException("%s: profile_ci needs an unpenalized fit (no glm_lambda, no prior)", fn);
	}
	return true;
}

//! Overwrite the Wald intervals of a successful GLM fit by profile intervals.
//! `x` are the columns the fit was given (offset column included).
inline void ApplyProfileConfint(const char *fn, AnofoxLikelihoodFamily family, AnofoxDataArray y,
                                const vector<AnofoxDataArray> &x, bool fit_intercept, idx_t offset_column,
                                double level, AnofoxGlmFitResultCore &core, AnofoxFitResultInference &inference) {
	const idx_t k = core.coefficients_len;
	vector<double> lower(k, std::nan("")), upper(k, std::nan(""));
	double intercept[2] = {std::nan(""), std::nan("")};
	AnofoxError error;
	if (!anofox_glm_profile_confint(family, y, x.data(), x.size(), core.coefficients, k, fit_intercept,
	                                static_cast<uint64_t>(offset_column), level, lower.data(), upper.data(),
	                                intercept, &error)) {
		// A data-dependent failure (e.g. a fit that had not converged) leaves NULL intervals.
		ThrowUnlessDegenerate(fn, error);
	}
	for (idx_t j = 0; j < inference.len && j < k; j++) {
		inference.ci_lower[j] = lower[j];
		inference.ci_upper[j] = upper[j];
	}
	if (fit_intercept) {
		core.summary.intercept_conf_low = intercept[0];
		core.summary.intercept_conf_high = intercept[1];
	}
}

} // namespace duckdb
