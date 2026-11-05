#pragma once

// Conditioning-aware ridge lambda (#305): the 'conditioning', 'lambda_small' and
// 'cond_threshold' options of the ridge_* functions, carried from bind to the
// AnofoxRidgeOptions of the fit. The selection itself is upstream's
// (RidgeRegressorBuilder::conditioning_aware).

#include "duckdb.hpp"
#include "anofox_stats_ffi.h"
#include "error_dispatch.hpp"
#include "map_options_parser.hpp"

#include <cmath>
#include <limits>

namespace duckdb {

struct RidgeConditioningOpt {
	bool enabled = false;
	//! NaN = upstream's default.
	double lambda_small = std::numeric_limits<double>::quiet_NaN();
	double cond_threshold = std::numeric_limits<double>::quiet_NaN();

	static const vector<string> &Keys() {
		static const vector<string> keys = {"conditioning", "lambda_small", "cond_threshold"};
		return keys;
	}

	//! Append the conditioning keys to a function's supported option keys.
	static vector<string> WithKeys(vector<string> keys) {
		for (auto &k : Keys()) {
			keys.push_back(k);
		}
		return keys;
	}

	//! 'lambda_small' / 'cond_threshold' imply 'conditioning'; together with an
	//! explicit {'conditioning': false} they are an error.
	void Apply(const RegressionMapOptions &opts, const string &function_name) {
		bool tuned = opts.lambda_small.has_value() || opts.cond_threshold.has_value();
		if (opts.conditioning.has_value() && !opts.conditioning.value() && tuned) {
			throw InvalidInputException(
			    "%s: 'lambda_small' / 'cond_threshold' need conditioning-aware lambda, but 'conditioning' is false",
			    function_name);
		}
		enabled = opts.conditioning.value_or(tuned);
		if (opts.lambda_small.has_value()) {
			lambda_small = opts.lambda_small.value();
		}
		if (opts.cond_threshold.has_value()) {
			cond_threshold = opts.cond_threshold.value();
		}
	}

	void ToFFI(AnofoxRidgeOptions &options) const {
		options.conditioning = enabled;
		options.lambda_small = lambda_small;
		options.cond_threshold = cond_threshold;
	}

	bool operator==(const RidgeConditioningOpt &o) const {
		auto same = [](double a, double b) {
			return a == b || (std::isnan(a) && std::isnan(b));
		};
		return enabled == o.enabled && same(lambda_small, o.lambda_small) && same(cond_threshold, o.cond_threshold);
	}
	bool operator!=(const RidgeConditioningOpt &o) const {
		return !(*this == o);
	}
};

//! The lambda the ridge fit of (y, x) with `options` used (same scale as
//! options.alpha): options.alpha unless conditioning-aware lambda is on.
inline double RidgeSelectedLambda(const AnofoxDataArray &y, const vector<AnofoxDataArray> &x,
                                  const AnofoxRidgeOptions &options, const char *function_name) {
	if (!options.conditioning) {
		return options.alpha;
	}
	double lambda = options.alpha;
	AnofoxError error;
	if (!anofox_ridge_selected_lambda(y, x.data(), x.size(), options, &lambda, &error)) {
		ThrowUnlessDegenerate(function_name, error);
		return options.alpha;
	}
	return lambda;
}

} // namespace duckdb
