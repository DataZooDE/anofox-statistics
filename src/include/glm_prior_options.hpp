//===----------------------------------------------------------------------===//
//                         anofox_statistics
//
// glm_prior_options.hpp
//
// Shared plumbing for explicit coefficient priors and the covariance type
// (issue #107), used by all six GLM aggregate functions.
//
// Name resolution is deliberately two-stage. Options are parsed at bind time,
// where the options expression is foldable, but the feature count is not known
// until the first LIST arrives in update. So bind stashes the raw prior value and
// the feature names, and the first update resolves them into a positional array.
// Names never cross the FFI boundary — it stays flat POD.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"
#include "anofox_stats_ffi.h"
#include "map_options_parser.hpp"

#include <optional>

namespace duckdb {

//! Prior/covariance settings carried in an aggregate's bind data.
struct GlmPriorBindData {
	//! Raw `prior` option value, resolved once n_features is known.
	std::optional<Value> prior_value;
	//! Feature names, needed because the aggregate only sees `x LIST(DOUBLE)`.
	std::optional<vector<string>> feature_names;
	//! Requested covariance type.
	VcovTypeOpt vcov = VcovTypeOpt::LAPLACE;

	void LoadFrom(const RegressionMapOptions &opts) {
		if (opts.prior_value.has_value()) {
			prior_value = opts.prior_value;
		}
		if (opts.feature_names.has_value()) {
			feature_names = opts.feature_names;
		}
		if (opts.vcov.has_value()) {
			vcov = opts.vcov.value();
		}
	}

	bool Equals(const GlmPriorBindData &other) const {
		if (vcov != other.vcov) {
			return false;
		}
		if (prior_value.has_value() != other.prior_value.has_value()) {
			return false;
		}
		if (prior_value.has_value() && !Value::NotDistinctFrom(*prior_value, *other.prior_value)) {
			return false;
		}
		return feature_names == other.feature_names;
	}

	//! Rebuild a RegressionMapOptions carrying just the prior fields, so the
	//! resolution logic lives in one place (map_options_parser.cpp).
	vector<PriorSpecOpt> Resolve(idx_t n_features, bool fit_intercept) const {
		RegressionMapOptions shim;
		shim.prior_value = prior_value;
		shim.feature_names = feature_names;
		return shim.ResolvePriors(n_features, fit_intercept);
	}
};

//! Translate a parsed prior onto the flat FFI struct.
inline AnofoxPriorSpec ToFfiPrior(const PriorSpecOpt &spec) {
	AnofoxPriorSpec out;
	switch (spec.kind) {
	case PriorKindOpt::NORMAL:
		out.kind = ANOFOX_PRIOR_NORMAL;
		break;
	case PriorKindOpt::LAPLACE:
		out.kind = ANOFOX_PRIOR_LAPLACE;
		break;
	case PriorKindOpt::FLAT:
	default:
		out.kind = ANOFOX_PRIOR_FLAT;
		break;
	}
	out.loc = spec.loc;
	out.scale = spec.scale;
	return out;
}

inline AnofoxVcovType ToFfiVcov(VcovTypeOpt v) {
	switch (v) {
	case VcovTypeOpt::SANDWICH:
		return ANOFOX_VCOV_SANDWICH;
	case VcovTypeOpt::NAIVE:
		return ANOFOX_VCOV_NAIVE;
	case VcovTypeOpt::LAPLACE:
	default:
		return ANOFOX_VCOV_LAPLACE;
	}
}

//! Prior state carried by an aggregate, materialized on the first update once
//! the feature count is known.
struct GlmPriorState {
	vector<AnofoxPriorSpec> priors;
	AnofoxVcovType vcov = ANOFOX_VCOV_LAPLACE;

	//! Resolve names to positions. Call when the feature count first becomes known.
	void Materialize(const GlmPriorBindData &bind, idx_t n_features, bool fit_intercept) {
		vcov = ToFfiVcov(bind.vcov);
		priors.clear();
		if (!bind.prior_value.has_value()) {
			return;
		}
		for (auto &spec : bind.Resolve(n_features, fit_intercept)) {
			priors.push_back(ToFfiPrior(spec));
		}
	}

	void Clear() {
		priors.clear();
		vcov = ANOFOX_VCOV_LAPLACE;
	}

	//! Point an options struct at this state's priors.
	template <typename OPTIONS>
	void Apply(OPTIONS &options) const {
		options.priors = priors.empty() ? nullptr : priors.data();
		options.priors_len = priors.size();
		options.vcov = vcov;
	}
};

} // namespace duckdb
