#include "map_options_parser.hpp"

#include <limits>
#include <unordered_map>
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"

namespace duckdb {

// Helper to convert string to lowercase
static string ToLower(const string &str) {
	string result = str;
	for (auto &c : result) {
		c = std::tolower(c);
	}
	return result;
}

// Helper to extract boolean from Value (supports BOOLEAN, INTEGER, FLOAT, DECIMAL)
static std::optional<bool> ExtractBool(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	switch (val.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return BooleanValue::Get(val);
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
		return val.GetValue<int64_t>() != 0;
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return val.GetValue<uint64_t>() != 0;
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return val.GetValue<double>() != 0.0;
	default:
		throw InvalidInputException("Cannot convert value of type %s to boolean", val.type().ToString());
	}
}

// Helper to extract double from Value
static std::optional<double> ExtractDouble(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	return val.GetValue<double>();
}

// Helper to extract uint32 from Value
static std::optional<uint32_t> ExtractUInt32(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	auto v = val.GetValue<int64_t>();
	if (v < 0) {
		throw InvalidInputException("Expected non-negative integer, got %lld", v);
	}
	return static_cast<uint32_t>(v);
}

// Helper to extract uint64 from Value (used for seeds; accepts any non-negative
// integer that fits in int64).
static std::optional<uint64_t> ExtractUInt64(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	auto v = val.GetValue<int64_t>();
	if (v < 0) {
		throw InvalidInputException("Expected non-negative integer, got %lld", v);
	}
	return static_cast<uint64_t>(v);
}

// Helper to extract NullPolicy from Value
static std::optional<NullPolicy> ExtractNullPolicy(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "drop") {
		return NullPolicy::DROP;
	} else if (str == "drop_y_zero_x") {
		return NullPolicy::DROP_Y_ZERO_X;
	} else {
		throw InvalidInputException("Invalid null_policy: '%s'. Valid values are 'drop', 'drop_y_zero_x'", str);
	}
}

// Helper to extract PoissonLink from Value
static std::optional<PoissonLink> ExtractPoissonLink(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "log") {
		return PoissonLink::LOG;
	} else if (str == "identity") {
		return PoissonLink::IDENTITY;
	} else if (str == "sqrt") {
		return PoissonLink::SQRT;
	} else {
		throw InvalidInputException("Invalid poisson link: '%s'. Valid values are 'log', 'identity', 'sqrt'", str);
	}
}

// Helper to extract BinomialLink from Value
static std::optional<BinomialLink> ExtractBinomialLink(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "logit") {
		return BinomialLink::LOGIT;
	} else if (str == "probit") {
		return BinomialLink::PROBIT;
	} else if (str == "cloglog") {
		return BinomialLink::CLOGLOG;
	} else {
		throw InvalidInputException("Invalid binomial link: '%s'. Valid values are 'logit', 'probit', 'cloglog'", str);
	}
}

// Helper to extract AlmDistribution from Value
static std::optional<AlmDistribution> ExtractAlmDistribution(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "normal")
		return AlmDistribution::NORMAL;
	if (str == "laplace")
		return AlmDistribution::LAPLACE;
	if (str == "student_t" || str == "studentt")
		return AlmDistribution::STUDENT_T;
	if (str == "logistic")
		return AlmDistribution::LOGISTIC;
	if (str == "asymmetric_laplace" || str == "asymmetriclaplace")
		return AlmDistribution::ASYMMETRIC_LAPLACE;
	if (str == "generalised_normal" || str == "generalisednormal")
		return AlmDistribution::GENERALISED_NORMAL;
	if (str == "s")
		return AlmDistribution::S;
	if (str == "log_normal" || str == "lognormal")
		return AlmDistribution::LOG_NORMAL;
	if (str == "log_laplace" || str == "loglaplace")
		return AlmDistribution::LOG_LAPLACE;
	if (str == "log_s" || str == "logs")
		return AlmDistribution::LOG_S;
	if (str == "log_generalised_normal" || str == "loggeneralisednormal")
		return AlmDistribution::LOG_GENERALISED_NORMAL;
	if (str == "folded_normal" || str == "foldednormal")
		return AlmDistribution::FOLDED_NORMAL;
	if (str == "rectified_normal" || str == "rectifiednormal")
		return AlmDistribution::RECTIFIED_NORMAL;
	if (str == "box_cox_normal" || str == "boxcoxnormal")
		return AlmDistribution::BOX_COX_NORMAL;
	if (str == "gamma")
		return AlmDistribution::GAMMA;
	if (str == "inverse_gaussian" || str == "inversegaussian")
		return AlmDistribution::INVERSE_GAUSSIAN;
	if (str == "exponential")
		return AlmDistribution::EXPONENTIAL;
	if (str == "beta")
		return AlmDistribution::BETA;
	if (str == "logit_normal" || str == "logitnormal")
		return AlmDistribution::LOGIT_NORMAL;
	if (str == "poisson")
		return AlmDistribution::POISSON;
	if (str == "negative_binomial" || str == "negativebinomial" || str == "negbinomial")
		return AlmDistribution::NEGATIVE_BINOMIAL;
	if (str == "binomial")
		return AlmDistribution::BINOMIAL;
	if (str == "geometric")
		return AlmDistribution::GEOMETRIC;
	if (str == "cumulative_logistic" || str == "cumulativelogistic")
		return AlmDistribution::CUMULATIVE_LOGISTIC;
	if (str == "cumulative_normal" || str == "cumulativenormal")
		return AlmDistribution::CUMULATIVE_NORMAL;
	throw InvalidInputException("Invalid ALM distribution: '%s'", str);
}

// Helper to extract AlmLoss from Value
static std::optional<AlmLoss> ExtractAlmLoss(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "likelihood")
		return AlmLoss::LIKELIHOOD;
	if (str == "mse")
		return AlmLoss::MSE;
	if (str == "mae")
		return AlmLoss::MAE;
	if (str == "ham")
		return AlmLoss::HAM;
	if (str == "role")
		return AlmLoss::ROLE;
	throw InvalidInputException("Invalid ALM loss: '%s'. Valid values are 'likelihood', 'mse', 'mae', 'ham', 'role'",
	                            str);
}

// Helper to extract AidOutlierMethod from Value
static std::optional<AidOutlierMethod> ExtractAidOutlierMethod(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "zscore" || str == "z_score" || str == "z-score")
		return AidOutlierMethod::ZSCORE;
	if (str == "iqr")
		return AidOutlierMethod::IQR;
	throw InvalidInputException("Invalid outlier_method: '%s'. Valid values are 'zscore', 'iqr'", str);
}

// Helper to extract SolverType from Value
static std::optional<SolverType> ExtractSolverType(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "qr")
		return SolverType::QR;
	if (str == "svd")
		return SolverType::SVD;
	if (str == "cholesky")
		return SolverType::CHOLESKY;
	throw InvalidInputException("Invalid solver: '%s'. Valid values are 'qr', 'svd', 'cholesky'", str);
}

// Helper to extract HcType from Value
static std::optional<HcType> ExtractHcType(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "none")
		return HcType::NONE;
	if (str == "hc0")
		return HcType::HC0;
	if (str == "hc1")
		return HcType::HC1;
	if (str == "hc2")
		return HcType::HC2;
	if (str == "hc3")
		return HcType::HC3;
	throw InvalidInputException("Invalid hc_type: '%s'. Valid values are 'none', 'hc0', 'hc1', 'hc2', 'hc3'", str);
}

// Helper to extract LambdaScaling from Value
static std::optional<LambdaScaling> ExtractLambdaScaling(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "raw")
		return LambdaScaling::RAW;
	if (str == "glmnet")
		return LambdaScaling::GLMNET;
	throw InvalidInputException("Invalid lambda_scaling: '%s'. Valid values are 'raw', 'glmnet'", str);
}

// ============================================================================
// Statistical Test Option Extractors
// ============================================================================

// Helper to extract Alternative from Value
static std::optional<Alternative> ExtractAlternative(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "two_sided" || str == "two-sided" || str == "twosided" || str == "two.sided") {
		return Alternative::TWO_SIDED;
	} else if (str == "less" || str == "left") {
		return Alternative::LESS;
	} else if (str == "greater" || str == "right") {
		return Alternative::GREATER;
	}
	throw InvalidInputException("Invalid alternative: '%s'. Valid values are 'two_sided', 'less', 'greater'", str);
}

// Helper to extract KendallType from Value
static std::optional<KendallType> ExtractKendallType(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	if (str == "tau_a" || str == "taua" || str == "a")
		return KendallType::TAU_A;
	if (str == "tau_b" || str == "taub" || str == "b")
		return KendallType::TAU_B;
	if (str == "tau_c" || str == "tauc" || str == "c")
		return KendallType::TAU_C;
	throw InvalidInputException("Invalid kendall variant: '%s'. Valid values are 'tau_a', 'tau_b', 'tau_c'", str);
}

// Helper to extract TTestKind from Value
static std::optional<TTestKind> ExtractTTestKind(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	// Can be specified as bool (var_equal=true => STUDENT) or string (welch/student)
	switch (val.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return BooleanValue::Get(val) ? TTestKind::STUDENT : TTestKind::WELCH;
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
		return val.GetValue<int64_t>() != 0 ? TTestKind::STUDENT : TTestKind::WELCH;
	case LogicalTypeId::VARCHAR: {
		string str = ToLower(StringValue::Get(val));
		if (str == "student" || str == "equal")
			return TTestKind::STUDENT;
		if (str == "welch" || str == "unequal")
			return TTestKind::WELCH;
		throw InvalidInputException("Invalid t-test kind: '%s'. Valid values are 'student', 'welch'", str);
	}
	default:
		throw InvalidInputException("Cannot convert value of type %s to t-test kind", val.type().ToString());
	}
}

// ----------------------------------------------------------------------------
// Generic MAP / STRUCT option traversal
// ----------------------------------------------------------------------------
//
// DuckDB renders `{'key': value}` as either a MAP or a STRUCT depending on
// context, so every option parser has to handle both. This walks whichever shape
// arrived and hands (key, value) pairs to a callback, so the key handling below
// exists once instead of once per shape.
//
// The regression parser previously carried two verbatim copies of its ~100-line
// if-chain, which is why nested and list-valued options had nowhere natural to
// live. Adding one now only has to be done in a single place.
template <typename Callback>
static void VisitOptionEntries(const Value &map_value, Callback callback) {
	if (map_value.IsNull()) {
		return;
	}

	if (map_value.type().id() == LogicalTypeId::MAP) {
		// A MAP Value is a LIST of STRUCT(key, value).
		//
		// The previous code read it as a STRUCT of two parallel lists and threw
		// "Invalid MAP structure" otherwise. No test in the suite ever passed a
		// real MAP literal, so that branch was dead and the error was reachable by
		// any user who wrote `MAP {...}` instead of `{...}`.
		for (auto &entry : MapValue::GetChildren(map_value)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv.size() != 2) {
				throw InvalidInputException("Invalid MAP entry: expected a key and a value");
			}
			if (kv[0].IsNull()) {
				throw InvalidInputException("MAP option keys must not be NULL");
			}
			callback(ToLower(kv[0].ToString()), kv[1]);
		}
	} else if (map_value.type().id() == LogicalTypeId::STRUCT) {
		auto &children = StructValue::GetChildren(map_value);
		auto &child_types = StructType::GetChildTypes(map_value.type());
		for (idx_t i = 0; i < child_types.size(); i++) {
			callback(ToLower(child_types[i].first), children[i]);
		}
	} else {
		throw InvalidInputException("Expected MAP or STRUCT type for options, got %s", map_value.type().ToString());
	}
}

static std::optional<VcovTypeOpt> ExtractVcovType(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string v = ToLower(val.ToString());
	if (v == "laplace" || v == "curvature" || v == "posterior") {
		return VcovTypeOpt::LAPLACE;
	}
	if (v == "sandwich" || v == "robust") {
		return VcovTypeOpt::SANDWICH;
	}
	if (v == "naive" || v == "unpenalized" || v == "unpenalised") {
		return VcovTypeOpt::NAIVE;
	}
	throw InvalidInputException("Unknown vcov type '%s'. Expected 'laplace', 'sandwich' or 'naive'.", val.ToString());
}

static std::optional<vector<string>> ExtractStringList(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	if (val.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("feature_names must be a LIST of VARCHAR, got %s", val.type().ToString());
	}
	vector<string> out;
	for (auto &child : ListValue::GetChildren(val)) {
		if (child.IsNull()) {
			throw InvalidInputException("feature_names must not contain NULL");
		}
		out.push_back(child.ToString());
	}
	return out;
}

static PriorKindOpt ParsePriorKindName(const string &raw) {
	string v = ToLower(raw);
	if (v == "normal" || v == "gaussian") {
		return PriorKindOpt::NORMAL;
	}
	if (v == "laplace" || v == "l1" || v == "lasso") {
		return PriorKindOpt::LAPLACE;
	}
	if (v == "flat" || v == "none" || v == "uniform") {
		return PriorKindOpt::FLAT;
	}
	throw InvalidInputException("Unknown prior distribution '%s'. Expected 'normal', 'laplace' or 'flat'.", raw);
}

// One prior entry. Two spellings are accepted:
//
//   canonical:  {'dist': 'normal', 'loc': 0.0, 'scale': 1.0}
//   shorthand:  {'normal': [0.0, 1.0]}
//
// The canonical form exists because a DuckDB MAP requires a single value type, so
// the shorthand cannot mix families within one map. The shorthand is still accepted
// when every entry uses the same family, since that is what the issue asked for.
static PriorSpecOpt ParsePriorSpecValue(const string &feature, const Value &val) {
	if (val.IsNull()) {
		return PriorSpecOpt {};
	}
	if (val.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException(
		    "Prior for '%s' must be a STRUCT such as {'dist':'normal','loc':0.0,'scale':1.0}, got %s", feature,
		    val.type().ToString());
	}

	auto &children = StructValue::GetChildren(val);
	auto &child_types = StructType::GetChildTypes(val.type());

	PriorSpecOpt spec;
	bool have_dist = false;
	bool have_scale = false;

	for (idx_t i = 0; i < child_types.size(); i++) {
		string key = ToLower(child_types[i].first);
		const Value &child = children[i];

		if (key == "dist" || key == "distribution" || key == "kind") {
			spec.kind = ParsePriorKindName(child.ToString());
			have_dist = true;
		} else if (key == "loc" || key == "mean" || key == "mu") {
			spec.loc = child.GetValue<double>();
		} else if (key == "scale" || key == "sd" || key == "sigma") {
			spec.scale = child.GetValue<double>();
			have_scale = true;
		} else {
			// Shorthand: the key *is* the distribution name and the value is
			// [loc, scale].
			PriorKindOpt kind = ParsePriorKindName(key);
			if (child.type().id() != LogicalTypeId::LIST) {
				throw InvalidInputException("Prior shorthand for '%s' must be {'%s': [loc, scale]}", feature, key);
			}
			auto &pair = ListValue::GetChildren(child);
			if (pair.size() != 2) {
				throw InvalidInputException("Prior shorthand for '%s' needs exactly [loc, scale], got %llu values",
				                            feature, (unsigned long long)pair.size());
			}
			spec.kind = kind;
			spec.loc = pair[0].GetValue<double>();
			spec.scale = pair[1].GetValue<double>();
			have_dist = true;
			have_scale = true;
		}
	}

	if (!have_dist) {
		throw InvalidInputException("Prior for '%s' is missing a distribution; expected a 'dist' field", feature);
	}
	if (spec.kind != PriorKindOpt::FLAT) {
		if (!have_scale) {
			throw InvalidInputException("Prior for '%s' is missing 'scale'", feature);
		}
		if (!(spec.scale > 0.0)) {
			throw InvalidInputException("Prior scale for '%s' must be strictly positive, got %f", feature, spec.scale);
		}
	}
	return spec;
}

vector<PriorSpecOpt> RegressionMapOptions::ResolvePriors(idx_t n_features, bool fit_intercept) const {
	const idx_t n_params = n_features + (fit_intercept ? 1 : 0);
	vector<PriorSpecOpt> resolved(n_params);

	if (!prior_value.has_value() || prior_value->IsNull()) {
		return resolved;
	}

	// Name -> position. Without feature_names only the reserved intercept key can
	// be addressed, since the aggregate never sees any other name.
	unordered_map<string, idx_t> index_of;
	if (feature_names.has_value()) {
		const auto &names = *feature_names;
		if (names.size() != n_features) {
			throw InvalidInputException("feature_names has %llu entries but x has %llu features",
			                            (unsigned long long)names.size(), (unsigned long long)n_features);
		}
		for (idx_t i = 0; i < names.size(); i++) {
			index_of[ToLower(names[i])] = i + (fit_intercept ? 1 : 0);
		}
	}

	std::optional<PriorSpecOpt> fallback;

	VisitOptionEntries(*prior_value, [&](const string &key, const Value &val) {
		PriorSpecOpt spec = ParsePriorSpecValue(key, val);

		if (key == "_default" || key == "default") {
			fallback = spec;
			return;
		}
		if (key == "(intercept)" || key == "intercept" || key == "_intercept") {
			if (!fit_intercept) {
				throw InvalidInputException("A prior was given for the intercept but fit_intercept is false");
			}
			resolved[0] = spec;
			return;
		}

		auto it = index_of.find(key);
		if (it == index_of.end()) {
			// Deliberately louder than the rest of the parser, which ignores
			// unknown keys for forward compatibility. A silently dropped prior
			// changes the estimate without any signal, so it is an error.
			if (!feature_names.has_value()) {
				throw InvalidInputException("Prior given for '%s' but no feature_names option was supplied, so "
				                            "names cannot be resolved to columns. Add "
				                            "'feature_names': ['...'] listing the x columns in order.",
				                            key);
			}
			throw InvalidInputException("Prior given for unknown feature '%s'. Known features: %s", key,
			                            StringUtil::Join(*feature_names, ", "));
		}
		resolved[it->second] = spec;
	});

	if (fallback.has_value()) {
		const idx_t start = fit_intercept ? 1 : 0;
		for (idx_t j = start; j < n_params; j++) {
			if (resolved[j].kind == PriorKindOpt::FLAT &&
			    resolved[j].scale == std::numeric_limits<double>::infinity()) {
				resolved[j] = *fallback;
			}
		}
	}

	return resolved;
}

RegressionMapOptions RegressionMapOptions::ParseFromValue(const Value &map_value) {
	RegressionMapOptions result;

	VisitOptionEntries(map_value, [&](const string &key, const Value &val) {
		if (key == "intercept" || key == "fit_intercept") {
			result.fit_intercept = ExtractBool(val);
		} else if (key == "compute_inference" || key == "inference") {
			result.compute_inference = ExtractBool(val);
		} else if (key == "confidence_level" || key == "confidence") {
			result.confidence_level = ExtractDouble(val);
		} else if (key == "alpha") {
			result.alpha = ExtractDouble(val);
		} else if (key == "lambda") {
			result.lambda = ExtractDouble(val);
		} else if (key == "l1_ratio") {
			result.l1_ratio = ExtractDouble(val);
		} else if (key == "max_iterations" || key == "max_iter") {
			result.max_iterations = ExtractUInt32(val);
		} else if (key == "tolerance" || key == "tol") {
			result.tolerance = ExtractDouble(val);
		} else if (key == "epsilon") {
			result.epsilon = ExtractDouble(val);
		} else if (key == "residual_threshold") {
			result.residual_threshold = ExtractDouble(val);
		} else if (key == "max_trials") {
			result.max_trials = ExtractUInt32(val);
		} else if (key == "stop_probability") {
			result.stop_probability = ExtractDouble(val);
		} else if (key == "stop_n_inliers") {
			result.stop_n_inliers = ExtractUInt32(val);
		} else if (key == "min_samples") {
			result.min_samples = ExtractUInt32(val);
		} else if (key == "random_state" || key == "seed") {
			result.random_state = ExtractUInt64(val);
		} else if (key == "max_subpopulation") {
			result.max_subpopulation = ExtractUInt32(val);
		} else if (key == "n_subsamples") {
			result.n_subsamples = ExtractUInt32(val);
		} else if (key == "forgetting_factor") {
			result.forgetting_factor = ExtractDouble(val);
		} else if (key == "initial_p_diagonal" || key == "p_diagonal") {
			result.initial_p_diagonal = ExtractDouble(val);
		} else if (key == "null_policy") {
			result.null_policy = ExtractNullPolicy(val);
		}
		// GLM options
		else if (key == "link" || key == "poisson_link") {
			result.poisson_link = ExtractPoissonLink(val);
		} else if (key == "binomial_link") {
			result.binomial_link = ExtractBinomialLink(val);
		} else if (key == "power" || key == "tweedie_power") {
			result.tweedie_power = ExtractDouble(val);
		}
		// ALM options
		else if (key == "distribution" || key == "dist") {
			result.distribution = ExtractAlmDistribution(val);
		} else if (key == "loss") {
			result.loss = ExtractAlmLoss(val);
		} else if (key == "quantile") {
			result.quantile = ExtractDouble(val);
		} else if (key == "role_trim") {
			result.role_trim = ExtractDouble(val);
		}
		// BLS options
		else if (key == "lower_bound" || key == "lower") {
			result.lower_bound = ExtractDouble(val);
		} else if (key == "upper_bound" || key == "upper") {
			result.upper_bound = ExtractDouble(val);
		}
		// AID options
		else if (key == "intermittent_threshold") {
			result.intermittent_threshold = ExtractDouble(val);
		} else if (key == "outlier_method") {
			result.outlier_method = ExtractAidOutlierMethod(val);
		}
		// PLS options
		else if (key == "n_components" || key == "components") {
			auto v = ExtractUInt32(val);
			if (v.has_value()) {
				result.n_components = static_cast<size_t>(v.value());
			}
		}
		// Quantile options
		else if (key == "tau") {
			result.tau = ExtractDouble(val);
		}
		// Isotonic options
		else if (key == "increasing") {
			result.increasing = ExtractBool(val);
		}
		// Solver/inference options
		else if (key == "solver") {
			result.solver = ExtractSolverType(val);
		} else if (key == "hc_type") {
			result.hc_type = ExtractHcType(val);
		}
		// Lambda scaling
		else if (key == "lambda_scaling") {
			result.lambda_scaling = ExtractLambdaScaling(val);
		}
		// GLM regularization
		else if (key == "glm_lambda") {
			result.glm_lambda = ExtractDouble(val);
		}
		// Classification (Logistic)
		else if (key == "threshold") {
			result.threshold = ExtractDouble(val);
		}
		// Priors, feature names and covariance type. Unlike the keys above, an
		// unrecognised *prior* key is an error rather than a silent no-op: a
		// dropped prior changes the estimate, so failing loudly is the safer
		// default. See ParsePriorMap.
		else if (key == "feature_names" || key == "features") {
			result.feature_names = ExtractStringList(val);
		} else if (key == "prior" || key == "priors") {
			result.prior_value = val;
		} else if (key == "theta" || key == "nb_theta" || key == "dispersion") {
			result.nb_theta = ExtractDouble(val);
		} else if (key == "vcov" || key == "vcov_type") {
			result.vcov = ExtractVcovType(val);
		}
		// Unknown keys are silently ignored for forward compatibility
	});

	return result;
}

RegressionMapOptions RegressionMapOptions::ParseFromExpression(ClientContext &context, Expression &expr) {
	if (!expr.IsFoldable()) {
		throw InvalidInputException("Options parameter must be a constant expression");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, expr);
	return ParseFromValue(val);
}

// ============================================================================
// Statistical Test Option Parsers
// ============================================================================

// Generic helper template for extracting options from MAP/STRUCT
template <typename T, typename Callback>
static T ParseTestOptions(const Value &map_value, Callback callback) {
	T result;

	if (map_value.IsNull()) {
		return result;
	}

	if (map_value.type().id() == LogicalTypeId::MAP) {
		auto &children = StructValue::GetChildren(map_value);
		if (children.size() != 2) {
			throw InvalidInputException("Invalid MAP structure");
		}

		auto &keys = ListValue::GetChildren(children[0]);
		auto &values = ListValue::GetChildren(children[1]);

		if (keys.size() != values.size()) {
			throw InvalidInputException("MAP keys and values have different lengths");
		}

		for (idx_t i = 0; i < keys.size(); i++) {
			string key = ToLower(StringValue::Get(keys[i]));
			const Value &val = values[i];
			callback(result, key, val);
		}
	} else if (map_value.type().id() == LogicalTypeId::STRUCT) {
		auto &struct_type = map_value.type();
		auto &children = StructValue::GetChildren(map_value);
		auto &child_types = StructType::GetChildTypes(struct_type);

		for (idx_t i = 0; i < child_types.size(); i++) {
			string key = ToLower(child_types[i].first);
			const Value &val = children[i];
			callback(result, key, val);
		}
	} else {
		throw InvalidInputException("Expected MAP or STRUCT type for options, got %s", map_value.type().ToString());
	}

	return result;
}

TTestMapOptions TTestMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<TTestMapOptions>(map_value,
	                                         [](TTestMapOptions &result, const string &key, const Value &val) {
		                                         if (key == "alternative") {
			                                         result.alternative = ExtractAlternative(val);
		                                         } else if (key == "confidence_level" || key == "confidence") {
			                                         result.confidence_level = ExtractDouble(val);
		                                         } else if (key == "kind" || key == "var_equal") {
			                                         result.kind = ExtractTTestKind(val);
		                                         } else if (key == "paired") {
			                                         result.paired = ExtractBool(val);
		                                         } else if (key == "mu") {
			                                         result.mu = ExtractDouble(val);
		                                         }
	                                         });
}

MannWhitneyMapOptions MannWhitneyMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<MannWhitneyMapOptions>(
	    map_value, [](MannWhitneyMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level" || key == "confidence") {
			    result.confidence_level = ExtractDouble(val);
		    } else if (key == "continuity_correction" || key == "correction") {
			    result.continuity_correction = ExtractBool(val);
		    }
	    });
}

WilcoxonMapOptions WilcoxonMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<WilcoxonMapOptions>(map_value,
	                                            [](WilcoxonMapOptions &result, const string &key, const Value &val) {
		                                            if (key == "alternative") {
			                                            result.alternative = ExtractAlternative(val);
		                                            } else if (key == "confidence_level" || key == "confidence") {
			                                            result.confidence_level = ExtractDouble(val);
		                                            } else if (key == "continuity_correction" || key == "correction") {
			                                            result.continuity_correction = ExtractBool(val);
		                                            }
	                                            });
}

BrunnerMunzelMapOptions BrunnerMunzelMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<BrunnerMunzelMapOptions>(
	    map_value, [](BrunnerMunzelMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level" || key == "confidence") {
			    result.confidence_level = ExtractDouble(val);
		    }
	    });
}

CorrelationMapOptions CorrelationMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<CorrelationMapOptions>(
	    map_value, [](CorrelationMapOptions &result, const string &key, const Value &val) {
		    if (key == "confidence_level" || key == "confidence") {
			    result.confidence_level = ExtractDouble(val);
		    }
	    });
}

KendallMapOptions KendallMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<KendallMapOptions>(map_value,
	                                           [](KendallMapOptions &result, const string &key, const Value &val) {
		                                           if (key == "confidence_level" || key == "confidence") {
			                                           result.confidence_level = ExtractDouble(val);
		                                           } else if (key == "variant" || key == "tau_type" || key == "type") {
			                                           result.variant = ExtractKendallType(val);
		                                           }
	                                           });
}

ChiSquareMapOptions ChiSquareMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<ChiSquareMapOptions>(
	    map_value, [](ChiSquareMapOptions &result, const string &key, const Value &val) {
		    if (key == "continuity_correction" || key == "correction" || key == "yates") {
			    result.continuity_correction = ExtractBool(val);
		    }
	    });
}

FisherExactMapOptions FisherExactMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<FisherExactMapOptions>(
	    map_value, [](FisherExactMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    }
	    });
}

EnergyDistanceMapOptions EnergyDistanceMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<EnergyDistanceMapOptions>(
	    map_value, [](EnergyDistanceMapOptions &result, const string &key, const Value &val) {
		    if (key == "n_permutations" || key == "permutations") {
			    result.n_permutations = ExtractUInt32(val);
		    }
	    });
}

MmdMapOptions MmdMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<MmdMapOptions>(map_value, [](MmdMapOptions &result, const string &key, const Value &val) {
		if (key == "bandwidth" || key == "sigma") {
			result.bandwidth = ExtractDouble(val);
		} else if (key == "n_permutations" || key == "permutations") {
			result.n_permutations = ExtractUInt32(val);
		}
	});
}

TostMapOptions TostMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<TostMapOptions>(map_value, [](TostMapOptions &result, const string &key, const Value &val) {
		if (key == "alternative") {
			result.alternative = ExtractAlternative(val);
		} else if (key == "confidence_level" || key == "confidence") {
			result.confidence_level = ExtractDouble(val);
		} else if (key == "kind" || key == "var_equal") {
			result.kind = ExtractTTestKind(val);
		} else if (key == "paired") {
			result.paired = ExtractBool(val);
		} else if (key == "mu") {
			result.mu = ExtractDouble(val);
		} else if (key == "delta" || key == "equivalence_bound") {
			result.delta = ExtractDouble(val);
		} else if (key == "bound_lower" || key == "lower" || key == "low") {
			result.bound_lower = ExtractDouble(val);
		} else if (key == "bound_upper" || key == "upper" || key == "high") {
			result.bound_upper = ExtractDouble(val);
		}
	});
}

YuenMapOptions YuenMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<YuenMapOptions>(map_value, [](YuenMapOptions &result, const string &key, const Value &val) {
		if (key == "alternative") {
			result.alternative = ExtractAlternative(val);
		} else if (key == "confidence_level" || key == "confidence") {
			result.confidence_level = ExtractDouble(val);
		} else if (key == "trim" || key == "trim_proportion") {
			result.trim = ExtractDouble(val);
		}
	});
}

PermutationMapOptions PermutationMapOptions::ParseFromValue(const Value &map_value) {
	return ParseTestOptions<PermutationMapOptions>(
	    map_value, [](PermutationMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "n_permutations" || key == "permutations") {
			    result.n_permutations = ExtractUInt32(val);
		    }
	    });
}

} // namespace duckdb
