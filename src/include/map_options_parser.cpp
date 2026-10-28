#include <cerrno>
#include <cstdlib>
#include "map_options_parser.hpp"

#include <cmath>
#include <limits>
#include <unordered_map>
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/string_util.hpp"
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
	case LogicalTypeId::VARCHAR: {
		// A MAP literal coerces all values to one type, so booleans mixed with
		// strings arrive as 'true' / 'false'.
		string str = ToLower(StringValue::Get(val));
		if (str == "true" || str == "t" || str == "1") {
			return true;
		}
		if (str == "false" || str == "f" || str == "0") {
			return false;
		}
		throw InvalidInputException("Cannot convert '%s' to boolean", StringValue::Get(val));
	}
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

// confidence_level must lie strictly inside (0, 1).
static std::optional<double> ExtractConfidenceLevel(const Value &val, const string &function_name = string()) {
	auto v = ExtractDouble(val);
	if (v.has_value() && !(v.value() > 0.0 && v.value() < 1.0)) {
		if (function_name.empty()) {
			throw InvalidInputException("confidence_level must be in (0, 1), got %s", val.ToString());
		}
		throw InvalidInputException("%s: confidence_level must be in (0, 1), got %s", function_name, val.ToString());
	}
	return v;
}

// The option being parsed, so the integer extractors can name it in their
// errors ("max_iterations: expected a non-negative integer, got -1"). Set by the
// option visitors before each value is extracted.
static thread_local string current_option_key;

static string OptionKeyPrefix() {
	return current_option_key.empty() ? string() : current_option_key + ": ";
}

// Extract a non-negative integer no larger than `max_value`. Negative,
// fractional or overflowing values raise instead of being truncated.
static std::optional<uint64_t> ExtractBoundedUInt(const Value &val, uint64_t max_value) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	uint64_t result;
	switch (val.type().id()) {
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		result = val.GetValue<uint64_t>();
		break;
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL: {
		auto d = val.GetValue<double>();
		if (d < 0) {
			throw InvalidInputException("%sexpected a non-negative integer, got %s", OptionKeyPrefix(), val.ToString());
		}
		if (d != std::floor(d)) {
			throw InvalidInputException("%sexpected an integer, got %s", OptionKeyPrefix(), val.ToString());
		}
		if (d > static_cast<double>(max_value)) {
			throw InvalidInputException("%sinteger value %s is too large (maximum %llu)", OptionKeyPrefix(),
			                            val.ToString(), (unsigned long long)max_value);
		}
		result = static_cast<uint64_t>(d);
		break;
	}
	default: {
		auto v = val.GetValue<int64_t>();
		if (v < 0) {
			throw InvalidInputException("%sexpected a non-negative integer, got %lld", OptionKeyPrefix(), (long long)v);
		}
		result = static_cast<uint64_t>(v);
		break;
	}
	}
	if (result > max_value) {
		throw InvalidInputException("%sinteger value %s is too large (maximum %llu)", OptionKeyPrefix(),
		                            val.ToString(), (unsigned long long)max_value);
	}
	return result;
}

// Helper to extract uint32 from Value
static std::optional<uint32_t> ExtractUInt32(const Value &val) {
	auto v = ExtractBoundedUInt(val, std::numeric_limits<uint32_t>::max());
	if (!v.has_value()) {
		return std::nullopt;
	}
	return static_cast<uint32_t>(v.value());
}

// Helper to extract uint64 from Value (used for seeds).
static std::optional<uint64_t> ExtractUInt64(const Value &val) {
	return ExtractBoundedUInt(val, std::numeric_limits<uint64_t>::max());
}

// Helper to extract NullPolicy from Value
static std::optional<NullPolicy> ExtractNullPolicy(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string str = ToLower(StringValue::Get(val));
	// 'ignore' (no handling, nulls already removed upstream) is an alias for 'drop':
	// a NULL can never reach the solver, so skipping such rows is the least handling possible.
	if (str == "drop" || str == "ignore") {
		return NullPolicy::DROP;
	} else if (str == "drop_y_zero_x") {
		return NullPolicy::DROP_Y_ZERO_X;
	} else if (str == "zero") {
		return NullPolicy::ZERO;
	} else {
		throw InvalidInputException(
		    "Invalid null_policy: '%s'. Valid values are 'drop', 'drop_y_zero_x', 'zero', 'ignore'", str);
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
	return std::nullopt;
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
		if (str == "student" || str == "equal" || str == "true")
			return TTestKind::STUDENT;
		if (str == "welch" || str == "unequal" || str == "false")
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
			callback(ToLower(NameString(child_types[i].first)), children[i]);
		}
	} else {
		throw InvalidInputException("Expected MAP or STRUCT type for options, got %s", map_value.type().ToString());
	}
}

// Values mirror AnofoxGlmmDdf in anofox_stats_ffi.h.
static std::optional<int> ParseGlmmDdf(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string v = ToLower(val.ToString());
	if (v == "wald" || v == "z" || v == "asymptotic") {
		return 1;
	}
	if (v == "satterthwaite") {
		return 2;
	}
	throw InvalidInputException("Unknown ddf '%s'. Expected 'satterthwaite' or 'wald'.", val.ToString());
}

// Values mirror AnofoxGlmmFamily in anofox_stats_ffi.h.
static std::optional<int> ParseGlmmFamily(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	string v = ToLower(val.ToString());
	if (v == "gaussian" || v == "normal" || v == "lmm") {
		return 0;
	}
	if (v == "poisson") {
		return 1;
	}
	if (v == "binomial" || v == "logistic") {
		return 2;
	}
	if (v == "negbinom" || v == "negbinomial" || v == "negative_binomial" || v == "negbin") {
		return 3;
	}
	if (v == "gamma") {
		return 4;
	}
	if (v == "tweedie") {
		return 5;
	}
	throw InvalidInputException("Unknown GLMM family '%s'. Expected 'gaussian', 'poisson', "
	                            "'binomial', 'negbinom', 'gamma' or 'tweedie'.",
	                            val.ToString());
}

static bool IsAftDistName(const string &raw) {
	string v = ToLower(raw);
	return v == "weibull" || v == "lognormal" || v == "log_normal" || v == "log-normal" || v == "loglogistic" ||
	       v == "log_logistic" || v == "log-logistic" || v == "exponential" || v == "exp";
}

// Values mirror AnofoxAftDistribution in anofox_stats_ffi.h.
static AnofoxAftDistribution_t ParseAftDist(const string &raw) {
	string v = ToLower(raw);
	if (v == "weibull") {
		return 0;
	}
	if (v == "lognormal" || v == "log_normal" || v == "log-normal") {
		return 1;
	}
	if (v == "loglogistic" || v == "log_logistic" || v == "log-logistic") {
		return 2;
	}
	return 3; // exponential
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

// Extract a LIST of positive integers (1-based column indices).
static idx_t CheckedIndex(int64_t v) {
	if (v < 1) {
		throw InvalidInputException("'random' indices are 1-based and must be >= 1, got %lld", (long long)v);
	}
	return (idx_t)v;
}

static std::optional<vector<idx_t>> ExtractIndexList(const Value &val) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	vector<idx_t> out;
	// A list rendered as text ('[1, 2]'), as it arrives through a MAP(VARCHAR, VARCHAR)
	// of options -- the form the table macros pass on. Parsed by hand: the Value cast
	// API differs between DuckDB 1.x and 2.0.
	if (val.type().id() == LogicalTypeId::VARCHAR) {
		string text = StringUtil::Replace(StringUtil::Replace(val.ToString(), "[", ""), "]", "");
		StringUtil::Trim(text);
		if (text.empty()) {
			return out;
		}
		for (auto &item : StringUtil::Split(text, ',')) {
			StringUtil::Trim(item);
			char *end = nullptr;
			errno = 0;
			long long v = std::strtoll(item.c_str(), &end, 10);
			if (item.empty() || errno != 0 || end == nullptr || *end != '\0') {
				throw InvalidInputException("'random' must be a LIST of integer column indices, got '%s'",
				                            val.ToString());
			}
			out.push_back(CheckedIndex(v));
		}
		return out;
	}
	if (val.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("'random' must be a LIST of integer column indices, got %s",
		                            val.type().ToString());
	}
	for (auto &child : ListValue::GetChildren(val)) {
		if (child.IsNull()) {
			throw InvalidInputException("'random' must not contain NULL");
		}
		out.push_back(CheckedIndex(child.GetValue<int64_t>()));
	}
	return out;
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

// A LIST of numbers without NULL / non-finite entries (lambdas, taus).
static std::optional<vector<double>> ExtractDoubleList(const Value &val, const char *key) {
	if (val.IsNull()) {
		return std::nullopt;
	}
	if (val.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("%s must be a LIST of numbers, got %s", key, val.type().ToString());
	}
	vector<double> out;
	for (auto &child : ListValue::GetChildren(val)) {
		if (child.IsNull()) {
			throw InvalidInputException("%s must not contain NULL", key);
		}
		double v = child.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		if (!std::isfinite(v)) {
			throw InvalidInputException("%s must contain finite numbers, got %s", key, child.ToString());
		}
		out.push_back(v);
	}
	if (out.empty()) {
		throw InvalidInputException("%s must not be empty", key);
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
		string key = ToLower(NameString(child_types[i].first));
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
			// A silently dropped prior changes the estimate without any signal,
			// so an unresolvable name is an error.
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

// ----------------------------------------------------------------------------
// Option key tables
// ----------------------------------------------------------------------------
//
// Each option has one canonical key plus fixed aliases. Functions declare the
// canonical keys they read; anything else is rejected with an error naming the
// function and listing what it supports.

struct OptionKeyDef {
	const char *canonical;
	vector<const char *> aliases;
};

static const OptionKeyDef *FindOptionKey(const vector<OptionKeyDef> &table, const string &key) {
	for (auto &def : table) {
		if (key == def.canonical) {
			return &def;
		}
		for (auto alias : def.aliases) {
			if (key == alias) {
				return &def;
			}
		}
	}
	return nullptr;
}

static string DescribeOptionKey(const string &canonical, const vector<string> &aliases) {
	if (aliases.empty()) {
		return canonical;
	}
	return canonical + " (alias: " + StringUtil::Join(aliases, ", ") + ")";
}

static string DescribeAllKeys(const vector<OptionKeyDef> &table) {
	vector<string> parts;
	for (auto &def : table) {
		vector<string> aliases(def.aliases.begin(), def.aliases.end());
		parts.push_back(DescribeOptionKey(def.canonical, aliases));
	}
	return StringUtil::Join(parts, ", ");
}

static const vector<OptionKeyDef> &RegressionKeyTable() {
	static const vector<OptionKeyDef> table = {
	    {"fit_intercept", {"intercept"}},
	    {"compute_inference", {"inference"}},
	    {"compute_loo", {}},
	    {"confidence_level", {"confidence"}},
	    {"alpha", {}},
	    {"lambda", {}},
	    {"l1_ratio", {}},
	    {"max_iterations", {"max_iter"}},
	    {"tolerance", {"tol"}},
	    {"epsilon", {}},
	    {"residual_threshold", {}},
	    {"max_trials", {}},
	    {"stop_probability", {}},
	    {"stop_n_inliers", {}},
	    {"min_samples", {}},
	    {"random_state", {"seed"}},
	    {"max_subpopulation", {}},
	    {"n_subsamples", {}},
	    {"forgetting_factor", {}},
	    {"initial_p_diagonal", {"p_diagonal"}},
	    {"null_policy", {}},
	    {"link", {}},
	    {"poisson_link", {}},
	    {"binomial_link", {}},
	    {"power", {"tweedie_power"}},
	    {"distribution", {"dist"}},
	    {"loss", {}},
	    {"quantile", {}},
	    {"role_trim", {}},
	    {"lower_bound", {"lower"}},
	    {"upper_bound", {"upper"}},
	    {"intermittent_threshold", {}},
	    {"outlier_method", {}},
	    {"n_components", {"components"}},
	    {"tau", {}},
	    {"increasing", {}},
	    {"solver", {}},
	    {"hc_type", {}},
	    {"alias_tolerance", {}},
	    {"lambda_scaling", {}},
	    {"glm_lambda", {}},
	    {"threshold", {}},
	    {"feature_names", {"features"}},
	    {"prior", {"priors"}},
	    {"family", {}},
	    {"reml", {}},
	    {"offset", {}},
	    {"random", {"random_slopes"}},
	    {"groups", {"crossed"}},
	    {"ddf", {}},
	    {"vcov", {"vcov_type"}},
	    {"tau_squared", {"tau2"}},
	    {"tau_method", {"shrinkage"}},
	    {"theta", {"nb_theta", "dispersion"}},
	    {"method", {}},
	    {"n_nonzero_coefs", {}},
	    {"standardize", {}},
	    {"lambdas", {}},
	    {"n_lambda", {"nlambda"}},
	    {"lambda_min_ratio", {}},
	    {"folds", {"n_folds", "nfolds", "outer_folds"}},
	    {"cv_strategy", {"strategy"}},
	    {"repeats", {"n_repeats"}},
	    {"test_size", {"horizon"}},
	    {"gap", {}},
	    {"window", {}},
	    {"max_train_size", {}},
	    {"n_bins", {}},
	    {"metric", {"type_measure"}},
	    {"inner_folds", {}},
	    {"rule", {"lambda_rule"}},
	    {"taus", {}},
	    {"se", {"se_type"}},
	    {"n_boot", {"n_bootstrap"}},
	    {"ci_method", {}},
	    {"alpha_1", {}},
	    {"alpha_2", {}},
	    {"lambda_1", {}},
	    {"lambda_2", {}},
	    {"threshold_lambda", {}},
	    {"c", {}},
	    {"variant", {"pa_variant"}},
	    {"ic", {"information_criterion"}},
	    {"span", {"lowess_span", "f"}},
	};
	return table;
}

// Context-dependent aliases: a key whose meaning depends on which canonical
// keys the function supports. The first supported target wins. A key that the
// function supports under its own name is never redirected.
struct ContextualAlias {
	const char *source;
	vector<const char *> targets;
};

static const vector<ContextualAlias> &RegressionContextualAliases() {
	static const vector<ContextualAlias> aliases = {
	    {"link", {"poisson_link", "binomial_link"}},
	    // GLM penalty; for RLS the conventional symbol of the forgetting factor.
	    {"lambda", {"glm_lambda", "forgetting_factor"}},
	    {"alpha", {"glm_lambda"}},
	    {"tau", {"quantile"}},
	    {"quantile", {"tau"}},
	};
	return aliases;
}

static bool Contains(const vector<string> &keys, const string &key) {
	for (auto &k : keys) {
		if (k == key) {
			return true;
		}
	}
	return false;
}

// Resolve a contextual-alias source key to the canonical key the function
// supports, or "" when it does not apply.
static string ResolveContextualAlias(const string &source, const vector<string> &supported) {
	for (auto &ca : RegressionContextualAliases()) {
		if (source != ca.source) {
			continue;
		}
		for (auto target : ca.targets) {
			if (Contains(supported, target)) {
				return target;
			}
		}
	}
	return string();
}

static string DescribeSupportedRegressionKeys(const vector<string> &supported) {
	vector<string> parts;
	for (auto &canonical : supported) {
		auto def = FindOptionKey(RegressionKeyTable(), canonical);
		vector<string> aliases;
		if (def) {
			aliases.assign(def->aliases.begin(), def->aliases.end());
		}
		for (auto &ca : RegressionContextualAliases()) {
			if (!Contains(supported, ca.source) && ResolveContextualAlias(ca.source, supported) == canonical) {
				aliases.push_back(ca.source);
			}
		}
		parts.push_back(DescribeOptionKey(canonical, aliases));
	}
	if (parts.empty()) {
		return "(none)";
	}
	return StringUtil::Join(parts, ", ");
}

RegressionMapOptions RegressionMapOptions::ParseFromValue(const Value &map_value, const string &function_name,
                                                          const vector<string> &supported_keys) {
#ifdef DEBUG
	for (auto &k : supported_keys) {
		D_ASSERT(FindOptionKey(RegressionKeyTable(), k) != nullptr);
	}
#endif
	RegressionMapOptions result;
	// (contextual source, raw value) pairs resolved after the main pass.
	vector<std::pair<string, Value>> contextual;

	VisitOptionEntries(map_value, [&](const string &user_key, const Value &val) {
		auto def = FindOptionKey(RegressionKeyTable(), user_key);
		if (!def) {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: %s", function_name, user_key,
			                            DescribeSupportedRegressionKeys(supported_keys));
		}
		const string key = def->canonical;
		current_option_key = user_key;
		if (!Contains(supported_keys, key)) {
			if (!ResolveContextualAlias(key, supported_keys).empty()) {
				contextual.emplace_back(key, val);
				return;
			}
			throw InvalidInputException("%s: unsupported option '%s'. Supported options: %s", function_name,
			                            user_key, DescribeSupportedRegressionKeys(supported_keys));
		}

		if (key == "fit_intercept") {
			result.fit_intercept = ExtractBool(val);
		} else if (key == "compute_inference") {
			result.compute_inference = ExtractBool(val);
		} else if (key == "compute_loo") {
			result.compute_loo = ExtractBool(val);
		} else if (key == "confidence_level") {
			result.confidence_level = ExtractConfidenceLevel(val, function_name);
		} else if (key == "alpha") {
			result.alpha = ExtractDouble(val);
		} else if (key == "lambda") {
			result.lambda = ExtractDouble(val);
		} else if (key == "l1_ratio") {
			result.l1_ratio = ExtractDouble(val);
		} else if (key == "max_iterations") {
			result.max_iterations = ExtractUInt32(val);
		} else if (key == "tolerance") {
			result.tolerance = ExtractDouble(val);
		} else if (key == "alpha_1") {
			result.alpha_1 = ExtractDouble(val);
		} else if (key == "alpha_2") {
			result.alpha_2 = ExtractDouble(val);
		} else if (key == "lambda_1") {
			result.lambda_1 = ExtractDouble(val);
		} else if (key == "lambda_2") {
			result.lambda_2 = ExtractDouble(val);
		} else if (key == "threshold_lambda") {
			result.threshold_lambda = ExtractDouble(val);
		} else if (key == "c") {
			result.pa_c = ExtractDouble(val);
		} else if (key == "variant") {
			if (!val.IsNull()) {
				const string v = ToLower(val.ToString());
				if (v == "pa1" || v == "pa-i" || v == "pa_i" || v == "epsilon_insensitive") {
					result.pa_squared_loss = false;
				} else if (v == "pa2" || v == "pa-ii" || v == "pa_ii" || v == "squared_epsilon_insensitive") {
					result.pa_squared_loss = true;
				} else {
					throw InvalidInputException("%s: variant must be 'pa1' or 'pa2', got '%s'", function_name,
					                            val.ToString());
				}
			}
		} else if (key == "ic") {
			if (!val.IsNull()) {
				const string v = ToLower(val.ToString());
				if (v == "aic") {
					result.lm_dynamic_ic = 0;
				} else if (v == "aicc") {
					result.lm_dynamic_ic = 1;
				} else if (v == "bic") {
					result.lm_dynamic_ic = 2;
				} else {
					throw InvalidInputException("%s: ic must be 'aic', 'aicc' or 'bic', got '%s'", function_name,
					                            val.ToString());
				}
			}
		} else if (key == "span") {
			result.lowess_span = ExtractDouble(val);
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
		} else if (key == "random_state") {
			result.random_state = ExtractUInt64(val);
		} else if (key == "max_subpopulation") {
			result.max_subpopulation = ExtractUInt32(val);
		} else if (key == "n_subsamples") {
			result.n_subsamples = ExtractUInt32(val);
		} else if (key == "forgetting_factor") {
			result.forgetting_factor = ExtractDouble(val);
		} else if (key == "initial_p_diagonal") {
			result.initial_p_diagonal = ExtractDouble(val);
		} else if (key == "null_policy") {
			result.null_policy = ExtractNullPolicy(val);
		} else if (key == "link") {
			// Only reachable if a function lists 'link' itself; normally resolved
			// contextually below.
			result.link_value = val;
		} else if (key == "poisson_link") {
			result.poisson_link = ExtractPoissonLink(val);
		} else if (key == "binomial_link") {
			result.binomial_link = ExtractBinomialLink(val);
		} else if (key == "power") {
			result.tweedie_power = ExtractDouble(val);
		} else if (key == "distribution") {
			// Shared key. ALM and AFT have overlapping distribution vocabularies --
			// "lognormal" and "gamma" name a valid distribution in both -- so set
			// whichever matches, possibly both, and let each function read the
			// field it cares about.
			const string raw = val.IsNull() ? string() : val.ToString();
			bool matched = false;
			if (!raw.empty() && IsAftDistName(raw)) {
				result.aft_dist = ParseAftDist(raw);
				matched = true;
			}
			if (auto alm = ExtractAlmDistribution(val)) {
				result.distribution = alm;
				matched = true;
			}
			if (!matched && !raw.empty()) {
				throw InvalidInputException("Unknown distribution: '%s'", raw);
			}
		} else if (key == "loss") {
			result.loss = ExtractAlmLoss(val);
		} else if (key == "quantile") {
			result.quantile = ExtractDouble(val);
		} else if (key == "role_trim") {
			result.role_trim = ExtractDouble(val);
		} else if (key == "lower_bound") {
			result.lower_bound = ExtractDouble(val);
		} else if (key == "upper_bound") {
			result.upper_bound = ExtractDouble(val);
		} else if (key == "intermittent_threshold") {
			result.intermittent_threshold = ExtractDouble(val);
		} else if (key == "outlier_method") {
			result.outlier_method = ExtractAidOutlierMethod(val);
		} else if (key == "n_components") {
			auto v = ExtractUInt32(val);
			if (v.has_value()) {
				result.n_components = static_cast<size_t>(v.value());
			}
		} else if (key == "tau") {
			result.tau = ExtractDouble(val);
		} else if (key == "increasing") {
			result.increasing = ExtractBool(val);
		} else if (key == "solver") {
			result.solver = ExtractSolverType(val);
		} else if (key == "hc_type") {
			result.hc_type = ExtractHcType(val);
		} else if (key == "alias_tolerance") {
			auto v = ExtractDouble(val);
			if (v.has_value() && !(std::isfinite(v.value()) && v.value() > 0.0)) {
				throw InvalidInputException("alias_tolerance must be a positive number, got %s", val.ToString());
			}
			result.alias_tolerance = v;
		} else if (key == "lambda_scaling") {
			result.lambda_scaling = ExtractLambdaScaling(val);
		} else if (key == "glm_lambda") {
			result.glm_lambda = ExtractDouble(val);
		} else if (key == "threshold") {
			result.threshold = ExtractDouble(val);
		} else if (key == "feature_names") {
			result.feature_names = ExtractStringList(val);
		} else if (key == "prior") {
			result.prior_value = val;
		} else if (key == "family") {
			result.glmm_family = ParseGlmmFamily(val);
		} else if (key == "reml") {
			result.reml = ExtractBool(val);
		} else if (key == "offset") {
			auto v = ExtractUInt32(val);
			if (v.has_value()) {
				result.offset_column = (idx_t)v.value();
			}
		} else if (key == "random") {
			result.random_slopes = ExtractIndexList(val);
		} else if (key == "groups") {
			result.group_columns = ExtractIndexList(val);
		} else if (key == "ddf") {
			result.glmm_ddf = ParseGlmmDdf(val);
		} else if (key == "tau_squared") {
			result.tau_squared = ExtractDouble(val);
		} else if (key == "tau_method") {
			const string m = ToLower(val.IsNull() ? string() : val.ToString());
			if (m == "dl" || m == "dersimonian_laird" || m == "dersimonian-laird") {
				result.tau_method = false;
			} else if (m == "none" || m == "pooled" || m == "complete") {
				result.tau_method = true;
			} else {
				throw InvalidInputException("Unknown tau_method '%s'. Expected 'dl' or 'none'.", m);
			}
		} else if (key == "theta") {
			result.nb_theta = ExtractDouble(val);
		} else if (key == "vcov") {
			result.vcov = ExtractVcovType(val);
		} else if (key == "method") {
			if (!val.IsNull()) {
				const string m = ToLower(val.ToString());
				if (m == "lar" || m == "lars") {
					result.lars_lasso = false;
				} else if (m == "lasso" || m == "lasso_lars" || m == "lassolars") {
					result.lars_lasso = true;
				} else {
					throw InvalidInputException("Invalid method: '%s'. Valid values are 'lar', 'lasso'", m);
				}
			}
		} else if (key == "n_nonzero_coefs") {
			if (!val.IsNull()) {
				auto v = val.GetValue<int64_t>();
				if (v < 0) {
					throw InvalidInputException("n_nonzero_coefs must be >= 0 (0 = unlimited), got %lld",
					                            (long long)v);
				}
				result.n_nonzero_coefs = v;
			}
		} else if (key == "standardize") {
			result.standardize = ExtractBool(val);
		} else if (key == "lambdas") {
			result.lambdas = ExtractDoubleList(val, "lambdas");
			for (auto v : *result.lambdas) {
				if (v < 0) {
					throw InvalidInputException("%s: lambdas must be >= 0, got %g", function_name, v);
				}
			}
		} else if (key == "n_lambda") {
			auto v = ExtractUInt32(val);
			if (v.has_value() && v.value() == 0) {
				throw InvalidInputException("%s: n_lambda must be >= 1", function_name);
			}
			result.n_lambda = v;
		} else if (key == "lambda_min_ratio") {
			result.lambda_min_ratio = ExtractDouble(val);
			if (result.lambda_min_ratio.has_value() &&
			    !(result.lambda_min_ratio.value() > 0.0 && result.lambda_min_ratio.value() < 1.0)) {
				throw InvalidInputException("%s: lambda_min_ratio must be in (0, 1), got %s", function_name,
				                            val.ToString());
			}
		} else if (key == "folds") {
			result.folds = ExtractUInt32(val);
		} else if (key == "cv_strategy" || key == "window" || key == "metric" || key == "rule") {
			if (!val.IsNull()) {
				const string v = ToLower(val.ToString());
				vector<string> allowed;
				if (key == "cv_strategy") {
					allowed = {"kfold", "fold_id", "group", "logo", "stratified", "time_series"};
				} else if (key == "window") {
					allowed = {"expanding", "sliding"};
				} else if (key == "metric") {
					allowed = {"mse", "mae"};
				} else {
					allowed = {"min", "1se"};
				}
				if (std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
					throw InvalidInputException("%s: %s must be one of '%s', got '%s'", function_name, key,
					                            StringUtil::Join(allowed, "', '"), val.ToString());
				}
				(key == "cv_strategy" ? result.cv_strategy
				 : key == "window"    ? result.window
				 : key == "metric"    ? result.cv_metric
				                      : result.lambda_rule) = v;
			}
		} else if (key == "repeats" || key == "test_size" || key == "max_train_size" || key == "n_bins" ||
		           key == "inner_folds") {
			auto v = ExtractUInt32(val);
			uint32_t min = key == "inner_folds" ? 2 : 1;
			if (v.has_value() && v.value() < min) {
				throw InvalidInputException("%s: %s must be >= %u, got %u", function_name, key, min, v.value());
			}
			(key == "repeats"          ? result.repeats
			 : key == "test_size"      ? result.test_size
			 : key == "max_train_size" ? result.max_train_size
			 : key == "n_bins"         ? result.n_bins
			                           : result.inner_folds) = v;
		} else if (key == "gap") {
			result.gap = ExtractUInt32(val);
		} else if (key == "taus") {
			result.taus = ExtractDoubleList(val, "taus");
		} else if (key == "se") {
			if (!val.IsNull()) {
				const string v = ToLower(val.ToString());
				if (v != "iid" && v != "nid" && v != "boot") {
					throw InvalidInputException("%s: se must be 'iid', 'nid' or 'boot', got '%s'", function_name,
					                            val.ToString());
				}
				result.se_type = v;
			}
		} else if (key == "n_boot") {
			auto v = ExtractUInt32(val);
			if (v.has_value() && v.value() < 2) {
				throw InvalidInputException("%s: n_boot must be >= 2", function_name);
			}
			result.n_boot = v;
		} else if (key == "ci_method") {
			if (!val.IsNull()) {
				const string v = ToLower(val.ToString());
				if (v == "exact") {
					result.ci_method = 1;
				} else if (v == "bootstrap") {
					result.ci_method = 2;
				} else if (v == "exact_or_none") {
					result.ci_method = 3;
				} else {
					throw InvalidInputException("%s: ci_method must be 'exact', 'bootstrap' or 'exact_or_none', got '%s'",
					                            function_name, val.ToString());
				}
			}
		} else {
			throw InternalException("RegressionMapOptions: unhandled option key '%s'", key);
		}
	});

	// Context-dependent aliases. An explicitly given canonical key wins over its
	// alias (e.g. 'glm_lambda' over 'lambda'/'alpha', 'poisson_link' over 'link').
	// Within the GLM penalty aliases 'lambda' wins over 'alpha'.
	std::optional<double> glm_from_lambda;
	std::optional<double> glm_from_alpha;
	for (auto &entry : contextual) {
		const string &source = entry.first;
		const Value &val = entry.second;
		const string target = ResolveContextualAlias(source, supported_keys);
		if (target == "poisson_link") {
			if (!result.poisson_link.has_value()) {
				result.poisson_link = ExtractPoissonLink(val);
			}
		} else if (target == "binomial_link") {
			if (!result.binomial_link.has_value()) {
				result.binomial_link = ExtractBinomialLink(val);
			}
		} else if (target == "glm_lambda") {
			(source == "lambda" ? glm_from_lambda : glm_from_alpha) = ExtractDouble(val);
		} else if (target == "forgetting_factor") {
			if (!result.forgetting_factor.has_value()) {
				result.forgetting_factor = ExtractDouble(val);
			}
		} else if (target == "quantile") {
			if (!result.quantile.has_value()) {
				result.quantile = ExtractDouble(val);
			}
		} else if (target == "tau") {
			if (!result.tau.has_value()) {
				result.tau = ExtractDouble(val);
			}
		}
	}
	if (!result.glm_lambda.has_value()) {
		result.glm_lambda = glm_from_lambda.has_value() ? glm_from_lambda : glm_from_alpha;
	}

	return result;
}

RegressionMapOptions RegressionMapOptions::ParseFromExpression(ClientContext &context, Expression &expr,
                                                               const string &function_name,
                                                               const vector<string> &supported_keys) {
	Value val = EvaluateConstantOptions(context, expr, function_name);
	return ParseFromValue(val, function_name, supported_keys);
}

Value EvaluateConstantOptions(ClientContext &context, Expression &expr, const string &function_name) {
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: options must be a constant expression (a MAP or STRUCT literal), not a "
		                            "value that varies per row",
		                            function_name);
	}
	return ExpressionExecutor::EvaluateScalar(context, expr);
}

double ResolveTostAlpha(const string &function_name, const std::optional<double> &alpha,
                        const std::optional<double> &confidence_level, double default_alpha) {
	if (alpha.has_value() && !(alpha.value() > 0.0 && alpha.value() < 0.5)) {
		throw InvalidInputException("%s: alpha must be strictly between 0 and 0.5, got %g", function_name,
		                            alpha.value());
	}
	if (alpha.has_value() && confidence_level.has_value()) {
		if (std::fabs(alpha.value() - (1.0 - confidence_level.value())) > 1e-12) {
			throw InvalidInputException("%s: alpha (%g) and confidence_level (%g) are inconsistent; for TOST "
			                            "alpha = 1 - confidence_level. Supply only one of them.",
			                            function_name, alpha.value(), confidence_level.value());
		}
	}
	if (alpha.has_value()) {
		return alpha.value();
	}
	if (confidence_level.has_value()) {
		return 1.0 - confidence_level.value();
	}
	return default_alpha;
}

// ============================================================================
// Statistical Test Option Parsers
// ============================================================================

// Shared driver: walks MAP or STRUCT entries, maps aliases to the canonical key
// and rejects keys outside `keys`.
template <typename T, typename Callback>
static T ParseTestOptions(const Value &map_value, const string &function_name, const vector<OptionKeyDef> &keys,
                          Callback callback) {
	T result;
	VisitOptionEntries(map_value, [&](const string &user_key, const Value &val) {
		auto def = FindOptionKey(keys, user_key);
		if (!def) {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: %s", function_name, user_key,
			                            DescribeAllKeys(keys));
		}
		const string canonical(def->canonical);
		current_option_key = user_key;
		if (canonical == "confidence_level") {
			ExtractConfidenceLevel(val, function_name); // validate with the function name in the message
		}
		callback(result, canonical, val);
	});
	return result;
}

static const OptionKeyDef KEY_ALTERNATIVE = {"alternative", {}};
static const OptionKeyDef KEY_CONFIDENCE = {"confidence_level", {"confidence"}};
static const OptionKeyDef KEY_SEED = {"seed", {"random_state"}};
static const OptionKeyDef KEY_PERMUTATIONS = {"n_permutations", {"permutations"}};

// Strict enum helper for small string-valued options.
static string ExtractChoice(const Value &val, const string &option, const vector<string> &choices) {
	string str = ToLower(val.ToString());
	for (auto &c : choices) {
		if (str == c) {
			return str;
		}
	}
	throw InvalidInputException("Invalid %s: '%s'. Valid values are '%s'", option, val.ToString(),
	                            StringUtil::Join(choices, "', '"));
}

TTestMapOptions TTestMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {
	    KEY_ALTERNATIVE, KEY_CONFIDENCE, {"kind", {"var_equal"}}, {"paired", {}}, {"mu", {}}};
	return ParseTestOptions<TTestMapOptions>(map_value, function_name, keys,
	                                         [](TTestMapOptions &result, const string &key, const Value &val) {
		                                         if (key == "alternative") {
			                                         result.alternative = ExtractAlternative(val);
		                                         } else if (key == "confidence_level") {
			                                         result.confidence_level = ExtractConfidenceLevel(val);
		                                         } else if (key == "kind") {
			                                         result.kind = ExtractTTestKind(val);
		                                         } else if (key == "paired") {
			                                         result.paired = ExtractBool(val);
		                                         } else if (key == "mu") {
			                                         result.mu = ExtractDouble(val);
		                                         }
	                                         });
}

PosthocMapOptions PosthocMapOptions::ParseFromValue(const Value &map_value, const string &function_name,
                                                    bool allow_p_adjust, bool allow_t_options, bool allow_confidence) {
	vector<OptionKeyDef> keys;
	if (allow_p_adjust) {
		keys.push_back({"p_adjust", {"p_adjust_method", "p.adjust.method"}});
	}
	if (allow_t_options) {
		keys.push_back({"pool_sd", {"pooled_sd", "pool.sd"}});
		keys.push_back(KEY_ALTERNATIVE);
	}
	if (allow_confidence) {
		keys.push_back({"confidence_level", {"confidence", "conf_level", "conf.level"}});
	}
	return ParseTestOptions<PosthocMapOptions>(map_value, function_name, keys,
	                                           [](PosthocMapOptions &result, const string &key, const Value &val) {
		                                           if (val.IsNull()) {
			                                           return;
		                                           }
		                                           if (key == "p_adjust") {
			                                           result.p_adjust = val.ToString();
		                                           } else if (key == "pool_sd") {
			                                           result.pool_sd = ExtractBool(val);
		                                           } else if (key == "alternative") {
			                                           result.alternative = ExtractAlternative(val);
		                                           } else if (key == "confidence_level") {
			                                           result.confidence_level = ExtractConfidenceLevel(val);
		                                           }
	                                           });
}

AnovaTableMapOptions AnovaTableMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {
	    {"fit_intercept", {"intercept"}}, {"type", {"anova_type"}}, {"terms", {"term_names"}}};
	return ParseTestOptions<AnovaTableMapOptions>(
	    map_value, function_name, keys, [&](AnovaTableMapOptions &result, const string &key, const Value &val) {
		    if (val.IsNull()) {
			    return;
		    }
		    if (key == "fit_intercept") {
			    result.fit_intercept = ExtractBool(val);
		    } else if (key == "type") {
			    string v = ToLower(val.ToString());
			    if (v == "i" || v == "1" || v == "sequential") {
				    result.anova_type = 1;
			    } else if (v == "ii" || v == "2" || v == "partial") {
				    result.anova_type = 2;
			    } else {
				    throw InvalidInputException("%s: invalid type '%s'. Valid values are 'I' (sequential) and 'II' "
				                                "(partial)",
				                                function_name, val.ToString());
			    }
		    } else if (key == "terms") {
			    if (val.type().id() != LogicalTypeId::LIST) {
				    throw InvalidInputException("%s: terms must be a LIST of VARCHAR labels, one per x column, got %s",
				                                function_name, val.type().ToString());
			    }
			    vector<string> labels;
			    for (auto &child : ListValue::GetChildren(val)) {
				    if (child.IsNull()) {
					    throw InvalidInputException("%s: terms must not contain NULL", function_name);
				    }
				    labels.push_back(child.ToString());
			    }
			    result.terms = std::move(labels);
		    }
	    });
}

MannWhitneyMapOptions MannWhitneyMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {
	    KEY_ALTERNATIVE, KEY_CONFIDENCE, {"continuity_correction", {"correction"}}, {"exact", {}}, {"mu", {}}};
	return ParseTestOptions<MannWhitneyMapOptions>(
	    map_value, function_name, keys, [](MannWhitneyMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    } else if (key == "continuity_correction") {
			    result.continuity_correction = ExtractBool(val);
		    } else if (key == "exact") {
			    result.exact = ExtractBool(val);
		    } else if (key == "mu") {
			    result.mu = ExtractDouble(val);
		    }
	    });
}

WilcoxonMapOptions WilcoxonMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_ALTERNATIVE, KEY_CONFIDENCE,
	                                          {"continuity_correction", {"correction"}}};
	return ParseTestOptions<WilcoxonMapOptions>(
	    map_value, function_name, keys, [](WilcoxonMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    } else if (key == "continuity_correction") {
			    result.continuity_correction = ExtractBool(val);
		    }
	    });
}

BrunnerMunzelMapOptions BrunnerMunzelMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_ALTERNATIVE, KEY_CONFIDENCE};
	return ParseTestOptions<BrunnerMunzelMapOptions>(
	    map_value, function_name, keys, [](BrunnerMunzelMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    }
	    });
}

CorrelationMapOptions CorrelationMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_CONFIDENCE};
	return ParseTestOptions<CorrelationMapOptions>(
	    map_value, function_name, keys, [](CorrelationMapOptions &result, const string &key, const Value &val) {
		    if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    }
	    });
}

KendallMapOptions KendallMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_CONFIDENCE, {"variant", {"tau_type", "type"}}};
	return ParseTestOptions<KendallMapOptions>(map_value, function_name, keys,
	                                           [](KendallMapOptions &result, const string &key, const Value &val) {
		                                           if (key == "confidence_level") {
			                                           result.confidence_level = ExtractConfidenceLevel(val);
		                                           } else if (key == "variant") {
			                                           result.variant = ExtractKendallType(val);
		                                           }
	                                           });
}

ChiSquareMapOptions ChiSquareMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"continuity_correction", {"correction", "yates"}}};
	return ParseTestOptions<ChiSquareMapOptions>(
	    map_value, function_name, keys, [](ChiSquareMapOptions &result, const string &key, const Value &val) {
		    if (key == "continuity_correction") {
			    result.continuity_correction = ExtractBool(val);
		    }
	    });
}

FisherExactMapOptions FisherExactMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_ALTERNATIVE, KEY_CONFIDENCE};
	return ParseTestOptions<FisherExactMapOptions>(
	    map_value, function_name, keys, [](FisherExactMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    }
	    });
}

EnergyDistanceMapOptions EnergyDistanceMapOptions::ParseFromValue(const Value &map_value,
                                                                  const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_PERMUTATIONS, KEY_SEED};
	return ParseTestOptions<EnergyDistanceMapOptions>(
	    map_value, function_name, keys, [](EnergyDistanceMapOptions &result, const string &key, const Value &val) {
		    if (key == "n_permutations") {
			    result.n_permutations = ExtractUInt32(val);
		    } else if (key == "seed") {
			    result.seed = ExtractUInt64(val);
		    }
	    });
}

MmdMapOptions MmdMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	// 'bandwidth'/'sigma' used to be accepted and silently ignored: the core
	// always uses the median heuristic. They are now rejected.
	static const vector<OptionKeyDef> keys = {KEY_PERMUTATIONS, KEY_SEED};
	return ParseTestOptions<MmdMapOptions>(map_value, function_name, keys,
	                                       [](MmdMapOptions &result, const string &key, const Value &val) {
		                                       if (key == "n_permutations") {
			                                       result.n_permutations = ExtractUInt32(val);
		                                       } else if (key == "seed") {
			                                       result.seed = ExtractUInt64(val);
		                                       }
	                                       });
}

// TOST keys shared by the three TOST aggregates.
static const OptionKeyDef KEY_TOST_ALPHA = {"alpha", {}};
static const OptionKeyDef KEY_TOST_DELTA = {"delta", {"equivalence_bound"}};
static const OptionKeyDef KEY_TOST_LOWER = {"bound_lower", {"lower", "low"}};
static const OptionKeyDef KEY_TOST_UPPER = {"bound_upper", {"upper", "high"}};

static std::optional<double> ExtractTostAlpha(const Value &val) {
	auto v = ExtractDouble(val);
	if (v.has_value() && !(v.value() > 0.0 && v.value() < 0.5)) {
		throw InvalidInputException("alpha must be strictly between 0 and 0.5, got %s", val.ToString());
	}
	return v;
}

TostMapOptions TostMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	// 'alternative', 'paired' and 'mu' were accepted and ignored before; TOST is
	// two one-sided tests by construction and this aggregate is two-sample.
	static const vector<OptionKeyDef> keys = {KEY_CONFIDENCE, KEY_TOST_ALPHA,      {"kind", {"var_equal"}},
	                                          KEY_TOST_DELTA, KEY_TOST_LOWER, KEY_TOST_UPPER};
	return ParseTestOptions<TostMapOptions>(map_value, function_name, keys,
	                                        [](TostMapOptions &result, const string &key, const Value &val) {
		                                        if (key == "confidence_level") {
			                                        result.confidence_level = ExtractConfidenceLevel(val);
		                                        } else if (key == "alpha") {
			                                        result.alpha = ExtractTostAlpha(val);
		                                        } else if (key == "kind") {
			                                        result.kind = ExtractTTestKind(val);
		                                        } else if (key == "delta") {
			                                        result.delta = ExtractDouble(val);
		                                        } else if (key == "bound_lower") {
			                                        result.bound_lower = ExtractDouble(val);
		                                        } else if (key == "bound_upper") {
			                                        result.bound_upper = ExtractDouble(val);
		                                        }
	                                        });
}

TostPairedMapOptions TostPairedMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_CONFIDENCE, KEY_TOST_ALPHA, KEY_TOST_DELTA, KEY_TOST_LOWER,
	                                          KEY_TOST_UPPER};
	return ParseTestOptions<TostPairedMapOptions>(
	    map_value, function_name, keys, [](TostPairedMapOptions &result, const string &key, const Value &val) {
		    if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    } else if (key == "alpha") {
			    result.alpha = ExtractTostAlpha(val);
		    } else if (key == "delta") {
			    result.delta = ExtractDouble(val);
		    } else if (key == "bound_lower") {
			    result.bound_lower = ExtractDouble(val);
		    } else if (key == "bound_upper") {
			    result.bound_upper = ExtractDouble(val);
		    }
	    });
}

TostCorrelationMapOptions TostCorrelationMapOptions::ParseFromValue(const Value &map_value,
                                                                    const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_CONFIDENCE, KEY_TOST_ALPHA,      KEY_TOST_DELTA,
	                                          KEY_TOST_LOWER, KEY_TOST_UPPER,      {"rho_null", {"rho"}},
	                                          {"method", {}}};
	return ParseTestOptions<TostCorrelationMapOptions>(
	    map_value, function_name, keys, [](TostCorrelationMapOptions &result, const string &key, const Value &val) {
		    if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    } else if (key == "alpha") {
			    result.alpha = ExtractTostAlpha(val);
		    } else if (key == "delta") {
			    result.delta = ExtractDouble(val);
		    } else if (key == "bound_lower") {
			    result.bound_lower = ExtractDouble(val);
		    } else if (key == "bound_upper") {
			    result.bound_upper = ExtractDouble(val);
		    } else if (key == "rho_null") {
			    result.rho_null = ExtractDouble(val);
		    } else if (key == "method" && !val.IsNull()) {
			    result.spearman = ExtractChoice(val, "method", {"pearson", "spearman"}) == "spearman";
		    }
	    });
}

YuenMapOptions YuenMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_ALTERNATIVE, KEY_CONFIDENCE, {"trim", {"trim_proportion"}}};
	return ParseTestOptions<YuenMapOptions>(map_value, function_name, keys,
	                                        [](YuenMapOptions &result, const string &key, const Value &val) {
		                                        if (key == "alternative") {
			                                        result.alternative = ExtractAlternative(val);
		                                        } else if (key == "confidence_level") {
			                                        result.confidence_level = ExtractConfidenceLevel(val);
		                                        } else if (key == "trim") {
			                                        result.trim = ExtractDouble(val);
		                                        }
	                                        });
}

PermutationMapOptions PermutationMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_ALTERNATIVE, KEY_PERMUTATIONS, KEY_SEED};
	return ParseTestOptions<PermutationMapOptions>(
	    map_value, function_name, keys, [](PermutationMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "n_permutations") {
			    result.n_permutations = ExtractUInt32(val);
		    } else if (key == "seed") {
			    result.seed = ExtractUInt64(val);
		    }
	    });
}

DistanceCorMapOptions DistanceCorMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_PERMUTATIONS, KEY_SEED};
	return ParseTestOptions<DistanceCorMapOptions>(
	    map_value, function_name, keys, [](DistanceCorMapOptions &result, const string &key, const Value &val) {
		    if (key == "n_permutations") {
			    result.n_permutations = ExtractUInt32(val);
		    } else if (key == "seed") {
			    result.seed = ExtractUInt64(val);
		    }
	    });
}

DieboldMarianoMapOptions DieboldMarianoMapOptions::ParseFromValue(const Value &map_value,
                                                                  const string &function_name) {
	static const vector<OptionKeyDef> keys = {
	    {"loss", {}}, {"var_estimator", {}}, {"horizon", {"h"}}, KEY_ALTERNATIVE};
	return ParseTestOptions<DieboldMarianoMapOptions>(
	    map_value, function_name, keys, [](DieboldMarianoMapOptions &result, const string &key, const Value &val) {
		    if (val.IsNull()) {
			    return;
		    }
		    if (key == "loss") {
			    result.absolute_loss = ExtractChoice(val, "loss", {"squared", "absolute"}) == "absolute";
		    } else if (key == "var_estimator") {
			    result.bartlett = ExtractChoice(val, "var_estimator", {"acf", "bartlett"}) == "bartlett";
		    } else if (key == "horizon") {
			    result.horizon = ExtractUInt32(val);
		    } else if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    }
	    });
}

ClarkWestMapOptions ClarkWestMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"horizon", {"h"}}};
	return ParseTestOptions<ClarkWestMapOptions>(
	    map_value, function_name, keys, [](ClarkWestMapOptions &result, const string &key, const Value &val) {
		    if (key == "horizon") {
			    result.horizon = ExtractUInt32(val);
		    }
	    });
}

ProportionMapOptions ProportionMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"p0", {"p"}}, KEY_ALTERNATIVE, KEY_CONFIDENCE};
	return ParseTestOptions<ProportionMapOptions>(
	    map_value, function_name, keys, [](ProportionMapOptions &result, const string &key, const Value &val) {
		    if (key == "p0") {
			    result.p0 = ExtractDouble(val);
		    } else if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    }
	    });
}

PropTestTwoMapOptions PropTestTwoMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {KEY_ALTERNATIVE, {"correction", {"continuity_correction"}}, KEY_CONFIDENCE};
	return ParseTestOptions<PropTestTwoMapOptions>(
	    map_value, function_name, keys, [](PropTestTwoMapOptions &result, const string &key, const Value &val) {
		    if (key == "alternative") {
			    result.alternative = ExtractAlternative(val);
		    } else if (key == "correction") {
			    result.correction = ExtractBool(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    }
	    });
}

IccMapOptions IccMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"type", {}}};
	return ParseTestOptions<IccMapOptions>(map_value, function_name, keys,
	                                       [](IccMapOptions &result, const string &key, const Value &val) {
		                                       if (key == "type" && !val.IsNull()) {
			                                       result.average =
			                                           ExtractChoice(val, "type", {"single", "average"}) == "average";
		                                       }
	                                       });
}

McNemarMapOptions McNemarMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"correction", {"continuity_correction"}}, {"exact", {}}};
	return ParseTestOptions<McNemarMapOptions>(map_value, function_name, keys,
	                                           [](McNemarMapOptions &result, const string &key, const Value &val) {
		                                           if (key == "correction") {
			                                           result.correction = ExtractBool(val);
		                                           } else if (key == "exact") {
			                                           result.exact = ExtractBool(val);
		                                           }
	                                           });
}

CohenKappaMapOptions CohenKappaMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"weighted", {}}};
	return ParseTestOptions<CohenKappaMapOptions>(
	    map_value, function_name, keys, [](CohenKappaMapOptions &result, const string &key, const Value &val) {
		    if (key == "weighted") {
			    result.weighted = ExtractBool(val);
		    }
	    });
}

MediationMapOptions MediationMapOptions::ParseFromValue(const Value &map_value, const string &function_name) {
	static const vector<OptionKeyDef> keys = {{"method", {}},
	                                          {"ci_type", {"boot_ci_type", "boot.ci.type"}},
	                                          {"n_bootstrap", {"n_sims", "sims", "n_boot"}},
	                                          KEY_SEED,
	                                          {"confidence_level", {"confidence", "conf_level", "conf.level"}},
	                                          {"treat_value", {"treat.value"}},
	                                          {"control_value", {"control.value"}}};
	return ParseTestOptions<MediationMapOptions>(
	    map_value, function_name, keys, [](MediationMapOptions &result, const string &key, const Value &val) {
		    if (val.IsNull()) {
			    return;
		    }
		    if (key == "method") {
			    result.method = ExtractChoice(val, "method", {"bootstrap", "sobel", "quasi_bayesian"});
		    } else if (key == "ci_type") {
			    result.ci_type = ExtractChoice(val, "ci_type", {"percentile", "bca"});
		    } else if (key == "n_bootstrap") {
			    result.n_bootstrap = ExtractUInt32(val);
		    } else if (key == "seed") {
			    result.seed = ExtractUInt64(val);
		    } else if (key == "confidence_level") {
			    result.confidence_level = ExtractConfidenceLevel(val);
		    } else if (key == "treat_value") {
			    result.treat_value = ExtractDouble(val);
		    } else if (key == "control_value") {
			    result.control_value = ExtractDouble(val);
		    }
	    });
}

} // namespace duckdb
