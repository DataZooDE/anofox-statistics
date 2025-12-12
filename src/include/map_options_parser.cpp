#include "map_options_parser.hpp"
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
        throw InvalidInputException("Cannot convert value of type %s to boolean",
                                    val.type().ToString());
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
    if (str == "normal") return AlmDistribution::NORMAL;
    if (str == "laplace") return AlmDistribution::LAPLACE;
    if (str == "student_t" || str == "studentt") return AlmDistribution::STUDENT_T;
    if (str == "logistic") return AlmDistribution::LOGISTIC;
    if (str == "asymmetric_laplace" || str == "asymmetriclaplace") return AlmDistribution::ASYMMETRIC_LAPLACE;
    if (str == "generalised_normal" || str == "generalisednormal") return AlmDistribution::GENERALISED_NORMAL;
    if (str == "s") return AlmDistribution::S;
    if (str == "log_normal" || str == "lognormal") return AlmDistribution::LOG_NORMAL;
    if (str == "log_laplace" || str == "loglaplace") return AlmDistribution::LOG_LAPLACE;
    if (str == "log_s" || str == "logs") return AlmDistribution::LOG_S;
    if (str == "log_generalised_normal" || str == "loggeneralisednormal") return AlmDistribution::LOG_GENERALISED_NORMAL;
    if (str == "folded_normal" || str == "foldednormal") return AlmDistribution::FOLDED_NORMAL;
    if (str == "rectified_normal" || str == "rectifiednormal") return AlmDistribution::RECTIFIED_NORMAL;
    if (str == "box_cox_normal" || str == "boxcoxnormal") return AlmDistribution::BOX_COX_NORMAL;
    if (str == "gamma") return AlmDistribution::GAMMA;
    if (str == "inverse_gaussian" || str == "inversegaussian") return AlmDistribution::INVERSE_GAUSSIAN;
    if (str == "exponential") return AlmDistribution::EXPONENTIAL;
    if (str == "beta") return AlmDistribution::BETA;
    if (str == "logit_normal" || str == "logitnormal") return AlmDistribution::LOGIT_NORMAL;
    if (str == "poisson") return AlmDistribution::POISSON;
    if (str == "negative_binomial" || str == "negativebinomial" || str == "negbinomial") return AlmDistribution::NEGATIVE_BINOMIAL;
    if (str == "binomial") return AlmDistribution::BINOMIAL;
    if (str == "geometric") return AlmDistribution::GEOMETRIC;
    if (str == "cumulative_logistic" || str == "cumulativelogistic") return AlmDistribution::CUMULATIVE_LOGISTIC;
    if (str == "cumulative_normal" || str == "cumulativenormal") return AlmDistribution::CUMULATIVE_NORMAL;
    throw InvalidInputException("Invalid ALM distribution: '%s'", str);
}

// Helper to extract AlmLoss from Value
static std::optional<AlmLoss> ExtractAlmLoss(const Value &val) {
    if (val.IsNull()) {
        return std::nullopt;
    }
    string str = ToLower(StringValue::Get(val));
    if (str == "likelihood") return AlmLoss::LIKELIHOOD;
    if (str == "mse") return AlmLoss::MSE;
    if (str == "mae") return AlmLoss::MAE;
    if (str == "ham") return AlmLoss::HAM;
    if (str == "role") return AlmLoss::ROLE;
    throw InvalidInputException("Invalid ALM loss: '%s'. Valid values are 'likelihood', 'mse', 'mae', 'ham', 'role'", str);
}

RegressionMapOptions RegressionMapOptions::ParseFromValue(const Value &map_value) {
    RegressionMapOptions result;

    if (map_value.IsNull()) {
        return result;
    }

    // Handle MAP type
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
            // Unknown keys are silently ignored for forward compatibility
        }
    }
    // Handle STRUCT type (DuckDB sometimes represents {'key': value} as STRUCT)
    else if (map_value.type().id() == LogicalTypeId::STRUCT) {
        auto &struct_type = map_value.type();
        auto &children = StructValue::GetChildren(map_value);
        auto &child_types = StructType::GetChildTypes(struct_type);

        for (idx_t i = 0; i < child_types.size(); i++) {
            string key = ToLower(child_types[i].first);
            const Value &val = children[i];

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
        }
    } else {
        throw InvalidInputException("Expected MAP or STRUCT type for options, got %s",
                                    map_value.type().ToString());
    }

    return result;
}

RegressionMapOptions RegressionMapOptions::ParseFromExpression(ClientContext &context, Expression &expr) {
    if (!expr.IsFoldable()) {
        throw InvalidInputException("Options parameter must be a constant expression");
    }
    Value val = ExpressionExecutor::EvaluateScalar(context, expr);
    return ParseFromValue(val);
}

} // namespace duckdb
