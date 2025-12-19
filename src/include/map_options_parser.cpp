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

// Helper to extract AidOutlierMethod from Value
static std::optional<AidOutlierMethod> ExtractAidOutlierMethod(const Value &val) {
    if (val.IsNull()) {
        return std::nullopt;
    }
    string str = ToLower(StringValue::Get(val));
    if (str == "zscore" || str == "z_score" || str == "z-score") return AidOutlierMethod::ZSCORE;
    if (str == "iqr") return AidOutlierMethod::IQR;
    throw InvalidInputException("Invalid outlier_method: '%s'. Valid values are 'zscore', 'iqr'", str);
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
    if (str == "tau_a" || str == "taua" || str == "a") return KendallType::TAU_A;
    if (str == "tau_b" || str == "taub" || str == "b") return KendallType::TAU_B;
    if (str == "tau_c" || str == "tauc" || str == "c") return KendallType::TAU_C;
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
        if (str == "student" || str == "equal") return TTestKind::STUDENT;
        if (str == "welch" || str == "unequal") return TTestKind::WELCH;
        throw InvalidInputException("Invalid t-test kind: '%s'. Valid values are 'student', 'welch'", str);
    }
    default:
        throw InvalidInputException("Cannot convert value of type %s to t-test kind",
                                    val.type().ToString());
    }
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
            // AID options
            else if (key == "intermittent_threshold") {
                result.intermittent_threshold = ExtractDouble(val);
            } else if (key == "outlier_method") {
                result.outlier_method = ExtractAidOutlierMethod(val);
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
            // AID options
            else if (key == "intermittent_threshold") {
                result.intermittent_threshold = ExtractDouble(val);
            } else if (key == "outlier_method") {
                result.outlier_method = ExtractAidOutlierMethod(val);
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

// ============================================================================
// Statistical Test Option Parsers
// ============================================================================

// Generic helper template for extracting options from MAP/STRUCT
template<typename T, typename Callback>
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
        throw InvalidInputException("Expected MAP or STRUCT type for options, got %s",
                                    map_value.type().ToString());
    }

    return result;
}

TTestMapOptions TTestMapOptions::ParseFromValue(const Value &map_value) {
    return ParseTestOptions<TTestMapOptions>(map_value, [](TTestMapOptions &result, const string &key, const Value &val) {
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
    return ParseTestOptions<MannWhitneyMapOptions>(map_value, [](MannWhitneyMapOptions &result, const string &key, const Value &val) {
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
    return ParseTestOptions<WilcoxonMapOptions>(map_value, [](WilcoxonMapOptions &result, const string &key, const Value &val) {
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
    return ParseTestOptions<BrunnerMunzelMapOptions>(map_value, [](BrunnerMunzelMapOptions &result, const string &key, const Value &val) {
        if (key == "alternative") {
            result.alternative = ExtractAlternative(val);
        } else if (key == "confidence_level" || key == "confidence") {
            result.confidence_level = ExtractDouble(val);
        }
    });
}

CorrelationMapOptions CorrelationMapOptions::ParseFromValue(const Value &map_value) {
    return ParseTestOptions<CorrelationMapOptions>(map_value, [](CorrelationMapOptions &result, const string &key, const Value &val) {
        if (key == "confidence_level" || key == "confidence") {
            result.confidence_level = ExtractDouble(val);
        }
    });
}

KendallMapOptions KendallMapOptions::ParseFromValue(const Value &map_value) {
    return ParseTestOptions<KendallMapOptions>(map_value, [](KendallMapOptions &result, const string &key, const Value &val) {
        if (key == "confidence_level" || key == "confidence") {
            result.confidence_level = ExtractDouble(val);
        } else if (key == "variant" || key == "tau_type" || key == "type") {
            result.variant = ExtractKendallType(val);
        }
    });
}

ChiSquareMapOptions ChiSquareMapOptions::ParseFromValue(const Value &map_value) {
    return ParseTestOptions<ChiSquareMapOptions>(map_value, [](ChiSquareMapOptions &result, const string &key, const Value &val) {
        if (key == "continuity_correction" || key == "correction" || key == "yates") {
            result.continuity_correction = ExtractBool(val);
        }
    });
}

FisherExactMapOptions FisherExactMapOptions::ParseFromValue(const Value &map_value) {
    return ParseTestOptions<FisherExactMapOptions>(map_value, [](FisherExactMapOptions &result, const string &key, const Value &val) {
        if (key == "alternative") {
            result.alternative = ExtractAlternative(val);
        }
    });
}

EnergyDistanceMapOptions EnergyDistanceMapOptions::ParseFromValue(const Value &map_value) {
    return ParseTestOptions<EnergyDistanceMapOptions>(map_value, [](EnergyDistanceMapOptions &result, const string &key, const Value &val) {
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
    return ParseTestOptions<PermutationMapOptions>(map_value, [](PermutationMapOptions &result, const string &key, const Value &val) {
        if (key == "alternative") {
            result.alternative = ExtractAlternative(val);
        } else if (key == "n_permutations" || key == "permutations") {
            result.n_permutations = ExtractUInt32(val);
        }
    });
}

} // namespace duckdb
