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
