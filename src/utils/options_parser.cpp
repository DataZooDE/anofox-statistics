#include "options_parser.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {
namespace anofox_statistics {

RegressionOptions RegressionOptions::ParseFromMap(const Value &options_map) {
	RegressionOptions opts;

	// Return defaults if no options provided or NULL
	if (options_map.IsNull()) {
		return opts;
	}

	// Handle STRUCT type (from {'key': value} syntax)
	if (options_map.type().id() == LogicalTypeId::STRUCT) {
		auto &struct_children = StructValue::GetChildren(options_map);
		for (size_t i = 0; i < struct_children.size(); i++) {
			auto &key = StructType::GetChildName(options_map.type(), i);
			auto &val_value = struct_children[i];

			// Parse based on key
			if (key == "intercept") {
				opts.intercept = val_value.GetValue<bool>();
			} else if (key == "full_output") {
				opts.full_output = val_value.GetValue<bool>();
			} else if (key == "lambda") {
				opts.lambda = val_value.GetValue<double>();
			} else if (key == "alpha") {
				opts.alpha = val_value.GetValue<double>();
			} else if (key == "forgetting_factor") {
				opts.forgetting_factor = val_value.GetValue<double>();
			} else if (key == "window_size") {
				// Handle both BIGINT and INTEGER
				if (val_value.type().id() == LogicalTypeId::BIGINT) {
					opts.window_size = static_cast<idx_t>(val_value.GetValue<int64_t>());
				} else if (val_value.type().id() == LogicalTypeId::INTEGER) {
					opts.window_size = static_cast<idx_t>(val_value.GetValue<int32_t>());
				} else {
					throw InvalidInputException("Option 'window_size' must be an integer type");
				}
			} else if (key == "min_periods") {
				// Handle both BIGINT and INTEGER
				if (val_value.type().id() == LogicalTypeId::BIGINT) {
					opts.min_periods = static_cast<idx_t>(val_value.GetValue<int64_t>());
				} else if (val_value.type().id() == LogicalTypeId::INTEGER) {
					opts.min_periods = static_cast<idx_t>(val_value.GetValue<int32_t>());
				} else {
					throw InvalidInputException("Option 'min_periods' must be an integer type");
				}
			} else if (key == "confidence_level") {
				opts.confidence_level = val_value.GetValue<double>();
			} else if (key == "robust_se") {
				opts.robust_se = val_value.GetValue<bool>();
			} else if (key == "max_iterations") {
				// Handle both BIGINT and INTEGER
				if (val_value.type().id() == LogicalTypeId::BIGINT) {
					opts.max_iterations = static_cast<idx_t>(val_value.GetValue<int64_t>());
				} else if (val_value.type().id() == LogicalTypeId::INTEGER) {
					opts.max_iterations = static_cast<idx_t>(val_value.GetValue<int32_t>());
				} else {
					throw InvalidInputException("Option 'max_iterations' must be an integer type");
				}
			} else if (key == "tolerance") {
				opts.tolerance = val_value.GetValue<double>();
			} else if (key == "solver") {
				opts.solver = val_value.GetValue<string>();
			} else if (key == "compute_per_group") {
				opts.compute_per_group = val_value.GetValue<bool>();
			} else if (key == "fit_predict_mode") {
				opts.fit_predict_mode = val_value.GetValue<string>();
			} else {
				throw InvalidInputException(
				    "Unknown option: '%s'. Valid options are: intercept, full_output, lambda, alpha, "
				    "forgetting_factor, window_size, min_periods, confidence_level, robust_se, "
				    "max_iterations, tolerance, solver, compute_per_group, fit_predict_mode",
				    key);
			}
		}
		return opts;
	}

	// Handle MAP type (from MAP{'key': value} syntax) - for backwards compatibility
	if (options_map.type().id() == LogicalTypeId::MAP) {
		auto &map_children = MapValue::GetChildren(options_map);

		for (auto &entry : map_children) {
			// Each entry is a STRUCT with two fields: key and value
			auto &struct_children = StructValue::GetChildren(entry);
			D_ASSERT(struct_children.size() == 2);

			auto &key_value = struct_children[0];
			auto &val_value = struct_children[1];

			// Extract key as string
			string key = key_value.GetValue<string>();

			// Parse based on key
			if (key == "intercept") {
				opts.intercept = val_value.GetValue<bool>();
			} else if (key == "full_output") {
				opts.full_output = val_value.GetValue<bool>();
			} else if (key == "lambda") {
				opts.lambda = val_value.GetValue<double>();
			} else if (key == "alpha") {
				opts.alpha = val_value.GetValue<double>();
			} else if (key == "forgetting_factor") {
				opts.forgetting_factor = val_value.GetValue<double>();
			} else if (key == "window_size") {
				// Handle both BIGINT and INTEGER
				if (val_value.type().id() == LogicalTypeId::BIGINT) {
					opts.window_size = static_cast<idx_t>(val_value.GetValue<int64_t>());
				} else if (val_value.type().id() == LogicalTypeId::INTEGER) {
					opts.window_size = static_cast<idx_t>(val_value.GetValue<int32_t>());
				} else {
					throw InvalidInputException("Option 'window_size' must be an integer type");
				}
			} else if (key == "min_periods") {
				// Handle both BIGINT and INTEGER
				if (val_value.type().id() == LogicalTypeId::BIGINT) {
					opts.min_periods = static_cast<idx_t>(val_value.GetValue<int64_t>());
				} else if (val_value.type().id() == LogicalTypeId::INTEGER) {
					opts.min_periods = static_cast<idx_t>(val_value.GetValue<int32_t>());
				} else {
					throw InvalidInputException("Option 'min_periods' must be an integer type");
				}
			} else if (key == "confidence_level") {
				opts.confidence_level = val_value.GetValue<double>();
			} else if (key == "robust_se") {
				opts.robust_se = val_value.GetValue<bool>();
			} else if (key == "max_iterations") {
				// Handle both BIGINT and INTEGER
				if (val_value.type().id() == LogicalTypeId::BIGINT) {
					opts.max_iterations = static_cast<idx_t>(val_value.GetValue<int64_t>());
				} else if (val_value.type().id() == LogicalTypeId::INTEGER) {
					opts.max_iterations = static_cast<idx_t>(val_value.GetValue<int32_t>());
				} else {
					throw InvalidInputException("Option 'max_iterations' must be an integer type");
				}
			} else if (key == "tolerance") {
				opts.tolerance = val_value.GetValue<double>();
			} else if (key == "solver") {
				opts.solver = val_value.GetValue<string>();
			} else if (key == "compute_per_group") {
				opts.compute_per_group = val_value.GetValue<bool>();
			} else if (key == "fit_predict_mode") {
				opts.fit_predict_mode = val_value.GetValue<string>();
			} else {
				throw InvalidInputException(
				    "Unknown option: '%s'. Valid options are: intercept, full_output, lambda, alpha, "
				    "forgetting_factor, window_size, min_periods, confidence_level, robust_se, "
				    "max_iterations, tolerance, solver, compute_per_group, fit_predict_mode",
				    key);
			}
		}
	}

	return opts;
}

void RegressionOptions::Validate() const {
	// Validate lambda
	if (lambda < 0.0) {
		throw InvalidInputException("Option 'lambda' must be non-negative, got: %f", lambda);
	}

	// Validate alpha
	if (alpha < 0.0 || alpha > 1.0) {
		throw InvalidInputException("Option 'alpha' must be in [0, 1], got: %f", alpha);
	}

	// Validate forgetting factor
	if (forgetting_factor <= 0.0 || forgetting_factor > 1.0) {
		throw InvalidInputException("Option 'forgetting_factor' must be in (0, 1], got: %f", forgetting_factor);
	}

	// Validate confidence level
	if (confidence_level <= 0.0 || confidence_level >= 1.0) {
		throw InvalidInputException("Option 'confidence_level' must be in (0, 1), got: %f", confidence_level);
	}

	// Validate tolerance
	if (tolerance <= 0.0) {
		throw InvalidInputException("Option 'tolerance' must be positive, got: %f", tolerance);
	}

	// Validate max_iterations
	if (max_iterations == 0) {
		throw InvalidInputException("Option 'max_iterations' must be positive, got: %llu", max_iterations);
	}

	// Validate solver
	if (solver != "qr" && solver != "svd" && solver != "cholesky") {
		throw InvalidInputException("Option 'solver' must be one of 'qr', 'svd', 'cholesky', got: '%s'", solver);
	}

	// Validate fit_predict_mode
	if (fit_predict_mode != "expanding" && fit_predict_mode != "fixed") {
		throw InvalidInputException("Option 'fit_predict_mode' must be 'expanding' or 'fixed', got: '%s'",
		                            fit_predict_mode);
	}

	// Validate window size if set
	if (window_size > 0 && min_periods > 0) {
		throw InvalidInputException("Cannot specify both 'window_size' (for rolling) and 'min_periods' (for "
		                            "expanding) in the same options MAP");
	}
}

// Template specializations for GetMapValueOrDefault

template <>
bool GetMapValueOrDefault<bool>(const Value &map_param, const string &key, bool default_value) {
	if (map_param.IsNull()) {
		return default_value;
	}

	auto &map_children = MapValue::GetChildren(map_param);
	for (auto &entry : map_children) {
		auto &struct_children = StructValue::GetChildren(entry);
		auto &map_key = struct_children[0];
		auto &map_value = struct_children[1];

		if (map_key.GetValue<string>() == key) {
			return map_value.GetValue<bool>();
		}
	}
	return default_value;
}

template <>
double GetMapValueOrDefault<double>(const Value &map_param, const string &key, double default_value) {
	if (map_param.IsNull()) {
		return default_value;
	}

	auto &map_children = MapValue::GetChildren(map_param);
	for (auto &entry : map_children) {
		auto &struct_children = StructValue::GetChildren(entry);
		auto &map_key = struct_children[0];
		auto &map_value = struct_children[1];

		if (map_key.GetValue<string>() == key) {
			return map_value.GetValue<double>();
		}
	}
	return default_value;
}

template <>
string GetMapValueOrDefault<string>(const Value &map_param, const string &key, string default_value) {
	if (map_param.IsNull()) {
		return default_value;
	}

	auto &map_children = MapValue::GetChildren(map_param);
	for (auto &entry : map_children) {
		auto &struct_children = StructValue::GetChildren(entry);
		auto &map_key = struct_children[0];
		auto &map_value = struct_children[1];

		if (map_key.GetValue<string>() == key) {
			return map_value.GetValue<string>();
		}
	}
	return default_value;
}

template <>
idx_t GetMapValueOrDefault<idx_t>(const Value &map_param, const string &key, idx_t default_value) {
	if (map_param.IsNull()) {
		return default_value;
	}

	auto &map_children = MapValue::GetChildren(map_param);
	for (auto &entry : map_children) {
		auto &struct_children = StructValue::GetChildren(entry);
		auto &map_key = struct_children[0];
		auto &map_value = struct_children[1];

		if (map_key.GetValue<string>() == key) {
			// Handle both BIGINT and INTEGER
			if (map_value.type().id() == LogicalTypeId::BIGINT) {
				return static_cast<idx_t>(map_value.GetValue<int64_t>());
			} else if (map_value.type().id() == LogicalTypeId::INTEGER) {
				return static_cast<idx_t>(map_value.GetValue<int32_t>());
			}
			throw InvalidInputException("Expected integer type for key '%s'", key);
		}
	}
	return default_value;
}

} // namespace anofox_statistics
} // namespace duckdb
