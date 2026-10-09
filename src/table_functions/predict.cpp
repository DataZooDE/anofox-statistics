#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "telemetry.hpp"
#include "anofox_statistics_banner.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>

namespace duckdb {

// Extract list of doubles from a DuckDB list value; NULL elements become NaN
static vector<double> ExtractDoubleList(Vector &vec, idx_t row_idx) {
    auto list_data = ListVector::GetData(vec);
    auto &child = ListVector::GetEntry(vec);
    auto child_data = FlatVector::GetData<double>(child);
    auto &child_validity = FlatVector::Validity(child);

    vector<double> result;
    auto offset = list_data[row_idx].offset;
    auto length = list_data[row_idx].length;

    for (idx_t i = 0; i < length; i++) {
        result.push_back(child_validity.RowIsValid(offset + i) ? child_data[offset + i]
                                                             : std::numeric_limits<double>::quiet_NaN());
    }

    return result;
}

// Predict function: anofox_stats_predict(x, coefficients, intercept) -> LIST(DOUBLE)
static void PredictFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
static void PredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	PredictFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void PredictFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
    PostHogTelemetry::Instance().RecordFunctionCall("predict");
    auto &x_vec = args.data[0];         // LIST(LIST(DOUBLE)) - new feature data
    auto &coef_vec = args.data[1];      // LIST(DOUBLE) - coefficients
    auto &intercept_vec = args.data[2]; // DOUBLE - intercept (can be NULL)

    idx_t count = args.size();

    // Flatten intercept if needed
    UnifiedVectorFormat intercept_data;
    intercept_vec.ToUnifiedFormat(count, intercept_data);
    auto intercept_values = UnifiedVectorFormat::GetData<double>(intercept_data);

    // NULL inputs give a NULL row (the inputs are flattened below so row indexing is direct)
    x_vec.Flatten(count);
    coef_vec.Flatten(count);
    auto &result_validity = FlatVector::Validity(result);

    // Process each row
    for (idx_t row = 0; row < count; row++) {
        if (FlatVector::IsNull(x_vec, row) || FlatVector::IsNull(coef_vec, row)) {
            ListVector::GetData(result)[row] = {ListVector::GetListSize(result), 0};
            result_validity.SetInvalid(row);
            continue;
        }
        // Extract coefficients
        vector<double> coefficients = ExtractDoubleList(coef_vec, row);

        // Extract intercept (use NaN if NULL)
        auto intercept_idx = intercept_data.sel->get_index(row);
        double intercept =
            intercept_data.validity.RowIsValid(intercept_idx) ? intercept_values[intercept_idx] : std::nan("");

        // Extract x values (list of columns)
        auto x_list_data = ListVector::GetData(x_vec);
        auto &x_child = ListVector::GetEntry(x_vec);

        vector<vector<double>> x_cols;
        auto x_offset = x_list_data[row].offset;
        auto x_length = x_list_data[row].length;

        for (idx_t col = 0; col < x_length; col++) {
            x_cols.push_back(ExtractDoubleList(x_child, x_offset + col));
        }

        // Prepare FFI data
        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        // Call Rust FFI
        double *predictions = nullptr;
        size_t predictions_len = 0;
        AnofoxError error;

        bool success = anofox_predict(x_arrays.data(), x_arrays.size(), coefficients.data(), coefficients.size(),
                                      intercept, &predictions, &predictions_len, &error);

        if (!success) {
            ThrowUnlessDegenerate("predict", error);
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Build result list
        auto &result_child = ListVector::GetEntry(result);
        auto result_offset = ListVector::GetListSize(result);
        ListVector::Reserve(result, result_offset + predictions_len);
        ListVector::SetListSize(result, result_offset + predictions_len);
        auto result_data = FlatVector::GetData<double>(result_child);

        auto &result_child_validity = FlatVector::Validity(result_child);
        for (size_t i = 0; i < predictions_len; i++) {
            // A NULL (NaN) feature value gives a NULL prediction for that observation
            if (std::isfinite(predictions[i])) {
                result_data[result_offset + i] = predictions[i];
            } else {
                result_child_validity.SetInvalid(result_offset + i);
            }
        }
        ListVector::GetData(result)[row] = {result_offset, predictions_len};

        // Free predictions
        anofox_free_predictions(predictions);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}


//===--------------------------------------------------------------------===//
// Model-aware predict: predict(model STRUCT, x LIST(DOUBLE)[, options]) -> DOUBLE
//===--------------------------------------------------------------------===//
// `model` is the STRUCT returned by any *_fit_agg / *_fit function. Linear models use
// intercept + coefficients . x (NaN coefficients, i.e. aliased columns, contribute
// nothing). GLM structs carry a `link` field and the prediction is mapped back to the
// response scale unless {'type': 'link'} is given. Isotonic structs (x knots + fitted
// values) are evaluated by linear interpolation at x[1], clamped to the end knots.
namespace {

enum class ModelKind : uint8_t { LINEAR, ISOTONIC };

struct PredictModelBindData : public FunctionData {
	ModelKind kind = ModelKind::LINEAR;
	idx_t coef_idx = DConstants::INVALID_INDEX;
	idx_t intercept_idx = DConstants::INVALID_INDEX;
	idx_t link_idx = DConstants::INVALID_INDEX;
	idx_t knots_idx = DConstants::INVALID_INDEX;
	idx_t fitted_idx = DConstants::INVALID_INDEX;
	bool response_scale = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PredictModelBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PredictModelBindData>();
		return kind == o.kind && coef_idx == o.coef_idx && intercept_idx == o.intercept_idx &&
		       link_idx == o.link_idx && knots_idx == o.knots_idx && fitted_idx == o.fitted_idx &&
		       response_scale == o.response_scale;
	}
};

bool IsDoubleList(const LogicalType &t) {
	return t.id() == LogicalTypeId::LIST && ListType::GetChildType(t).id() == LogicalTypeId::DOUBLE;
}

unique_ptr<FunctionData> PredictModelBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->return_type.id() != LogicalTypeId::STRUCT) {
		if (arguments.size() == 3 && arguments[0]->return_type.id() == LogicalTypeId::LIST) {
			// Legacy column-layout call predict(x, coefficients, intercept) whose argument
			// types (e.g. DECIMAL literals) made the binder pick this overload.
			arguments[0] = BoundCastExpression::AddCastToType(
			    context, std::move(arguments[0]), LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)));
			arguments[1] = BoundCastExpression::AddCastToType(context, std::move(arguments[1]),
			                                                  LogicalType::LIST(LogicalType::DOUBLE));
			arguments[2] = BoundCastExpression::AddCastToType(context, std::move(arguments[2]), LogicalType::DOUBLE);
			bound_function.arguments = {LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
			                            LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE};
			bound_function.return_type = LogicalType::LIST(LogicalType::DOUBLE);
			bound_function.function = DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, PredictFunction);
			bound_function.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
			return nullptr;
		}
		throw InvalidInputException("predict(model, x): model must be the STRUCT returned by a *_fit_agg or *_fit "
		                            "function, got %s",
		                            arguments[0]->return_type.ToString());
	}
	// Numeric fields of a hand-written model literal (e.g. DECIMAL) are cast to DOUBLE.
	{
		auto &in_type = arguments[0]->return_type;
		child_list_t<LogicalType> cast_children;
		bool needs_cast = false;
		for (auto &child : StructType::GetChildTypes(in_type)) {
			auto t = child.second;
			if (t.IsNumeric() && t.id() != LogicalTypeId::DOUBLE) {
				t = LogicalType::DOUBLE;
				needs_cast = true;
			} else if (t.id() == LogicalTypeId::LIST && ListType::GetChildType(t).IsNumeric() &&
			           ListType::GetChildType(t).id() != LogicalTypeId::DOUBLE) {
				t = LogicalType::LIST(LogicalType::DOUBLE);
				needs_cast = true;
			}
			cast_children.push_back(make_pair(child.first, t));
		}
		if (needs_cast) {
			arguments[0] = BoundCastExpression::AddCastToType(context, std::move(arguments[0]),
			                                                  LogicalType::STRUCT(std::move(cast_children)));
		}
	}
	auto &model_type = arguments[0]->return_type;
	auto data = make_uniq<PredictModelBindData>();
	auto &children = StructType::GetChildTypes(model_type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = children[i].first;
		auto &type = children[i].second;
		if (name == "coefficients" && IsDoubleList(type)) {
			data->coef_idx = i;
		} else if (name == "intercept" && type.id() == LogicalTypeId::DOUBLE) {
			data->intercept_idx = i;
		} else if (name == "link" && type.id() == LogicalTypeId::VARCHAR) {
			data->link_idx = i;
		} else if (name == "x" && IsDoubleList(type)) {
			data->knots_idx = i;
		} else if (name == "fitted" && IsDoubleList(type)) {
			data->fitted_idx = i;
		}
	}
	if (data->coef_idx == DConstants::INVALID_INDEX) {
		if (data->knots_idx != DConstants::INVALID_INDEX && data->fitted_idx != DConstants::INVALID_INDEX) {
			data->kind = ModelKind::ISOTONIC;
		} else {
			throw InvalidInputException("predict(model, x): model has no 'coefficients' field (and is not an "
			                            "isotonic_fit_agg model); got %s",
			                            model_type.ToString());
		}
	}
	if (arguments.size() >= 3) {
		if (!arguments[2]->IsFoldable()) {
			throw InvalidInputException("predict: options must be a constant MAP or STRUCT");
		}
		auto options = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
		if (!options.IsNull()) {
			auto visit = [&](const string &key, const Value &val) {
				if (key == "type") {
					auto v = StringUtil::Lower(val.ToString());
					if (v == "response") {
						data->response_scale = true;
					} else if (v == "link") {
						data->response_scale = false;
					} else {
						throw InvalidInputException("predict: option 'type' must be 'response' or 'link', got '%s'",
						                            val.ToString());
					}
				} else {
					throw InvalidInputException("predict: unknown option '%s' (supported: type)", key);
				}
			};
			if (options.type().id() == LogicalTypeId::STRUCT) {
				auto &vals = StructValue::GetChildren(options);
				for (idx_t i = 0; i < vals.size(); i++) {
					visit(StructType::GetChildName(options.type(), i), vals[i]);
				}
			} else if (options.type().id() == LogicalTypeId::MAP) {
				for (auto &entry : MapValue::GetChildren(options)) {
					auto &kv = StructValue::GetChildren(entry);
					visit(kv[0].ToString(), kv[1]);
				}
			} else {
				throw InvalidInputException("predict: options must be a MAP or STRUCT");
			}
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall("predict");
	return std::move(data);
}

double InverseLink(const string &link, double eta) {
	if (link == "log") {
		return std::exp(eta);
	} else if (link == "logit") {
		return 1.0 / (1.0 + std::exp(-eta));
	} else if (link == "probit") {
		return 0.5 * std::erfc(-eta / std::sqrt(2.0));
	} else if (link == "cloglog") {
		return 1.0 - std::exp(-std::exp(eta));
	} else if (link == "sqrt") {
		return eta * eta;
	} else if (link == "inverse") {
		return 1.0 / eta;
	}
	return eta; // identity
}

static void PredictModelFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
void PredictModelFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	PredictModelFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void PredictModelFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<PredictModelBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &model_vec = args.data[0];
	auto &x_vec = args.data[1];
	auto &entries = StructVector::GetEntries(model_vec);
	auto result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto x_lists = FlatVector::GetData<list_entry_t>(x_vec);
	auto &x_child = ListVector::GetEntry(x_vec);
	auto x_child_data = FlatVector::GetData<double>(x_child);
	auto &x_child_validity = FlatVector::Validity(x_child);

	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(model_vec, row) || FlatVector::IsNull(x_vec, row)) {
			result_validity.SetInvalid(row);
			continue;
		}
		auto x_entry = x_lists[row];
		bool x_has_null = false;
		for (idx_t j = 0; j < x_entry.length; j++) {
			x_has_null |= !x_child_validity.RowIsValid(x_entry.offset + j);
		}
		if (x_has_null) {
			result_validity.SetInvalid(row);
			continue;
		}
		const double *x = x_child_data + x_entry.offset;

		if (bind.kind == ModelKind::ISOTONIC) {
			auto &knot_vec = *entries[bind.knots_idx];
			auto &fit_vec = *entries[bind.fitted_idx];
			if (FlatVector::IsNull(knot_vec, row) || FlatVector::IsNull(fit_vec, row) || x_entry.length != 1) {
				if (x_entry.length != 1 && !FlatVector::IsNull(knot_vec, row)) {
					throw InvalidInputException("predict: an isotonic model takes exactly one feature, got %llu",
					                            (unsigned long long)x_entry.length);
				}
				result_validity.SetInvalid(row);
				continue;
			}
			auto k = FlatVector::GetData<list_entry_t>(knot_vec)[row];
			auto f = FlatVector::GetData<list_entry_t>(fit_vec)[row];
			auto kd = FlatVector::GetData<double>(ListVector::GetEntry(knot_vec)) + k.offset;
			auto fd = FlatVector::GetData<double>(ListVector::GetEntry(fit_vec)) + f.offset;
			idx_t n = MinValue<idx_t>(k.length, f.length);
			if (n == 0) {
				result_validity.SetInvalid(row);
				continue;
			}
			double xv = x[0];
			if (xv <= kd[0]) {
				result_data[row] = fd[0];
			} else if (xv >= kd[n - 1]) {
				result_data[row] = fd[n - 1];
			} else {
				idx_t hi = std::upper_bound(kd, kd + n, xv) - kd;
				idx_t lo = hi - 1;
				double t = (xv - kd[lo]) / (kd[hi] - kd[lo]);
				result_data[row] = fd[lo] + t * (fd[hi] - fd[lo]);
			}
			continue;
		}

		auto &coef_vec = *entries[bind.coef_idx];
		if (FlatVector::IsNull(coef_vec, row)) {
			result_validity.SetInvalid(row);
			continue;
		}
		auto c = FlatVector::GetData<list_entry_t>(coef_vec)[row];
		auto &coef_child = ListVector::GetEntry(coef_vec);
		auto cd = FlatVector::GetData<double>(coef_child);
		auto &coef_validity = FlatVector::Validity(coef_child);
		if (c.length != x_entry.length) {
			throw InvalidInputException("predict: model has %llu coefficients but x has %llu values",
			                            (unsigned long long)c.length, (unsigned long long)x_entry.length);
		}
		double eta = 0.0;
		if (bind.intercept_idx != DConstants::INVALID_INDEX) {
			auto &icpt_vec = *entries[bind.intercept_idx];
			if (!FlatVector::IsNull(icpt_vec, row)) {
				double icpt = FlatVector::GetData<double>(icpt_vec)[row];
				if (std::isfinite(icpt)) {
					eta += icpt;
				}
			}
		}
		for (idx_t j = 0; j < c.length; j++) {
			idx_t ci = c.offset + j;
			if (coef_validity.RowIsValid(ci) && std::isfinite(cd[ci])) {
				eta += cd[ci] * x[j];
			}
		}
		if (bind.response_scale && bind.link_idx != DConstants::INVALID_INDEX) {
			auto &link_vec = *entries[bind.link_idx];
			if (!FlatVector::IsNull(link_vec, row)) {
				eta = InverseLink(FlatVector::GetData<string_t>(link_vec)[row].GetString(), eta);
			}
		}
		result_data[row] = eta;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

} // namespace

// Register the function
void RegisterPredictFunction(ExtensionLoader &loader) {
    ScalarFunction func({LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                         LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
                        LogicalType::LIST(LogicalType::DOUBLE),
                        DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, PredictFunction));

    ScalarFunctionSet func_set("predict");
    func_set.AddFunction(func);
    // Model-aware overloads: predict(model, x[, options])
    for (bool with_options : {false, true}) {
        vector<LogicalType> model_args = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE)};
        if (with_options) {
            model_args.push_back(LogicalType::ANY);
        }
        ScalarFunction model_func(model_args, LogicalType::DOUBLE,
                                  DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, PredictModelFunction), PredictModelBind);
        model_func.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
        func_set.AddFunction(model_func);
    }
    CreateScalarFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription desc;
    desc.description     = "Applies pre-fitted coefficients and intercept to feature data to generate predictions.";
    desc.examples        = {"predict(x, coefficients, intercept)"};
    desc.categories      = {"regression", "prediction"};
    desc.parameter_names = {"x", "coefficients", "intercept"};
    desc.parameter_types = {LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                            LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE};
    info.descriptions.push_back(std::move(desc));
    FunctionDescription model_desc;
    model_desc.description = "Predicts one row from a fitted model STRUCT (as returned by any *_fit_agg or *_fit "
                             "function). GLM models are mapped to the response scale through their link unless "
                             "{'type': 'link'} is given; isotonic models are interpolated.";
    model_desc.examples = {"predict(ols_fit_agg(y, [x1, x2]), [1.5, 2.0])",
                           "predict(model, [x1, x2], {'type': 'link'})"};
    model_desc.categories = {"regression", "prediction"};
    model_desc.parameter_names = {"model", "x"};
    model_desc.parameter_types = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE)};
    info.descriptions.push_back(std::move(model_desc));
    loader.RegisterFunction(std::move(info));

    // linear_predict: the column-layout form under an unambiguous name
    ScalarFunctionSet linear_set("linear_predict");
    linear_set.AddFunction(func);
    CreateScalarFunctionInfo linear_info(std::move(linear_set));
    linear_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription linear_desc;
    linear_desc.description = "Applies coefficients and an intercept to column-layout feature data "
                              "(LIST of feature columns) and returns the list of predictions. Same as the "
                              "3-argument predict().";
    linear_desc.examples = {"linear_predict([[1.0, 2.0], [3.0, 4.0]], [0.5, 1.5], 2.0)"};
    linear_desc.categories = {"regression", "prediction"};
    linear_desc.parameter_names = {"x", "coefficients", "intercept"};
    linear_desc.parameter_types = {LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                                   LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE};
    linear_info.descriptions.push_back(std::move(linear_desc));
    loader.RegisterFunction(std::move(linear_info));
}

} // namespace duckdb
