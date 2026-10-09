// tidy(model[, names]) and glance(model): broom-style views of a fitted model STRUCT.
//
//   SELECT unnest(tidy(ols_fit_agg(y, [x1, x2], {'compute_inference': true})), recursive := true) FROM t;
//   SELECT g, glance(ols_fit_agg(y, [x1, x2])).* FROM t GROUP BY g;
//
// Both are scalar functions over the model STRUCT rather than table functions, so they
// compose with GROUP BY and window aggregates. The model's field layout is inspected
// at bind time; fields a model does not have come back NULL.

#include <cmath>
#include <string>

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "anofox_statistics_banner.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr idx_t kNone = DConstants::INVALID_INDEX;

struct TidyBindData : public FunctionData {
	idx_t coef = kNone, intercept = kNone, std_errors = kNone, statistic = kNone, p_values = kNone,
	      ci_lower = kNone, ci_upper = kNone, intercept_se = kNone;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TidyBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<TidyBindData>();
		return coef == o.coef && intercept == o.intercept && std_errors == o.std_errors &&
		       statistic == o.statistic && p_values == o.p_values && ci_lower == o.ci_lower &&
		       ci_upper == o.ci_upper && intercept_se == o.intercept_se;
	}
};

bool IsDoubleList(const LogicalType &t) {
	return t.id() == LogicalTypeId::LIST && ListType::GetChildType(t).id() == LogicalTypeId::DOUBLE;
}

const LogicalType &RequireModelStruct(const char *fn, const unique_ptr<Expression> &arg) {
	auto &type = arg->return_type;
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s(model): model must be the STRUCT returned by a *_fit_agg or *_fit function, "
		                            "got %s",
		                            fn, type.ToString());
	}
	return type;
}

LogicalType TidyRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("estimate", LogicalType::DOUBLE));
	c.push_back(make_pair("std_error", LogicalType::DOUBLE));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	c.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

unique_ptr<FunctionData> TidyBind(ClientContext &, ScalarFunction &, vector<unique_ptr<Expression>> &arguments) {
	auto &type = RequireModelStruct("tidy", arguments[0]);
	auto data = make_uniq<TidyBindData>();
	auto &children = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = children[i].first;
		auto &t = children[i].second;
		if (IsDoubleList(t)) {
			if (name == "coefficients") {
				data->coef = i;
			} else if (name == "std_errors") {
				data->std_errors = i;
			} else if (name == "t_values" || name == "z_values") {
				data->statistic = i;
			} else if (name == "p_values") {
				data->p_values = i;
			} else if (name == "ci_lower") {
				data->ci_lower = i;
			} else if (name == "ci_upper") {
				data->ci_upper = i;
			}
		} else if (t.id() == LogicalTypeId::DOUBLE) {
			if (name == "intercept") {
				data->intercept = i;
			} else if (name == "intercept_std_error") {
				data->intercept_se = i;
			}
		}
	}
	if (data->coef == kNone) {
		throw InvalidInputException("tidy(model): model has no 'coefficients' field; got %s", type.ToString());
	}
	PostHogTelemetry::Instance().RecordFunctionCall("tidy");
	return std::move(data);
}

//! A LIST(DOUBLE) child of the (flattened) model struct at one row.
struct DoubleListView {
	const double *data = nullptr;
	const ValidityMask *validity = nullptr;
	idx_t offset = 0;
	idx_t length = 0;
	bool present = false;

	DoubleListView(const vector<unique_ptr<Vector>> &entries, idx_t field, idx_t row) {
		if (field == kNone) {
			return;
		}
		auto &vec = *entries[field];
		if (FlatVector::IsNull(vec, row)) {
			return;
		}
		auto entry = FlatVector::GetData<list_entry_t>(vec)[row];
		auto &child = ListVector::GetEntry(vec);
		data = FlatVector::GetData<double>(child);
		validity = &FlatVector::Validity(child);
		offset = entry.offset;
		length = entry.length;
		present = true;
	}

	//! Element k, or NaN when missing / NULL.
	double At(idx_t k) const {
		if (!present || k >= length || !validity->RowIsValid(offset + k)) {
			return std::nan("");
		}
		return data[offset + k];
	}
};

double ScalarField(const vector<unique_ptr<Vector>> &entries, idx_t field, idx_t row) {
	if (field == kNone || FlatVector::IsNull(*entries[field], row)) {
		return std::nan("");
	}
	return FlatVector::GetData<double>(*entries[field])[row];
}

void WriteDouble(Vector &vec, idx_t idx, double v) {
	if (std::isfinite(v)) {
		FlatVector::GetData<double>(vec)[idx] = v;
	} else {
		FlatVector::SetNull(vec, idx, true);
	}
}

static void TidyFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
void TidyFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	TidyFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void TidyFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<TidyBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &model = args.data[0];
	auto &entries = StructVector::GetEntries(model);
	bool has_names = args.ColumnCount() >= 2;

	auto result_lists = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(model, row)) {
			result_validity.SetInvalid(row);
			continue;
		}
		DoubleListView coef(entries, bind.coef, row);
		if (!coef.present) {
			result_validity.SetInvalid(row);
			continue;
		}
		DoubleListView se(entries, bind.std_errors, row), stat(entries, bind.statistic, row),
		    pv(entries, bind.p_values, row), lo(entries, bind.ci_lower, row), hi(entries, bind.ci_upper, row);
		double intercept = ScalarField(entries, bind.intercept, row);
		bool has_intercept = std::isfinite(intercept);
		idx_t k = coef.length;
		// Inference lists either cover the slopes only, or the intercept first and then the slopes.
		bool inference_has_intercept = se.present && se.length == k + 1;
		idx_t shift = inference_has_intercept ? 1 : 0;

		// Optional term names
		vector<string> names;
		if (has_names && !FlatVector::IsNull(args.data[1], row)) {
			auto entry = FlatVector::GetData<list_entry_t>(args.data[1])[row];
			auto &child = ListVector::GetEntry(args.data[1]);
			auto child_data = FlatVector::GetData<string_t>(child);
			for (idx_t j = 0; j < entry.length; j++) {
				names.push_back(FlatVector::IsNull(child, entry.offset + j) ? ""
				                                                           : child_data[entry.offset + j].GetString());
			}
			if (names.size() != k) {
				throw InvalidInputException("tidy: got %llu names for %llu coefficients", (unsigned long long)names.size(),
				                            (unsigned long long)k);
			}
		}

		idx_t n_terms = k + (has_intercept ? 1 : 0);
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_terms);
		auto &rows = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(rows);
		idx_t out = list_offset;

		if (has_intercept) {
			fields[0]->SetValue(out, Value("(Intercept)"));
			WriteDouble(*fields[1], out, intercept);
			double i_se = inference_has_intercept ? se.At(0) : ScalarField(entries, bind.intercept_se, row);
			WriteDouble(*fields[2], out, i_se);
			WriteDouble(*fields[3], out, inference_has_intercept ? stat.At(0) : std::nan(""));
			WriteDouble(*fields[4], out, inference_has_intercept ? pv.At(0) : std::nan(""));
			WriteDouble(*fields[5], out, inference_has_intercept ? lo.At(0) : std::nan(""));
			WriteDouble(*fields[6], out, inference_has_intercept ? hi.At(0) : std::nan(""));
			out++;
		}
		for (idx_t j = 0; j < k; j++, out++) {
			fields[0]->SetValue(out, Value(names.empty() ? "x" + std::to_string(j + 1) : names[j]));
			WriteDouble(*fields[1], out, coef.At(j));
			WriteDouble(*fields[2], out, se.At(j + shift));
			WriteDouble(*fields[3], out, stat.At(j + shift));
			WriteDouble(*fields[4], out, pv.At(j + shift));
			WriteDouble(*fields[5], out, lo.At(j + shift));
			WriteDouble(*fields[6], out, hi.At(j + shift));
		}
		result_lists[row] = {list_offset, n_terms};
		ListVector::SetListSize(result, list_offset + n_terms);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

//===--------------------------------------------------------------------===//
// glance: the model's scalar (non-LIST, non-STRUCT) fields
//===--------------------------------------------------------------------===//
struct GlanceBindData : public FunctionData {
	vector<idx_t> fields;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<GlanceBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return fields == other_p.Cast<GlanceBindData>().fields;
	}
};

unique_ptr<FunctionData> GlanceBind(ClientContext &, ScalarFunction &bound_function,
                                    vector<unique_ptr<Expression>> &arguments) {
	auto &type = RequireModelStruct("glance", arguments[0]);
	auto data = make_uniq<GlanceBindData>();
	child_list_t<LogicalType> out;
	auto &children = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto id = children[i].second.id();
		if (id == LogicalTypeId::LIST || id == LogicalTypeId::STRUCT || id == LogicalTypeId::MAP ||
		    id == LogicalTypeId::ARRAY) {
			continue;
		}
		data->fields.push_back(i);
		out.push_back(children[i]);
	}
	if (out.empty()) {
		throw InvalidInputException("glance(model): model has no scalar fields");
	}
	bound_function.return_type = LogicalType::STRUCT(std::move(out));
	PostHogTelemetry::Instance().RecordFunctionCall("glance");
	return std::move(data);
}

static void GlanceFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
void GlanceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	GlanceFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void GlanceFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<GlanceBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &model = args.data[0];
	auto &entries = StructVector::GetEntries(model);
	auto &out = StructVector::GetEntries(result);
	for (idx_t k = 0; k < bind.fields.size(); k++) {
		out[k]->Reference(*entries[bind.fields[k]]);
	}
	auto &validity = FlatVector::Validity(result);
	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(model, row)) {
			validity.SetInvalid(row);
		}
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

} // namespace

void RegisterTidyGlanceFunctions(ExtensionLoader &loader) {
	ScalarFunctionSet tidy_set("tidy");
	ScalarFunction tidy1({LogicalType::ANY}, LogicalType::LIST(TidyRowType()),
	                     DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, TidyFunction), TidyBind);
	ScalarFunction tidy2({LogicalType::ANY, LogicalType::LIST(LogicalType::VARCHAR)}, LogicalType::LIST(TidyRowType()),
	                     DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, TidyFunction), TidyBind);
	tidy1.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	tidy2.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	tidy_set.AddFunction(tidy1);
	tidy_set.AddFunction(tidy2);
	CreateScalarFunctionInfo tidy_info(std::move(tidy_set));
	tidy_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription td;
	td.description = "One entry per model term (intercept first): term, estimate, std_error, statistic, p_value, "
	                 "conf_low, conf_high. Unnest it to get a coefficient table; pass term names as a second "
	                 "argument.";
	td.examples = {"unnest(tidy(ols_fit_agg(y, [x1, x2], {'compute_inference': true})), recursive := true)",
	               "tidy(model, ['price', 'promo'])"};
	td.categories = {"regression"};
	td.parameter_names = {"model"};
	td.parameter_types = {LogicalType::ANY};
	tidy_info.descriptions.push_back(std::move(td));
	loader.RegisterFunction(std::move(tidy_info));

	ScalarFunction glance("glance", {LogicalType::ANY}, LogicalType::ANY, DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, GlanceFunction),
	                      GlanceBind);
	glance.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	CreateScalarFunctionInfo glance_info(glance);
	glance_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription gd;
	gd.description = "Model-level summary: the scalar fields of a fitted model STRUCT (r_squared, aic, "
	                 "n_observations, ...), without the per-coefficient lists.";
	gd.examples = {"glance(ols_fit_agg(y, [x1, x2])).*"};
	gd.categories = {"regression"};
	gd.parameter_names = {"model"};
	gd.parameter_types = {LogicalType::ANY};
	glance_info.descriptions.push_back(std::move(gd));
	loader.RegisterFunction(std::move(glance_info));
}

} // namespace duckdb
