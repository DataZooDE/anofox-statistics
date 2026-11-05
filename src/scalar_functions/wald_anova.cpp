// wald_anova(model[, terms[, options]]) and fastbw(model[, terms[, options]]): the
// rms-style Wald ANOVA table of chunk tests and the fast backward step-down of any
// fitted model STRUCT with `coefficients` and a coefficient covariance `vcov` (#311).
//
//   SELECT unnest(wald_anova(ols_fit_agg(y, [a1, a2, a3, b], {'compute_inference': true}),
//                            ['a', 'a', 'a', 'b'], {'nonlinear': [false, true, true, false]}),
//                 recursive := true);
//
// `terms` gives one label per slope (NULL: the slope is in no term); slopes with the
// same label form one term, in order of first appearance. By default every slope is
// its own term x1 .. xk. The model's coefficients, intercept, row-major vcov
// (intercept first), df_residual and residual_std_error are read per row and handed
// to anofox-regression's inference::anova (rms::anova.rms) / inference::fastbw
// (rms::fastbw). A NULL model gives NULL; a model without vcov is an input error.

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb_compat.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "anofox_statistics_banner.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr idx_t kNone = DConstants::INVALID_INDEX;

//! Where a model STRUCT keeps the fields the tests read.
struct CovFields {
	idx_t coefficients = kNone, intercept = kNone, vcov = kNone, df_residual = kNone, rse = kNone;

	bool operator==(const CovFields &o) const {
		return coefficients == o.coefficients && intercept == o.intercept && vcov == o.vcov &&
		       df_residual == o.df_residual && rse == o.rse;
	}
};

bool IsNumericList(const LogicalType &t) {
	return t.id() == LogicalTypeId::LIST && ListType::GetChildType(t).IsNumeric();
}

CovFields FindCovFields(const char *fn, const Expression &arg) {
	auto &type = ReturnTypeOf(arg);
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: model must be the STRUCT returned by a *_fit_agg or *_fit function, got %s",
		                            fn, type.ToString());
	}
	CovFields f;
	auto &children = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = NameString(children[i].first);
		auto &t = children[i].second;
		if (name == "coefficients" && IsNumericList(t)) {
			f.coefficients = i;
		} else if (name == "vcov" && IsNumericList(t)) {
			f.vcov = i;
		} else if (t.IsNumeric()) {
			if (name == "intercept") {
				f.intercept = i;
			} else if (name == "df_residual") {
				f.df_residual = i;
			} else if (name == "residual_std_error") {
				f.rse = i;
			}
		}
	}
	if (f.coefficients == kNone) {
		throw InvalidInputException("%s: model has no 'coefficients' field; got %s", fn, type.ToString());
	}
	if (f.vcov == kNone) {
		throw InvalidInputException("%s: model has no coefficient covariance 'vcov' (an OLS, WLS, Ridge or GLM "
		                            "fit with {'compute_inference': true}); got %s",
		                            fn, type.ToString());
	}
	return f;
}

//! A constant argument (terms or options), or NULL.
Value ConstantArgument(const char *fn, const char *what, ClientContext &context, Expression &arg) {
	RequireResolvedParameters(arg);
	if (!arg.IsFoldable()) {
		throw InvalidInputException("%s: %s must be a constant", fn, what);
	}
	return UnwrapVariant(ExpressionExecutor::EvaluateScalar(context, arg));
}

//! The term specification shared by both functions: one label per slope.
struct TermLabels {
	bool given = false;
	vector<int64_t> term_of_column; // -1: no term
	vector<string> names;           // by term index

	bool operator==(const TermLabels &o) const {
		return given == o.given && term_of_column == o.term_of_column && names == o.names;
	}
};

TermLabels ParseTerms(const char *fn, const Value &terms) {
	TermLabels t;
	if (terms.IsNull()) {
		return t;
	}
	if (terms.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("%s: terms must be a VARCHAR[] with one label per coefficient, got %s", fn,
		                            terms.type().ToString());
	}
	t.given = true;
	std::unordered_map<string, int64_t> index;
	for (auto &v : ListValue::GetChildren(terms)) {
		auto u = UnwrapVariant(v);
		if (u.IsNull()) {
			t.term_of_column.push_back(-1);
			continue;
		}
		auto label = u.ToString();
		auto it = index.find(label);
		if (it == index.end()) {
			it = index.emplace(label, static_cast<int64_t>(t.names.size())).first;
			t.names.push_back(label);
		}
		t.term_of_column.push_back(it->second);
	}
	if (t.names.empty()) {
		throw InvalidInputException("%s: terms has no non-NULL label", fn);
	}
	return t;
}

//! Visit the entries of a constant MAP / STRUCT option bag.
template <class F>
void VisitOptions(const char *fn, const Value &options, F &&visit) {
	if (options.IsNull()) {
		return;
	}
	if (options.type().id() == LogicalTypeId::STRUCT) {
		auto &vals = StructValue::GetChildren(options);
		for (idx_t i = 0; i < vals.size(); i++) {
			visit(StringUtil::Lower(NameString(StructType::GetChildName(options.type(), i))), UnwrapVariant(vals[i]));
		}
	} else if (options.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(options)) {
			auto &kv = StructValue::GetChildren(entry);
			visit(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else {
		throw InvalidInputException("%s: options must be a MAP or STRUCT", fn);
	}
}

double OptionDouble(const char *fn, const string &key, const Value &v) {
	if (v.IsNull()) {
		throw InvalidInputException("%s: option '%s' must not be NULL", fn, key);
	}
	return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
}

//! The model at `row`: slopes (NULL elements as NaN), intercept, vcov.
struct ModelRow {
	vector<double> coefficients;
	double intercept = std::numeric_limits<double>::quiet_NaN();
	vector<double> vcov;
	idx_t dim = 0;
	double df_residual = std::numeric_limits<double>::quiet_NaN();
	double rse = std::numeric_limits<double>::quiet_NaN();
};

double FieldDouble(const StructEntries &entries, idx_t field, idx_t row) {
	if (field == kNone) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	auto v = StructEntry(entries[field]).GetValue(row);
	return v.IsNull() ? std::numeric_limits<double>::quiet_NaN() : v.GetValue<double>();
}

vector<double> ListDoubles(const Value &v) {
	vector<double> out;
	for (auto &c : ListValue::GetChildren(v)) {
		out.push_back(c.IsNull() ? std::numeric_limits<double>::quiet_NaN() : c.GetValue<double>());
	}
	return out;
}

//! false when the model or its coefficients are NULL (the result is NULL).
bool ReadModelRow(const char *fn, Vector &model, const CovFields &f, idx_t row, ModelRow &out) {
	if (FlatVector::IsNull(model, row)) {
		return false;
	}
	auto &entries = StructVector::GetEntries(model);
	auto coef = StructEntry(entries[f.coefficients]).GetValue(row);
	if (coef.IsNull()) {
		return false;
	}
	auto vcov = StructEntry(entries[f.vcov]).GetValue(row);
	if (vcov.IsNull()) {
		throw InvalidInputException("%s: the model has no coefficient covariance (vcov is NULL); fit it with "
		                            "{'compute_inference': true}",
		                            fn);
	}
	out.coefficients = ListDoubles(coef);
	out.vcov = ListDoubles(vcov);
	idx_t dim = 0;
	while (dim * dim < out.vcov.size()) {
		dim++;
	}
	if (dim * dim != out.vcov.size()) {
		throw InvalidInputException("%s: vcov has %llu entries, not a square matrix", fn,
		                            (unsigned long long)out.vcov.size());
	}
	out.dim = dim;
	out.intercept = FieldDouble(entries, f.intercept, row);
	out.df_residual = FieldDouble(entries, f.df_residual, row);
	out.rse = FieldDouble(entries, f.rse, row);
	return true;
}

//! The labels of the row's model: the given ones, or x1 .. xk.
TermLabels LabelsFor(const char *fn, const TermLabels &given, idx_t p) {
	if (given.given) {
		if (given.term_of_column.size() != p) {
			throw InvalidInputException("%s: terms has %llu labels but the model has %llu coefficients", fn,
			                            (unsigned long long)given.term_of_column.size(), (unsigned long long)p);
		}
		return given;
	}
	TermLabels t;
	t.given = true;
	for (idx_t j = 0; j < p; j++) {
		t.term_of_column.push_back(static_cast<int64_t>(j));
		t.names.push_back("x" + std::to_string(j + 1));
	}
	return t;
}

Value NullableDouble(double v) {
	return std::isfinite(v) ? Value::DOUBLE(v) : Value(LogicalType::DOUBLE);
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding asserts this
// in debug builds).
template <void (*IMPL)(DataChunk &, ExpressionState &, Vector &)>
void ConstantAware(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	IMPL(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// wald_anova
//===--------------------------------------------------------------------===//

constexpr const char *kAnova = "wald_anova";

enum class WaldTestKind : uint8_t { AUTO, CHISQ, F };

struct WaldAnovaBindData : public FunctionData {
	CovFields fields;
	TermLabels terms;
	vector<bool> nonlinear;                // empty: all linear
	vector<std::pair<string, std::pair<string, string>>> interactions;
	WaldTestKind test = WaldTestKind::AUTO;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<WaldAnovaBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<WaldAnovaBindData>();
		return fields == o.fields && terms == o.terms && nonlinear == o.nonlinear &&
		       interactions == o.interactions && test == o.test;
	}
};

LogicalType WaldAnovaRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("label", LogicalType::VARCHAR));
	c.push_back(make_pair("kind", LogicalType::VARCHAR));
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("columns", LogicalType::LIST(LogicalType::BIGINT)));
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("df2", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("chi_square", LogicalType::DOUBLE));
	c.push_back(make_pair("chi_square_minus_df", LogicalType::DOUBLE));
	c.push_back(make_pair("partial_ss", LogicalType::DOUBLE));
	c.push_back(make_pair("mean_sq", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

std::pair<string, string> ParseFactors(const string &label, const Value &v) {
	if (v.IsNull() || v.type().id() != LogicalTypeId::LIST || ListValue::GetChildren(v).size() != 2) {
		throw InvalidInputException("%s: interaction '%s' must map to a list of its two factor labels", kAnova, label);
	}
	auto &f = ListValue::GetChildren(v);
	if (f[0].IsNull() || f[1].IsNull()) {
		throw InvalidInputException("%s: interaction '%s' has a NULL factor label", kAnova, label);
	}
	return {UnwrapVariant(f[0]).ToString(), UnwrapVariant(f[1]).ToString()};
}

unique_ptr<FunctionData> WaldAnovaBind(ClientContext &context, ScalarBindFunction &,
                                       vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<WaldAnovaBindData>();
	data->fields = FindCovFields(kAnova, *arguments[0]);
	if (arguments.size() >= 2) {
		data->terms = ParseTerms(kAnova, ConstantArgument(kAnova, "terms", context, *arguments[1]));
	}
	if (arguments.size() >= 3) {
		auto options = ConstantArgument(kAnova, "options", context, *arguments[2]);
		VisitOptions(kAnova, options, [&](const string &key, const Value &val) {
			if (key == "nonlinear") {
				if (val.IsNull()) {
					return;
				}
				if (val.type().id() != LogicalTypeId::LIST) {
					throw InvalidInputException("%s: option 'nonlinear' must be a BOOLEAN[] with one flag per "
					                            "coefficient",
					                            kAnova);
				}
				data->nonlinear.clear();
				for (auto &b : ListValue::GetChildren(val)) {
					auto u = UnwrapVariant(b);
					data->nonlinear.push_back(!u.IsNull() && u.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>());
				}
			} else if (key == "interactions") {
				if (val.IsNull()) {
					return;
				}
				if (val.type().id() == LogicalTypeId::MAP) {
					for (auto &entry : MapValue::GetChildren(val)) {
						auto &kv = StructValue::GetChildren(entry);
						auto label = kv[0].ToString();
						data->interactions.emplace_back(label, ParseFactors(label, UnwrapVariant(kv[1])));
					}
				} else if (val.type().id() == LogicalTypeId::STRUCT) {
					auto &vals = StructValue::GetChildren(val);
					for (idx_t i = 0; i < vals.size(); i++) {
						auto label = NameString(StructType::GetChildName(val.type(), i));
						data->interactions.emplace_back(label, ParseFactors(label, UnwrapVariant(vals[i])));
					}
				} else {
					throw InvalidInputException("%s: option 'interactions' must be a MAP from an interaction label "
					                            "to its two factor labels, e.g. MAP {'age * sex': ['age', 'sex']}",
					                            kAnova);
				}
			} else if (key == "test") {
				auto v = val.IsNull() ? string("auto") : StringUtil::Lower(val.ToString());
				if (v == "auto") {
					data->test = WaldTestKind::AUTO;
				} else if (v == "chisq" || v == "chi-square" || v == "chi_square" || v == "chi2") {
					data->test = WaldTestKind::CHISQ;
				} else if (v == "f") {
					data->test = WaldTestKind::F;
				} else {
					throw InvalidInputException("%s: option 'test' must be 'auto', 'chisq' or 'F', got '%s'", kAnova,
					                            val.ToString());
				}
			} else {
				throw InvalidInputException("%s: unknown option '%s' (supported: nonlinear, interactions, test)",
				                            kAnova, key);
			}
		});
	}
	if (!data->interactions.empty() && !data->terms.given) {
		throw InvalidInputException("%s: option 'interactions' needs term labels", kAnova);
	}
	for (auto &ia : data->interactions) {
		for (auto *name : {&ia.first, &ia.second.first, &ia.second.second}) {
			bool found = false;
			for (auto &n : data->terms.names) {
				found = found || n == *name;
			}
			if (!found) {
				throw InvalidInputException("%s: interaction label '%s' is not one of the term labels", kAnova, *name);
			}
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(kAnova);
	return std::move(data);
}

void WaldAnovaImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<WaldAnovaBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto row_type = WaldAnovaRowType();
	// Whether the model is linear (OLS / WLS / Ridge): it carries a residual standard error.
	const bool linear = bind.fields.rse != kNone;
	for (idx_t row = 0; row < count; row++) {
		ModelRow m;
		if (!ReadModelRow(kAnova, args.data[0], bind.fields, row, m)) {
			result.SetValue(row, Value(LogicalType::LIST(row_type)));
			continue;
		}
		const idx_t p = m.coefficients.size();
		auto labels = LabelsFor(kAnova, bind.terms, p);
		if (!bind.nonlinear.empty() && bind.nonlinear.size() != p) {
			throw InvalidInputException("%s: nonlinear has %llu flags but the model has %llu coefficients", kAnova,
			                            (unsigned long long)bind.nonlinear.size(), (unsigned long long)p);
		}
		bool f_test = bind.test == WaldTestKind::F ||
		              (bind.test == WaldTestKind::AUTO && linear && std::isfinite(m.df_residual));
		if (f_test && !std::isfinite(m.df_residual)) {
			throw InvalidInputException("%s: the F test needs the model's df_residual", kAnova);
		}
		const idx_t n_terms = labels.names.size();
		vector<int64_t> factors(2 * n_terms, -1);
		auto index_of = [&](const string &name) {
			for (idx_t t = 0; t < n_terms; t++) {
				if (labels.names[t] == name) {
					return static_cast<int64_t>(t);
				}
			}
			return int64_t(-1);
		};
		for (auto &ia : bind.interactions) {
			auto t = index_of(ia.first);
			factors[2 * t] = index_of(ia.second.first);
			factors[2 * t + 1] = index_of(ia.second.second);
		}
		vector<const char *> names;
		for (auto &n : labels.names) {
			names.push_back(n.c_str());
		}
		// std::vector<bool> is packed: copy into a plain array.
		std::unique_ptr<bool[]> nonlinear(new bool[p > 0 ? p : 1]);
		for (idx_t j = 0; j < p; j++) {
			nonlinear[j] = !bind.nonlinear.empty() && bind.nonlinear[j];
		}
		double sigma2 = linear && std::isfinite(m.rse) ? m.rse * m.rse : std::numeric_limits<double>::quiet_NaN();
		AnofoxWaldAnovaResult res {};
		AnofoxError error;
		if (!anofox_wald_anova(m.coefficients.data(), p, m.vcov.data(), m.dim, labels.term_of_column.data(),
		                       nonlinear.get(), names.data(), n_terms, factors.data(), f_test, m.df_residual, sigma2,
		                       &res, &error)) {
			ThrowUnlessDegenerate(kAnova, error);
			result.SetValue(row, Value(LogicalType::LIST(row_type)));
			continue;
		}
		const string method = f_test ? "Wald F" : "Wald chi-square";
		vector<Value> rows;
		for (idx_t i = 0; i < res.n_rows; i++) {
			auto &r = res.rows[i];
			vector<Value> cols;
			for (idx_t c = 0; c < r.n_columns; c++) {
				cols.push_back(Value::BIGINT(static_cast<int64_t>(r.columns[c]) + 1));
			}
			double ms = r.partial_ss / r.df;
			vector<Value> v;
			v.push_back(Value(string(r.label)));
			v.push_back(Value(string(r.kind)));
			v.push_back(r.term >= 0 ? Value(labels.names[r.term]) : Value(LogicalType::VARCHAR));
			v.push_back(Value::LIST(LogicalType::BIGINT, std::move(cols)));
			v.push_back(Value(method));
			v.push_back(NullableDouble(f_test ? r.f_statistic : r.chi_square));
			v.push_back(NullableDouble(r.df));
			v.push_back(NullableDouble(f_test ? r.df_residual : std::numeric_limits<double>::quiet_NaN()));
			v.push_back(NullableDouble(r.p_value));
			v.push_back(NullableDouble(r.chi_square));
			v.push_back(NullableDouble(r.chi_square - r.df));
			v.push_back(NullableDouble(r.partial_ss));
			v.push_back(NullableDouble(ms));
			rows.push_back(Value::STRUCT(row_type, std::move(v)));
		}
		if (std::isfinite(res.error_df)) {
			vector<Value> v;
			v.push_back(Value("ERROR"));
			v.push_back(Value("error"));
			v.push_back(Value(LogicalType::VARCHAR));
			v.push_back(Value(LogicalType::LIST(LogicalType::BIGINT)));
			v.push_back(Value(method));
			v.push_back(Value(LogicalType::DOUBLE));
			v.push_back(NullableDouble(res.error_df));
			v.push_back(Value(LogicalType::DOUBLE));
			v.push_back(Value(LogicalType::DOUBLE));
			v.push_back(Value(LogicalType::DOUBLE));
			v.push_back(Value(LogicalType::DOUBLE));
			v.push_back(NullableDouble(res.error_ss));
			v.push_back(NullableDouble(res.error_ms));
			rows.push_back(Value::STRUCT(row_type, std::move(v)));
		}
		anofox_free_wald_anova(&res);
		result.SetValue(row, Value::LIST(row_type, std::move(rows)));
	}
}

//===--------------------------------------------------------------------===//
// fastbw
//===--------------------------------------------------------------------===//

constexpr const char *kFastbw = "fastbw";

struct FastbwBindData : public FunctionData {
	CovFields fields;
	TermLabels terms;
	int32_t rule = 0; // 0 aic, 1 p
	bool individual = false;
	double k_aic = 2.0;
	double aics = 0.0;
	double sls = 0.05;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<FastbwBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<FastbwBindData>();
		return fields == o.fields && terms == o.terms && rule == o.rule && individual == o.individual &&
		       k_aic == o.k_aic && aics == o.aics && sls == o.sls;
	}
};

LogicalType FastbwStepType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("chi_square", LogicalType::DOUBLE));
	c.push_back(make_pair("df", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_chi_square", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_df", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("aic", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType FastbwResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("deleted", LogicalType::LIST(FastbwStepType())));
	c.push_back(make_pair("kept_terms", LogicalType::LIST(LogicalType::VARCHAR)));
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("rule", LogicalType::VARCHAR));
	c.push_back(make_pair("type", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

unique_ptr<FunctionData> FastbwBind(ClientContext &context, ScalarBindFunction &,
                                    vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<FastbwBindData>();
	data->fields = FindCovFields(kFastbw, *arguments[0]);
	if (arguments.size() >= 2) {
		data->terms = ParseTerms(kFastbw, ConstantArgument(kFastbw, "terms", context, *arguments[1]));
	}
	if (arguments.size() >= 3) {
		auto options = ConstantArgument(kFastbw, "options", context, *arguments[2]);
		VisitOptions(kFastbw, options, [&](const string &key, const Value &val) {
			if (key == "rule") {
				auto v = val.IsNull() ? string("aic") : StringUtil::Lower(val.ToString());
				if (v == "aic") {
					data->rule = 0;
				} else if (v == "p") {
					data->rule = 1;
				} else {
					throw InvalidInputException("%s: option 'rule' must be 'aic' or 'p', got '%s'", kFastbw,
					                            val.ToString());
				}
			} else if (key == "type") {
				auto v = val.IsNull() ? string("residual") : StringUtil::Lower(val.ToString());
				if (v == "residual") {
					data->individual = false;
				} else if (v == "individual") {
					data->individual = true;
				} else {
					throw InvalidInputException("%s: option 'type' must be 'residual' or 'individual', got '%s'",
					                            kFastbw, val.ToString());
				}
			} else if (key == "sls") {
				data->sls = OptionDouble(kFastbw, key, val);
				if (!(data->sls > 0.0 && data->sls < 1.0)) {
					throw InvalidInputException("%s: option 'sls' must be in (0, 1), got %s", kFastbw,
					                            val.ToString());
				}
			} else if (key == "aics") {
				data->aics = OptionDouble(kFastbw, key, val);
				if (!std::isfinite(data->aics)) {
					throw InvalidInputException("%s: option 'aics' must be finite", kFastbw);
				}
			} else if (key == "k_aic") {
				data->k_aic = OptionDouble(kFastbw, key, val);
				if (!(std::isfinite(data->k_aic) && data->k_aic >= 0.0)) {
					throw InvalidInputException("%s: option 'k_aic' must be finite and >= 0, got %s", kFastbw,
					                            val.ToString());
				}
			} else {
				throw InvalidInputException("%s: unknown option '%s' (supported: rule, type, sls, aics, k_aic)",
				                            kFastbw, key);
			}
		});
	}
	PostHogTelemetry::Instance().RecordFunctionCall(kFastbw);
	return std::move(data);
}

void FastbwImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<FastbwBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto result_type = FastbwResultType();
	auto step_type = FastbwStepType();
	for (idx_t row = 0; row < count; row++) {
		ModelRow m;
		if (!ReadModelRow(kFastbw, args.data[0], bind.fields, row, m)) {
			result.SetValue(row, Value(result_type));
			continue;
		}
		const idx_t p = m.coefficients.size();
		auto labels = LabelsFor(kFastbw, bind.terms, p);
		AnofoxFastbwResult res {};
		AnofoxError error;
		if (!anofox_fastbw(m.coefficients.data(), p, m.intercept, m.vcov.data(), m.dim, labels.term_of_column.data(),
		                   labels.names.size(), bind.rule, bind.individual, bind.k_aic, bind.aics, bind.sls, &res,
		                   &error)) {
			ThrowUnlessDegenerate(kFastbw, error);
			result.SetValue(row, Value(result_type));
			continue;
		}
		vector<Value> steps;
		for (idx_t i = 0; i < res.n_deleted; i++) {
			auto &d = res.deleted[i];
			vector<Value> v;
			v.push_back(Value(labels.names[d.term]));
			v.push_back(NullableDouble(d.chi_square));
			v.push_back(NullableDouble(d.df));
			v.push_back(NullableDouble(d.p_value));
			v.push_back(NullableDouble(d.residual_chi_square));
			v.push_back(NullableDouble(d.residual_df));
			v.push_back(NullableDouble(d.residual_p_value));
			v.push_back(NullableDouble(d.aic));
			steps.push_back(Value::STRUCT(step_type, std::move(v)));
		}
		vector<Value> kept;
		for (idx_t i = 0; i < res.n_kept; i++) {
			kept.push_back(Value(labels.names[res.kept_terms[i]]));
		}
		vector<Value> coefs;
		for (idx_t i = 0; i < res.n_coefficients; i++) {
			coefs.push_back(NullableDouble(res.coefficients[i]));
		}
		vector<Value> v;
		v.push_back(Value::LIST(step_type, std::move(steps)));
		v.push_back(Value::LIST(LogicalType::VARCHAR, std::move(kept)));
		v.push_back(Value::LIST(LogicalType::DOUBLE, std::move(coefs)));
		v.push_back(NullableDouble(res.intercept));
		v.push_back(Value(bind.rule == 0 ? "aic" : "p"));
		v.push_back(Value(bind.individual ? "individual" : "residual"));
		anofox_free_fastbw(&res);
		result.SetValue(row, Value::STRUCT(result_type, std::move(v)));
	}
}

template <class FN, class BIND>
void RegisterSet(ExtensionLoader &loader, const char *name, const LogicalType &return_type, FN fn, BIND bind,
                 const string &description, const vector<string> &examples) {
	ScalarFunctionSet set(name);
	vector<vector<LogicalType>> signatures = {{LogicalType::ANY},
	                                          {LogicalType::ANY, LogicalType::ANY},
	                                          {LogicalType::ANY, LogicalType::ANY, LogicalType::ANY}};
	for (auto &args : signatures) {
		ScalarFunction f(args, return_type, fn, bind);
		MarkFallible(f);
		SetNullHandlingOf(f, FunctionNullHandling::SPECIAL_HANDLING);
		set.AddFunction(f);
	}
	CreateScalarFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	// One description per overload (model / model, terms / model, terms, options).
	const vector<string> names = {"model", "terms", "options"};
	const vector<LogicalType> types = {LogicalType::ANY, LogicalType::ANY, LogicalType::ANY};
	for (idx_t n = 1; n <= 3; n++) {
		FunctionDescription d;
		d.description = description;
		d.examples = examples;
		d.categories = {"regression", "hypothesis-test"};
		d.parameter_names = vector<string>(names.begin(), names.begin() + n);
		d.parameter_types = vector<LogicalType>(types.begin(), types.begin() + n);
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterWaldAnovaFunctions(ExtensionLoader &loader) {
	RegisterSet(loader, kAnova, LogicalType::LIST(WaldAnovaRowType()),
	            DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, ConstantAware<WaldAnovaImpl>), ANOFOX_SCALAR_BIND(WaldAnovaBind),
	            "rms-style ANOVA table of Wald chunk tests (rms::anova.rms) from a fitted model's coefficients and "
	            "covariance: per term the test of all its columns (with its interactions), its nonlinear and "
	            "interaction chunks, then TOTAL NONLINEAR, TOTAL INTERACTION and TOTAL. terms: one label per "
	            "coefficient; options: nonlinear (BOOLEAN[]), interactions (MAP label -> [factor_a, factor_b]), test "
	            "('auto' = F for linear models, 'chisq', 'F').",
	            {"wald_anova(ols_fit_agg(y, [x1, x2, x3], {'compute_inference': true}), ['a', 'a', 'b'])"});
	RegisterSet(loader, kFastbw, FastbwResultType(),
	            DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, ConstantAware<FastbwImpl>), ANOFOX_SCALAR_BIND(FastbwBind),
	            "Fast backward step-down (rms::fastbw, Lawless & Singhal) from a fitted model's coefficients and "
	            "covariance: the deletion order with the Wald and residual chi-square tests, the kept terms and the "
	            "approximate reduced-model coefficients. Coefficients with a NULL term label are never deleted. "
	            "options: rule ('aic' | 'p'), type ('residual' | 'individual'), sls, aics, k_aic.",
	            {"fastbw(logistic_fit_agg(y, [x1, x2, x3], {'compute_inference': true}))"});
}

} // namespace duckdb
