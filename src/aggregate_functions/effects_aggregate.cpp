// Effects, contrasts, effect grids and nomograms on the covariate scale (#319),
// as rms::summary, rms::contrast, rms::Predict and rms::nomogram:
//
//   effects_term(name, kind [, levels]) / effects_term(name, spline_spec(...))   -> term STRUCT
//   effects_interaction(a, b [, restricted])                                     -> term STRUCT
//   effects_design(x, terms)                                -> DOUBLE[] design row (no intercept)
//   effects_summary_agg(x, model, terms [, options])        -> LIST(STRUCT) summary.rms
//   contrast_agg(x, model, terms, a, b [, options])         -> STRUCT        contrast.rms(fit, a, b)
//   double_contrast_agg(x, model, terms, a, b, a2, b2 [, options]) -> STRUCT (a - b) - (a2 - b2)
//   predict_grid_agg(x, model, terms, term [, options])     -> LIST(STRUCT)  Predict(fit, term)
//   nomogram_agg(x, lp, model, terms [, options])           -> STRUCT        nomogram(fit)
//   predict_grid_by(source, group_col, x_cols, model, terms, term, ...) table macro (prediction schema)
//
// `terms` is a LIST of term STRUCTs, one per predictor (linear, rcs, lsp,
// categorical) plus the interactions; `x` holds the raw values of the
// predictors in term order (interactions excluded) and the aggregates collect
// them: the datadist limits (quartiles, medians, prediction ranges, modes) come
// from these data, as rms::datadist. `model` is a fit STRUCT with coefficients,
// intercept and vcov (fitted with {'compute_inference': true}) on the design of
// effects_design(x, terms). The model and the terms are taken from the first
// row of a group where they are not NULL. Rows with a NULL x, a NULL / NaN
// element (or a NULL lp) are skipped; no rows give NULL. All statistics are
// computed by anofox-regression (effects::ModelSpec); this file collects rows,
// reads the STRUCTs and writes the results.

#include <cmath>
#include <functional>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_statistics_extension.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/duckdb_compat.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------

LogicalType TermType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("name", LogicalType::VARCHAR));
	c.push_back(MakeChild("kind", LogicalType::VARCHAR));
	c.push_back(MakeChild("knots", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(MakeChild("norm", LogicalType::INTEGER));
	c.push_back(MakeChild("levels", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(MakeChild("a", LogicalType::VARCHAR));
	c.push_back(MakeChild("b", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

const char *const KIND_NAMES[] = {"linear", "rcs", "lsp", "categorical", "interaction", "restricted_interaction"};

int32_t KindCode(const string &raw, const char *fn) {
	auto kind = StringUtil::Lower(raw);
	if (kind == "factor") {
		return ANOFOX_TERM_CATEGORICAL;
	}
	if (kind == "ia" || kind == "%ia%") {
		return ANOFOX_TERM_RESTRICTED_INTERACTION;
	}
	for (int32_t i = 0; i < 6; i++) {
		if (kind == KIND_NAMES[i]) {
			return i;
		}
	}
	throw InvalidInputException("%s: term kind must be 'linear', 'rcs', 'lsp', 'categorical', 'interaction' or "
	                            "'restricted_interaction', got '%s'",
	                            fn, raw);
}

struct Term {
	string name;
	int32_t kind = ANOFOX_TERM_LINEAR;
	int32_t norm = 2;
	vector<double> values; // knots or levels
	string a, b;
	size_t a_idx = 0, b_idx = 0;
};

vector<double> DoubleList(const Value &v, const char *fn, const char *what) {
	vector<double> out;
	if (v.IsNull()) {
		return out;
	}
	auto list = v.DefaultCastAs(LogicalType::LIST(LogicalType::DOUBLE));
	for (auto &e : ListValue::GetChildren(list)) {
		if (e.IsNull()) {
			throw InvalidInputException("%s: %s must not contain NULL", fn, what);
		}
		out.push_back(e.GetValue<double>());
	}
	return out;
}

//! Fields of a STRUCT value by lower-case name.
template <class F>
void ForEachField(const Value &v, F &&f) {
	auto &children = StructValue::GetChildren(v);
	for (idx_t i = 0; i < children.size(); i++) {
		f(StringUtil::Lower(NameString(StructType::GetChildName(v.type(), i))), children[i]);
	}
}

//! One term from a term STRUCT (effects_term / effects_interaction, or a literal
//! with the fields name, kind and optionally knots, norm, levels, a, b, spline).
Term ParseTerm(const Value &v, const char *fn) {
	if (v.IsNull() || v.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: every term must be a STRUCT as returned by effects_term or "
		                            "effects_interaction, got %s",
		                            fn, v.IsNull() ? string("NULL") : v.type().ToString());
	}
	Term t;
	bool have_kind = false;
	Value spline;
	ForEachField(v, [&](const string &key, const Value &val) {
		if (val.IsNull()) {
			return;
		}
		if (key == "name") {
			t.name = val.ToString();
		} else if (key == "kind") {
			t.kind = KindCode(val.ToString(), fn);
			have_kind = true;
		} else if (key == "knots") {
			auto knots = DoubleList(val, fn, "knots");
			if (!knots.empty()) {
				t.values = knots;
			}
		} else if (key == "levels") {
			auto levels = DoubleList(val, fn, "levels");
			if (!levels.empty()) {
				t.values = levels;
			}
		} else if (key == "norm") {
			t.norm = val.DefaultCastAs(LogicalType::INTEGER).GetValue<int32_t>();
		} else if (key == "a") {
			t.a = val.ToString();
		} else if (key == "b") {
			t.b = val.ToString();
		} else if (key == "spline") {
			spline = val;
		}
	});
	if (!spline.IsNull() && spline.type().id() == LogicalTypeId::STRUCT) {
		ForEachField(spline, [&](const string &key, const Value &val) {
			if (val.IsNull()) {
				return;
			}
			if (key == "kind" && !have_kind) {
				t.kind = KindCode(val.ToString(), fn);
				have_kind = true;
			} else if (key == "knots") {
				t.values = DoubleList(val, fn, "knots");
			} else if (key == "norm") {
				t.norm = val.DefaultCastAs(LogicalType::INTEGER).GetValue<int32_t>();
			}
		});
	}
	if (!have_kind) {
		throw InvalidInputException("%s: a term has no kind", fn);
	}
	if (t.kind == ANOFOX_TERM_INTERACTION || t.kind == ANOFOX_TERM_RESTRICTED_INTERACTION) {
		if (t.a.empty() || t.b.empty()) {
			throw InvalidInputException("%s: an interaction term needs the predictor names a and b", fn);
		}
		if (t.name.empty()) {
			t.name = t.a + (t.kind == ANOFOX_TERM_INTERACTION ? " * " : " %ia% ") + t.b;
		}
	} else {
		if (t.name.empty()) {
			throw InvalidInputException("%s: a term has no name", fn);
		}
		if ((t.kind == ANOFOX_TERM_RCS || t.kind == ANOFOX_TERM_LSP) && t.values.empty()) {
			throw InvalidInputException("%s: %s term '%s' needs knots", fn, KIND_NAMES[t.kind], t.name);
		}
		if (t.kind == ANOFOX_TERM_LINEAR) {
			t.values.clear();
		}
	}
	return t;
}

//! The parsed term list: the predictors (non-interaction terms) in order.
struct Terms {
	vector<Term> terms;
	vector<idx_t> predictor_terms; // term index of each predictor
	vector<AnofoxEffectsTerm> ffi;

	idx_t NPredictors() const {
		return predictor_terms.size();
	}
	const string &PredictorName(idx_t k) const {
		return terms[predictor_terms[k]].name;
	}
	bool IsCategorical(idx_t k) const {
		return terms[predictor_terms[k]].kind == ANOFOX_TERM_CATEGORICAL;
	}
	idx_t Predictor(const string &name, const char *fn) const {
		for (idx_t k = 0; k < predictor_terms.size(); k++) {
			if (StringUtil::CIEquals(PredictorName(k), name)) {
				return k;
			}
		}
		string names;
		for (idx_t k = 0; k < predictor_terms.size(); k++) {
			names += (k ? ", " : "") + PredictorName(k);
		}
		throw InvalidInputException("%s: unknown predictor '%s' (the terms define %s)", fn, name, names);
	}
};

Terms ParseTerms(const Value &v, const char *fn) {
	if (v.IsNull() || v.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("%s: terms must be a LIST of term STRUCTs (effects_term, effects_interaction)", fn);
	}
	Terms out;
	for (auto &child : ListValue::GetChildren(v)) {
		out.terms.push_back(ParseTerm(child, fn));
	}
	for (idx_t i = 0; i < out.terms.size(); i++) {
		auto kind = out.terms[i].kind;
		if (kind != ANOFOX_TERM_INTERACTION && kind != ANOFOX_TERM_RESTRICTED_INTERACTION) {
			for (auto k : out.predictor_terms) {
				if (StringUtil::CIEquals(out.terms[k].name, out.terms[i].name)) {
					throw InvalidInputException("%s: duplicate term name '%s'", fn, out.terms[i].name);
				}
			}
			out.predictor_terms.push_back(i);
		}
	}
	if (out.predictor_terms.empty()) {
		throw InvalidInputException("%s: the terms contain no predictor", fn);
	}
	for (auto &t : out.terms) {
		if (t.kind == ANOFOX_TERM_INTERACTION || t.kind == ANOFOX_TERM_RESTRICTED_INTERACTION) {
			t.a_idx = out.Predictor(t.a, fn);
			t.b_idx = out.Predictor(t.b, fn);
		}
	}
	for (auto &t : out.terms) {
		AnofoxEffectsTerm f;
		f.kind = t.kind;
		f.norm = t.norm;
		f.values = t.values.empty() ? nullptr : t.values.data();
		f.n_values = t.values.size();
		f.a = t.a_idx;
		f.b = t.b_idx;
		out.ffi.push_back(f);
	}
	return out;
}

// ---------------------------------------------------------------------------
// Model STRUCT
// ---------------------------------------------------------------------------

struct Model {
	vector<double> beta; // intercept first when intercept
	vector<double> vcov;
	bool intercept = false;
	bool has_vcov = false;
	//! Linear model (residual_std_error present): t reference with df_residual.
	bool linear = false;
	double df_residual = std::nan("");
	string link;
};

Model ParseModel(const Value &v, const char *fn) {
	if (v.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: model must be the STRUCT returned by a *_fit_agg function, got %s", fn,
		                            v.type().ToString());
	}
	Model m;
	vector<double> coefs;
	bool have_coefs = false;
	double intercept = std::nan("");
	ForEachField(v, [&](const string &key, const Value &val) {
		if (key == "coefficients") {
			have_coefs = !val.IsNull();
			if (have_coefs) {
				auto list = val.DefaultCastAs(LogicalType::LIST(LogicalType::DOUBLE));
				for (auto &e : ListValue::GetChildren(list)) {
					coefs.push_back(e.IsNull() ? std::nan("") : e.GetValue<double>());
				}
			}
		} else if (key == "intercept" && !val.IsNull()) {
			intercept = val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} else if (key == "vcov" && !val.IsNull()) {
			auto list = val.DefaultCastAs(LogicalType::LIST(LogicalType::DOUBLE));
			for (auto &e : ListValue::GetChildren(list)) {
				m.vcov.push_back(e.IsNull() ? std::nan("") : e.GetValue<double>());
			}
			m.has_vcov = true;
		} else if (key == "df_residual" && !val.IsNull()) {
			m.df_residual = val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} else if (key == "residual_std_error") {
			m.linear = true;
		} else if (key == "link" && !val.IsNull()) {
			m.link = StringUtil::Lower(val.ToString());
		}
	});
	if (!have_coefs) {
		throw InvalidInputException("%s: model has no 'coefficients' field", fn);
	}
	if (m.has_vcov) {
		idx_t dim = 0;
		while (dim * dim < m.vcov.size()) {
			dim++;
		}
		if (dim * dim != m.vcov.size() || (dim != coefs.size() && dim != coefs.size() + 1)) {
			throw InvalidInputException("%s: the model's vcov (%llu values) does not fit its %llu coefficients", fn,
			                            (unsigned long long)m.vcov.size(), (unsigned long long)coefs.size());
		}
		m.intercept = dim == coefs.size() + 1;
	} else {
		m.intercept = !std::isnan(intercept);
	}
	if (m.intercept) {
		m.beta.push_back(intercept);
	}
	m.beta.insert(m.beta.end(), coefs.begin(), coefs.end());
	return m;
}

// ---------------------------------------------------------------------------
// Options and settings
// ---------------------------------------------------------------------------

template <class F>
void VisitOptions(const Value &options, const char *fn, F &&visit) {
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

using Setting = vector<pair<string, double>>;

Setting ParseSetting(ClientContext &context, Expression &expr, const char *fn, const char *what) {
	RequireResolvedParameters(expr);
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: setting %s must be a constant STRUCT or MAP {predictor: value}", fn, what);
	}
	auto v = UnwrapVariant(ExpressionExecutor::EvaluateScalar(context, expr));
	if (v.IsNull() || (v.type().id() != LogicalTypeId::STRUCT && v.type().id() != LogicalTypeId::MAP)) {
		throw InvalidInputException("%s: setting %s must be a STRUCT or MAP {predictor: value}", fn, what);
	}
	Setting s;
	VisitOptions(v, fn, [&](const string &key, const Value &val) {
		if (val.IsNull()) {
			throw InvalidInputException("%s: setting %s gives NULL for '%s'", fn, what, key);
		}
		s.emplace_back(key, val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>());
	});
	return s;
}

enum class Op : uint8_t { SUMMARY, CONTRAST, DOUBLE_CONTRAST, PREDICT_GRID, NOMOGRAM };
enum class CriticalKind : uint8_t { AUTO, NORMAL, T };

struct EffectsBindData : public FunctionData {
	Op op = Op::SUMMARY;
	const char *fn = "";
	idx_t model_arg = 1, terms_arg = 2;
	bool has_lp = false;
	double level = 0.95;
	CriticalKind critical = CriticalKind::AUTO;
	int8_t ratio = -1; // -1 auto (log / logit link), 0 no, 1 yes
	Setting a, b, a2, b2;
	string term;
	vector<double> values;
	bool simultaneous = false;
	double maxscale = 100.0;
	int64_t nint = 10;
	int32_t fun = 0;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<EffectsBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<EffectsBindData>();
		return op == o.op && level == o.level && critical == o.critical && ratio == o.ratio && a == o.a && b == o.b &&
		       a2 == o.a2 && b2 == o.b2 && term == o.term && values == o.values && simultaneous == o.simultaneous &&
		       maxscale == o.maxscale && nint == o.nint && fun == o.fun;
	}
};

void ParseEffectsOptions(const Value &options, EffectsBindData &bind) {
	const char *fn = bind.fn;
	VisitOptions(options, fn, [&](const string &key, const Value &val) {
		if (val.IsNull()) {
			return;
		}
		bool nomogram = bind.op == Op::NOMOGRAM;
		if (key == "level" && !nomogram) {
			bind.level = val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
			if (!(bind.level > 0.0 && bind.level < 1.0)) {
				throw InvalidInputException("%s: option 'level' must be in (0, 1), got %s", fn, val.ToString());
			}
		} else if (key == "critical" && !nomogram) {
			auto c = StringUtil::Lower(val.ToString());
			if (c == "auto") {
				bind.critical = CriticalKind::AUTO;
			} else if (c == "normal" || c == "z") {
				bind.critical = CriticalKind::NORMAL;
			} else if (c == "t") {
				bind.critical = CriticalKind::T;
			} else {
				throw InvalidInputException("%s: option 'critical' must be 'auto', 'normal' or 't', got '%s'", fn,
				                            val.ToString());
			}
		} else if (key == "ratio" && bind.op == Op::SUMMARY) {
			bind.ratio = val.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>() ? 1 : 0;
		} else if (key == "values" && bind.op == Op::PREDICT_GRID) {
			bind.values = DoubleList(val, fn, "values");
			if (bind.values.empty()) {
				throw InvalidInputException("%s: option 'values' must not be empty", fn);
			}
		} else if (key == "simultaneous" && bind.op == Op::PREDICT_GRID) {
			bind.simultaneous = val.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
		} else if (key == "maxscale" && nomogram) {
			bind.maxscale = val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
			if (!(bind.maxscale > 0.0) || !std::isfinite(bind.maxscale)) {
				throw InvalidInputException("%s: option 'maxscale' must be positive, got %s", fn, val.ToString());
			}
		} else if (key == "nint" && nomogram) {
			bind.nint = val.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
			if (bind.nint < 1 || bind.nint > 1000) {
				throw InvalidInputException("%s: option 'nint' must be between 1 and 1000, got %s", fn, val.ToString());
			}
		} else if (key == "fun" && nomogram) {
			auto f = StringUtil::Lower(val.ToString());
			if (f == "none") {
				bind.fun = 0;
			} else if (f == "plogis" || f == "logistic") {
				bind.fun = 1;
			} else if (f == "exp") {
				bind.fun = 2;
			} else {
				throw InvalidInputException("%s: option 'fun' must be 'none', 'plogis' or 'exp', got '%s'", fn,
				                            val.ToString());
			}
		} else {
			const char *supported = bind.op == Op::SUMMARY        ? "level, critical, ratio"
			                        : bind.op == Op::PREDICT_GRID ? "level, critical, values, simultaneous"
			                        : nomogram                    ? "maxscale, nint, fun"
			                                                      : "level, critical";
			throw InvalidInputException("%s: unknown option '%s' (supported: %s)", fn, key, supported);
		}
	});
}

// ---------------------------------------------------------------------------
// Aggregate state
// ---------------------------------------------------------------------------

struct EffectsState {
	vector<vector<double>> x; // one column per predictor
	vector<double> lp;
	Value model;
	Value terms;
	bool initialized = false;
	bool has_model = false;
	bool has_terms = false;
};

void EffectsInitialize(data_ptr_t state_p) {
	new (state_p) EffectsState();
}

void EffectsDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EffectsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~EffectsState();
	}
}

void EffectsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<EffectsBindData>();
	UnifiedVectorFormat x_data, lp_data, model_data, terms_data, sdata;
	inputs[0].ToUnifiedFormat(count, x_data);
	if (bind.has_lp) {
		inputs[1].ToUnifiedFormat(count, lp_data);
	}
	inputs[bind.model_arg].ToUnifiedFormat(count, model_data);
	inputs[bind.terms_arg].ToUnifiedFormat(count, terms_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto entries = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child_vec = ListVector::GetEntry(inputs[0]);
	auto child = FlatDataMut<double>(child_vec);
	auto &child_validity = FlatValidityMut(child_vec);
	auto lp = bind.has_lp ? UnifiedVectorFormat::GetData<double>(lp_data) : nullptr;
	auto states = (EffectsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (!state.has_model && model_data.validity.RowIsValid(model_data.sel->get_index(i))) {
			state.model = inputs[bind.model_arg].GetValue(i);
			state.has_model = true;
		}
		if (!state.has_terms && terms_data.validity.RowIsValid(terms_data.sel->get_index(i))) {
			state.terms = inputs[bind.terms_arg].GetValue(i);
			state.has_terms = true;
		}
		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx) || ListHasNullElement(child_validity, entries[x_idx])) {
			continue;
		}
		double lp_value = 0.0;
		if (bind.has_lp) {
			auto lp_idx = lp_data.sel->get_index(i);
			if (!lp_data.validity.RowIsValid(lp_idx) || !std::isfinite(lp[lp_idx])) {
				continue;
			}
			lp_value = lp[lp_idx];
		}
		auto &entry = entries[x_idx];
		bool finite = true;
		for (idx_t j = 0; j < entry.length; j++) {
			finite = finite && std::isfinite(child[entry.offset + j]);
		}
		if (!finite) {
			continue;
		}
		if (!state.initialized) {
			state.x.resize(entry.length);
			state.initialized = true;
		} else if (entry.length != state.x.size()) {
			throw InvalidInputException("%s: inconsistent x length: expected %llu, got %llu", bind.fn,
			                            (unsigned long long)state.x.size(), (unsigned long long)entry.length);
		}
		for (idx_t j = 0; j < entry.length; j++) {
			state.x[j].push_back(child[entry.offset + j]);
		}
		if (bind.has_lp) {
			state.lp.push_back(lp_value);
		}
	}
}

void EffectsCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<EffectsBindData>();
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (EffectsState **)sdata.data;
	auto targets = (EffectsState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!target.has_model && source.has_model) {
			target.model = source.model;
			target.has_model = true;
		}
		if (!target.has_terms && source.has_terms) {
			target.terms = source.terms;
			target.has_terms = true;
		}
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.x = CombineTake(source.x, aggr_input_data);
			target.lp = CombineTake(source.lp, aggr_input_data);
			target.initialized = true;
			continue;
		}
		if (target.x.size() != source.x.size()) {
			throw InvalidInputException("%s: inconsistent x length: expected %llu, got %llu", bind.fn,
			                            (unsigned long long)target.x.size(), (unsigned long long)source.x.size());
		}
		for (idx_t j = 0; j < target.x.size(); j++) {
			target.x[j].insert(target.x[j].end(), source.x[j].begin(), source.x[j].end());
		}
		target.lp.insert(target.lp.end(), source.lp.begin(), source.lp.end());
	}
}

//! Everything an FFI call needs; `input` points into the other members.
struct Prepared {
	Terms terms;
	Model model;
	vector<double> data;
	AnofoxEffectsInput input;
};

//! false: NULL result (no rows, no model or no terms).
bool Prepare(EffectsState &state, const EffectsBindData &bind, Prepared &p) {
	if (!state.initialized || !state.has_model || !state.has_terms || state.x.empty() || state.x[0].empty()) {
		return false;
	}
	p.terms = ParseTerms(state.terms, bind.fn);
	if (p.terms.NPredictors() != state.x.size()) {
		throw InvalidInputException("%s: x has %llu values per row but the terms define %llu predictors", bind.fn,
		                            (unsigned long long)state.x.size(), (unsigned long long)p.terms.NPredictors());
	}
	p.model = ParseModel(state.model, bind.fn);
	if (bind.op != Op::NOMOGRAM && !p.model.has_vcov) {
		throw InvalidInputException("%s: the model has no coefficient covariance (vcov is NULL); fit it with "
		                            "{'compute_inference': true}",
		                            bind.fn);
	}
	const idx_t n = state.x[0].size();
	p.data.reserve(n * state.x.size());
	for (auto &col : state.x) {
		p.data.insert(p.data.end(), col.begin(), col.end());
	}
	double df = std::nan("");
	if (bind.critical == CriticalKind::T || (bind.critical == CriticalKind::AUTO && p.model.linear)) {
		df = p.model.df_residual;
		if (bind.critical == CriticalKind::T && !(df > 0.0)) {
			throw InvalidInputException("%s: critical 't' needs the model's df_residual", bind.fn);
		}
	}
	auto &in = p.input;
	in.terms = p.terms.ffi.data();
	in.n_terms = p.terms.ffi.size();
	in.data = p.data.data();
	in.n_rows = n;
	in.beta = p.model.beta.data();
	in.n_beta = p.model.beta.size();
	in.vcov = p.model.has_vcov ? p.model.vcov.data() : nullptr;
	in.intercept = p.model.intercept;
	in.df = df;
	in.level = bind.level;
	return true;
}

//! Calls an FFI function that writes a variable-length result into a buffer,
//! growing the buffer once when the result does not fit.
template <class T, class F>
bool CallWithBuffer(vector<T> &buffer, F &&call) {
	size_t len = 0;
	if (!call(buffer.data(), buffer.size(), &len)) {
		return false;
	}
	if (len > buffer.size()) {
		buffer.resize(len);
		if (!call(buffer.data(), buffer.size(), &len)) {
			return false;
		}
	}
	buffer.resize(len);
	return true;
}

void WriteDouble(Vector &vec, idx_t row, double v) {
	if (std::isfinite(v)) {
		FlatDataMut<double>(vec)[row] = v;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

void WriteDoubleList(Vector &vec, idx_t row, const double *values, idx_t n) {
	auto offset = ListVector::GetListSize(vec);
	ListVector::Reserve(vec, offset + n);
	auto &child = ListVector::GetEntry(vec);
	for (idx_t j = 0; j < n; j++) {
		WriteDouble(child, offset + j, values[j]);
	}
	ListVector::SetListSize(vec, offset + n);
	ListVector::GetData(vec)[row] = list_entry_t(offset, n);
}

string FormatValue(double v) {
	if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
		return std::to_string(static_cast<long long>(v));
	}
	return Value::DOUBLE(v).ToString();
}

vector<size_t> ResolveSetting(const Terms &terms, const Setting &s, vector<double> &values, const char *fn) {
	vector<size_t> idx;
	values.clear();
	for (auto &kv : s) {
		idx.push_back(static_cast<size_t>(terms.Predictor(kv.first, fn)));
		values.push_back(kv.second);
	}
	return idx;
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

LogicalType SummaryRowType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("term", LogicalType::VARCHAR));
	c.push_back(MakeChild("label", LogicalType::VARCHAR));
	c.push_back(MakeChild("low", LogicalType::DOUBLE));
	c.push_back(MakeChild("high", LogicalType::DOUBLE));
	c.push_back(MakeChild("diff", LogicalType::DOUBLE));
	c.push_back(MakeChild("estimate", LogicalType::DOUBLE));
	c.push_back(MakeChild("std_error", LogicalType::DOUBLE));
	c.push_back(MakeChild("conf_low", LogicalType::DOUBLE));
	c.push_back(MakeChild("conf_high", LogicalType::DOUBLE));
	c.push_back(MakeChild("statistic", LogicalType::DOUBLE));
	c.push_back(MakeChild("p_value", LogicalType::DOUBLE));
	c.push_back(MakeChild("ratio", LogicalType::DOUBLE));
	c.push_back(MakeChild("ratio_low", LogicalType::DOUBLE));
	c.push_back(MakeChild("ratio_high", LogicalType::DOUBLE));
	c.push_back(MakeChild("conf_level", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType ContrastType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("estimate", LogicalType::DOUBLE));
	c.push_back(MakeChild("std_error", LogicalType::DOUBLE));
	c.push_back(MakeChild("conf_low", LogicalType::DOUBLE));
	c.push_back(MakeChild("conf_high", LogicalType::DOUBLE));
	c.push_back(MakeChild("statistic", LogicalType::DOUBLE));
	c.push_back(MakeChild("p_value", LogicalType::DOUBLE));
	c.push_back(MakeChild("conf_level", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType GridRowType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("x", LogicalType::DOUBLE));
	c.push_back(MakeChild("yhat", LogicalType::DOUBLE));
	c.push_back(MakeChild("se_fit", LogicalType::DOUBLE));
	c.push_back(MakeChild("yhat_lower", LogicalType::DOUBLE));
	c.push_back(MakeChild("yhat_upper", LogicalType::DOUBLE));
	c.push_back(MakeChild("critical", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType NomogramType() {
	const auto DL = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> axis;
	axis.push_back(MakeChild("term", LogicalType::VARCHAR));
	axis.push_back(MakeChild("values", DL));
	axis.push_back(MakeChild("xbeta", DL));
	axis.push_back(MakeChild("points", DL));
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("axes", LogicalType::LIST(LogicalType::STRUCT(std::move(axis)))));
	c.push_back(MakeChild("scale", LogicalType::DOUBLE));
	c.push_back(MakeChild("intercept", LogicalType::DOUBLE));
	c.push_back(MakeChild("total_points", DL));
	c.push_back(MakeChild("lp_at", DL));
	c.push_back(MakeChild("lp_points", DL));
	c.push_back(MakeChild("fun_at", DL));
	c.push_back(MakeChild("fun_points", DL));
	return LogicalType::STRUCT(std::move(c));
}

void WriteEstimate(StructEntries &fields, idx_t first, idx_t row, const AnofoxContrastEstimate &e) {
	WriteDouble(StructEntry(fields[first]), row, e.estimate);
	WriteDouble(StructEntry(fields[first + 1]), row, e.se);
	WriteDouble(StructEntry(fields[first + 2]), row, e.lower);
	WriteDouble(StructEntry(fields[first + 3]), row, e.upper);
	WriteDouble(StructEntry(fields[first + 4]), row, e.statistic);
	WriteDouble(StructEntry(fields[first + 5]), row, e.p_value);
}

// ---------------------------------------------------------------------------
// Finalize
// ---------------------------------------------------------------------------

void FinalizeSummary(Prepared &p, const EffectsBindData &bind, Vector &result, idx_t row) {
	vector<AnofoxEffectRow> rows(8);
	AnofoxError error;
	if (!CallWithBuffer(rows, [&](AnofoxEffectRow *out, size_t cap, size_t *len) {
		    return anofox_effects_summary(&p.input, out, cap, len, &error);
	    })) {
		ThrowUnlessDegenerate(bind.fn, error);
		FlatVector::SetNull(result, row, true);
		return;
	}
	bool ratio = bind.ratio == 1 || (bind.ratio == -1 && (p.model.link == "logit" || p.model.link == "log"));
	auto n = static_cast<idx_t>(rows.size());
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + n);
	auto &list = ListVector::GetEntry(result);
	auto &f = StructVector::GetEntries(list);
	for (idx_t r = 0; r < n; r++) {
		auto out = offset + r;
		auto &er = rows[r];
		auto &name = p.terms.PredictorName(er.predictor);
		string label = p.terms.IsCategorical(er.predictor)
		                   ? name + " - " + FormatValue(er.high) + ":" + FormatValue(er.low)
		                   : name;
		SetResultString(StructEntry(f[0]), out, name.c_str());
		SetResultString(StructEntry(f[1]), out, label.c_str());
		WriteDouble(StructEntry(f[2]), out, er.low);
		WriteDouble(StructEntry(f[3]), out, er.high);
		WriteDouble(StructEntry(f[4]), out, er.high - er.low);
		WriteEstimate(f, 5, out, er.effect);
		if (ratio) {
			WriteDouble(StructEntry(f[11]), out, std::exp(er.effect.estimate));
			WriteDouble(StructEntry(f[12]), out, std::exp(er.effect.lower));
			WriteDouble(StructEntry(f[13]), out, std::exp(er.effect.upper));
		} else {
			for (idx_t j = 11; j <= 13; j++) {
				FlatVector::SetNull(StructEntry(f[j]), out, true);
			}
		}
		FlatDataMut<double>(StructEntry(f[14]))[out] = bind.level;
	}
	ListVector::SetListSize(result, offset + n);
	ListVector::GetData(result)[row] = list_entry_t(offset, n);
}

void FinalizeContrast(Prepared &p, const EffectsBindData &bind, Vector &result, idx_t row) {
	vector<double> va, vb, va2, vb2;
	auto ia = ResolveSetting(p.terms, bind.a, va, bind.fn);
	auto ib = ResolveSetting(p.terms, bind.b, vb, bind.fn);
	AnofoxEffectsSetting a {ia.data(), va.data(), ia.size()};
	AnofoxEffectsSetting b {ib.data(), vb.data(), ib.size()};
	AnofoxEffectsSetting a2 {}, b2 {};
	bool second = bind.op == Op::DOUBLE_CONTRAST;
	vector<size_t> ia2, ib2;
	if (second) {
		ia2 = ResolveSetting(p.terms, bind.a2, va2, bind.fn);
		ib2 = ResolveSetting(p.terms, bind.b2, vb2, bind.fn);
		a2 = {ia2.data(), va2.data(), ia2.size()};
		b2 = {ib2.data(), vb2.data(), ib2.size()};
	}
	AnofoxContrastEstimate est;
	AnofoxError error;
	if (!anofox_effects_contrast(&p.input, &a, &b, second ? &a2 : nullptr, second ? &b2 : nullptr, &est, &error)) {
		ThrowUnlessDegenerate(bind.fn, error);
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto &f = StructVector::GetEntries(result);
	WriteEstimate(f, 0, row, est);
	FlatDataMut<double>(StructEntry(f[6]))[row] = bind.level;
}

void FinalizeGrid(Prepared &p, const EffectsBindData &bind, Vector &result, idx_t row) {
	auto k = p.terms.Predictor(bind.term, bind.fn);
	size_t cap = bind.values.empty() ? 200 : bind.values.size();
	vector<double> values(cap);
	vector<AnofoxContrastEstimate> est(cap);
	double critical = std::nan("");
	size_t len = 0;
	AnofoxError error;
	auto call = [&]() {
		return anofox_effects_predict_grid(
		    &p.input, static_cast<size_t>(k), bind.values.empty() ? nullptr : bind.values.data(), bind.values.size(),
		    bind.simultaneous, values.data(), est.data(), values.size(), &len, &critical, &error);
	};
	bool ok = call();
	if (ok && len > values.size()) {
		values.resize(len);
		est.resize(len);
		ok = call();
	}
	if (!ok) {
		ThrowUnlessDegenerate(bind.fn, error);
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto n = static_cast<idx_t>(len);
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + n);
	auto &list = ListVector::GetEntry(result);
	auto &f = StructVector::GetEntries(list);
	for (idx_t r = 0; r < n; r++) {
		auto out = offset + r;
		WriteDouble(StructEntry(f[0]), out, values[r]);
		WriteDouble(StructEntry(f[1]), out, est[r].estimate);
		WriteDouble(StructEntry(f[2]), out, est[r].se);
		WriteDouble(StructEntry(f[3]), out, est[r].lower);
		WriteDouble(StructEntry(f[4]), out, est[r].upper);
		WriteDouble(StructEntry(f[5]), out, critical);
	}
	ListVector::SetListSize(result, offset + n);
	ListVector::GetData(result)[row] = list_entry_t(offset, n);
}

void FinalizeNomogram(Prepared &p, EffectsState &state, const EffectsBindData &bind, Vector &result, idx_t row) {
	AnofoxNomogramResult nom;
	AnofoxError error;
	if (!anofox_effects_nomogram(&p.input, state.lp.data(), state.lp.size(), bind.maxscale,
	                             static_cast<size_t>(bind.nint), bind.fun, &nom, &error)) {
		ThrowUnlessDegenerate(bind.fn, error);
		FlatVector::SetNull(result, row, true);
		return;
	}
	auto &f = StructVector::GetEntries(result);
	auto &axes = StructEntry(f[0]);
	auto offset = ListVector::GetListSize(axes);
	ListVector::Reserve(axes, offset + nom.n_axes);
	auto &axis_rows = ListVector::GetEntry(axes);
	auto &af = StructVector::GetEntries(axis_rows);
	for (idx_t a = 0; a < nom.n_axes; a++) {
		auto out = offset + a;
		auto lo = nom.axis_offsets[a];
		auto n = nom.axis_offsets[a + 1] - lo;
		SetResultString(StructEntry(af[0]), out, p.terms.PredictorName(a).c_str());
		WriteDoubleList(StructEntry(af[1]), out, nom.axis_values + lo, n);
		WriteDoubleList(StructEntry(af[2]), out, nom.axis_xbeta + lo, n);
		WriteDoubleList(StructEntry(af[3]), out, nom.axis_points + lo, n);
	}
	ListVector::SetListSize(axes, offset + nom.n_axes);
	ListVector::GetData(axes)[row] = list_entry_t(offset, nom.n_axes);
	WriteDouble(StructEntry(f[1]), row, nom.scale);
	WriteDouble(StructEntry(f[2]), row, nom.intercept);
	WriteDoubleList(StructEntry(f[3]), row, nom.total_points, nom.n_total_points);
	WriteDoubleList(StructEntry(f[4]), row, nom.lp_at, nom.n_lp);
	WriteDoubleList(StructEntry(f[5]), row, nom.lp_points, nom.n_lp);
	if (bind.fun == 0) {
		FlatVector::SetNull(StructEntry(f[6]), row, true);
		FlatVector::SetNull(StructEntry(f[7]), row, true);
	} else {
		WriteDoubleList(StructEntry(f[6]), row, nom.fun_at, nom.n_fun);
		WriteDoubleList(StructEntry(f[7]), row, nom.fun_points, nom.n_fun);
	}
	anofox_free_nomogram(&nom);
}

void EffectsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                     idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<EffectsBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EffectsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		Prepared p;
		if (!Prepare(state, bind, p)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		switch (bind.op) {
		case Op::SUMMARY:
			FinalizeSummary(p, bind, result, row);
			break;
		case Op::CONTRAST:
		case Op::DOUBLE_CONTRAST:
			FinalizeContrast(p, bind, result, row);
			break;
		case Op::PREDICT_GRID:
			FinalizeGrid(p, bind, result, row);
			break;
		case Op::NOMOGRAM:
			FinalizeNomogram(p, state, bind, result, row);
			break;
		}
	}
}

// ---------------------------------------------------------------------------
// Bind
// ---------------------------------------------------------------------------

unique_ptr<FunctionData> EffectsBind(ClientContext &context, AggregateBindFunction &function,
                                     vector<unique_ptr<Expression>> &args, Op op, const char *fn) {
	auto bind = make_uniq<EffectsBindData>();
	bind->op = op;
	bind->fn = fn;
	idx_t n_fixed = 3; // x, model, terms
	if (op == Op::NOMOGRAM) {
		bind->has_lp = true;
		bind->model_arg = 2;
		bind->terms_arg = 3;
		n_fixed = 4;
	} else if (op == Op::CONTRAST) {
		bind->a = ParseSetting(context, *args[3], fn, "a");
		bind->b = ParseSetting(context, *args[4], fn, "b");
		n_fixed = 5;
	} else if (op == Op::DOUBLE_CONTRAST) {
		bind->a = ParseSetting(context, *args[3], fn, "a");
		bind->b = ParseSetting(context, *args[4], fn, "b");
		bind->a2 = ParseSetting(context, *args[5], fn, "a2");
		bind->b2 = ParseSetting(context, *args[6], fn, "b2");
		n_fixed = 7;
	} else if (op == Op::PREDICT_GRID) {
		RequireResolvedParameters(*args[3]);
		if (!args[3]->IsFoldable()) {
			throw InvalidInputException("%s: term must be a constant predictor name", fn);
		}
		auto term = ExpressionExecutor::EvaluateScalar(context, *args[3]);
		if (term.IsNull()) {
			throw InvalidInputException("%s: term must not be NULL", fn);
		}
		bind->term = term.ToString();
		n_fixed = 4;
	}
	if (ReturnTypeOf(*args[bind->model_arg]).id() != LogicalTypeId::STRUCT &&
	    ReturnTypeOf(*args[bind->model_arg]).id() != LogicalTypeId::SQLNULL) {
		throw InvalidInputException("%s: model must be the STRUCT returned by a *_fit_agg function, got %s", fn,
		                            ReturnTypeOf(*args[bind->model_arg]).ToString());
	}
	if (ReturnTypeOf(*args[bind->terms_arg]).id() != LogicalTypeId::LIST &&
	    ReturnTypeOf(*args[bind->terms_arg]).id() != LogicalTypeId::SQLNULL) {
		throw InvalidInputException("%s: terms must be a LIST of term STRUCTs (effects_term, effects_interaction), "
		                            "got %s",
		                            fn, ReturnTypeOf(*args[bind->terms_arg]).ToString());
	}
	if (args.size() > n_fixed) {
		ParseEffectsOptions(EvaluateConstantOptions(context, *args[n_fixed], fn), *bind);
	}
	switch (op) {
	case Op::SUMMARY:
		SetReturnTypeOf(function, LogicalType::LIST(SummaryRowType()));
		break;
	case Op::CONTRAST:
	case Op::DOUBLE_CONTRAST:
		SetReturnTypeOf(function, ContrastType());
		break;
	case Op::PREDICT_GRID:
		SetReturnTypeOf(function, LogicalType::LIST(GridRowType()));
		break;
	case Op::NOMOGRAM:
		SetReturnTypeOf(function, NomogramType());
		break;
	}
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

unique_ptr<FunctionData> SummaryBind(ClientContext &context, AggregateBindFunction &function,
                                     vector<unique_ptr<Expression>> &args) {
	return EffectsBind(context, function, args, Op::SUMMARY, "effects_summary_agg");
}
unique_ptr<FunctionData> ContrastBind(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &args) {
	return EffectsBind(context, function, args, Op::CONTRAST, "contrast_agg");
}
unique_ptr<FunctionData> DoubleContrastBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &args) {
	return EffectsBind(context, function, args, Op::DOUBLE_CONTRAST, "double_contrast_agg");
}
unique_ptr<FunctionData> GridBind(ClientContext &context, AggregateBindFunction &function,
                                  vector<unique_ptr<Expression>> &args) {
	return EffectsBind(context, function, args, Op::PREDICT_GRID, "predict_grid_agg");
}
unique_ptr<FunctionData> NomogramBind(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &args) {
	return EffectsBind(context, function, args, Op::NOMOGRAM, "nomogram_agg");
}

void RegisterEffectsAggregate(ExtensionLoader &loader, const char *name, const vector<LogicalType> &fixed,
                              const vector<string> &names, bind_aggregate_function_t bind, const string &description,
                              const string &example) {
	AggregateFunctionSet set(name);
	for (bool with_options : {false, true}) {
		auto args = fixed;
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    name, args, LogicalType::ANY, AggregateFunction::StateSize<EffectsState>,
		    ANOFOX_AGG_INIT(EffectsInitialize), ANOFOX_GUARDED_UPDATE(EffectsUpdate, EffectsDestroy, EffectsInitialize),
		    EffectsCombine, ANOFOX_GUARDED_FINALIZE(EffectsFinalize, EffectsDestroy, EffectsInitialize), nullptr, bind,
		    EffectsDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	for (bool with_options : {false, true}) {
		FunctionDescription d;
		d.description = description;
		d.examples = {example};
		d.categories = {"regression", "effects"};
		d.parameter_names = names;
		d.parameter_types = fixed;
		if (with_options) {
			d.parameter_names.push_back("options");
			d.parameter_types.push_back(LogicalType::ANY);
		}
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

// ---------------------------------------------------------------------------
// Scalars: effects_term, effects_interaction, effects_design
// ---------------------------------------------------------------------------

Value TermValue(const string &name, int32_t kind, const vector<double> &knots, Value norm, const vector<double> &levels,
                const string &a, const string &b) {
	auto list = [](const vector<double> &v) {
		if (v.empty()) {
			return Value(LogicalType::LIST(LogicalType::DOUBLE));
		}
		vector<Value> vals;
		for (auto d : v) {
			vals.push_back(Value::DOUBLE(d));
		}
		return Value::LIST(LogicalType::DOUBLE, std::move(vals));
	};
	// The typed overload: STRUCT child names are Identifiers on DuckDB 2.0.
	vector<Value> c;
	c.push_back(Value(name));
	c.push_back(Value(KIND_NAMES[kind]));
	c.push_back(list(knots));
	c.push_back(norm);
	c.push_back(list(levels));
	c.push_back(a.empty() ? Value(LogicalType::VARCHAR) : Value(a));
	c.push_back(b.empty() ? Value(LogicalType::VARCHAR) : Value(b));
	return Value::STRUCT(TermType(), std::move(c));
}

template <void (*IMPL)(DataChunk &, ExpressionState &, Vector &)>
void Flattened(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	args.Flatten();
	IMPL(args, state, result);
	result.SetVectorType(all_constant ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
}

void EffectsTermImpl(DataChunk &args, ExpressionState &, Vector &result) {
	const char *fn = "effects_term";
	for (idx_t row = 0; row < args.size(); row++) {
		auto name = args.data[0].GetValue(row);
		auto spec = args.data[1].GetValue(row);
		if (name.IsNull() || spec.IsNull()) {
			result.SetValue(row, Value(TermType()));
			continue;
		}
		vector<double> levels;
		if (args.ColumnCount() >= 3) {
			levels = DoubleList(args.data[2].GetValue(row), fn, "levels");
		}
		if (spec.type().id() == LogicalTypeId::STRUCT) {
			// A spline_spec STRUCT: kind rcs / lsp with its knots (and norm).
			string kind_name;
			vector<double> knots;
			Value norm(LogicalType::INTEGER);
			ForEachField(spec, [&](const string &key, const Value &val) {
				if (val.IsNull()) {
					return;
				}
				if (key == "kind") {
					kind_name = val.ToString();
				} else if (key == "knots") {
					knots = DoubleList(val, fn, "knots");
				} else if (key == "norm") {
					norm = val.DefaultCastAs(LogicalType::INTEGER);
				}
			});
			if (StringUtil::Lower(kind_name) == "bspline") {
				throw InvalidInputException("%s: B-spline terms are not supported by effects (anofox-regression's "
				                            "ModelSpec has linear, rcs, lsp and categorical predictors)",
				                            fn);
			}
			auto kind = KindCode(kind_name, fn);
			if (kind != ANOFOX_TERM_RCS && kind != ANOFOX_TERM_LSP) {
				throw InvalidInputException("%s: a spline spec must have kind 'rcs' or 'lsp', got '%s'", fn, kind_name);
			}
			if (knots.empty()) {
				throw InvalidInputException("%s: the spline spec has no knots", fn);
			}
			if (kind == ANOFOX_TERM_RCS && norm.IsNull()) {
				norm = Value::INTEGER(2);
			}
			if (kind == ANOFOX_TERM_LSP) {
				norm = Value(LogicalType::INTEGER);
			}
			result.SetValue(row, TermValue(name.ToString(), kind, knots, norm, {}, "", ""));
			continue;
		}
		auto kind = KindCode(spec.ToString(), fn);
		if (kind != ANOFOX_TERM_LINEAR && kind != ANOFOX_TERM_CATEGORICAL) {
			throw InvalidInputException("%s: kind '%s' needs knots: pass a spline_spec STRUCT (e.g. "
			                            "effects_term('%s', spline_spec('%s', knots))), or use effects_interaction",
			                            fn, spec.ToString(), name.ToString(), spec.ToString());
		}
		if (!levels.empty() && kind != ANOFOX_TERM_CATEGORICAL) {
			throw InvalidInputException("%s: levels apply only to categorical terms", fn);
		}
		result.SetValue(row, TermValue(name.ToString(), kind, {}, Value(LogicalType::INTEGER), levels, "", ""));
	}
}

void EffectsInteractionImpl(DataChunk &args, ExpressionState &, Vector &result) {
	for (idx_t row = 0; row < args.size(); row++) {
		auto a = args.data[0].GetValue(row);
		auto b = args.data[1].GetValue(row);
		if (a.IsNull() || b.IsNull()) {
			result.SetValue(row, Value(TermType()));
			continue;
		}
		bool restricted = false;
		if (args.ColumnCount() >= 3) {
			auto r = args.data[2].GetValue(row);
			restricted = !r.IsNull() && r.GetValue<bool>();
		}
		auto kind = restricted ? ANOFOX_TERM_RESTRICTED_INTERACTION : ANOFOX_TERM_INTERACTION;
		auto name = a.ToString() + (restricted ? " %ia% " : " * ") + b.ToString();
		result.SetValue(row, TermValue(name, kind, {}, Value(LogicalType::INTEGER), {}, a.ToString(), b.ToString()));
	}
}

unique_ptr<FunctionData> EffectsTermBind(ClientContext &, ScalarBindFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	auto id = ReturnTypeOf(*arguments[1]).id();
	if (id != LogicalTypeId::VARCHAR && id != LogicalTypeId::STRUCT && id != LogicalTypeId::SQLNULL) {
		throw InvalidInputException("effects_term: the second argument must be a kind ('linear', 'categorical') or a "
		                            "spline_spec STRUCT, got %s",
		                            ReturnTypeOf(*arguments[1]).ToString());
	}
	if (arguments.size() >= 3 && id == LogicalTypeId::STRUCT) {
		throw InvalidInputException("effects_term: levels apply only to categorical terms");
	}
	PostHogTelemetry::Instance().RecordFunctionCall("effects_term");
	return nullptr;
}

//! Rows of a chunk sharing one terms value are expanded with one FFI call.
void EffectsDesignImpl(DataChunk &args, ExpressionState &, Vector &result) {
	const char *fn = "effects_design";
	const idx_t count = args.size();
	auto &x_vec = args.data[0];
	auto &terms_vec = args.data[1];
	auto entries = FlatDataMut<list_entry_t>(x_vec);
	auto &child = ListVector::GetEntry(x_vec);
	auto x = FlatDataMut<double>(child);
	auto &validity = FlatValidityMut(child);

	idx_t row = 0;
	while (row < count) {
		if (FlatVector::IsNull(terms_vec, row)) {
			FlatVector::SetNull(result, row, true);
			row++;
			continue;
		}
		// The run of rows with the same terms value.
		auto terms_value = terms_vec.GetValue(row);
		idx_t end = row + 1;
		while (end < count && !FlatVector::IsNull(terms_vec, end) && terms_vec.GetValue(end) == terms_value) {
			end++;
		}
		auto terms = ParseTerms(terms_value, fn);
		const idx_t p = terms.NPredictors();
		vector<idx_t> rows;
		for (idx_t r = row; r < end; r++) {
			if (FlatVector::IsNull(x_vec, r) || ListHasNullElement(validity, entries[r])) {
				FlatVector::SetNull(result, r, true);
				continue;
			}
			if (entries[r].length != p) {
				throw InvalidInputException("%s: x has %llu values but the terms define %llu predictors", fn,
				                            (unsigned long long)entries[r].length, (unsigned long long)p);
			}
			bool finite = true;
			for (idx_t j = 0; j < p; j++) {
				finite = finite && std::isfinite(x[entries[r].offset + j]);
			}
			if (!finite) {
				FlatVector::SetNull(result, r, true);
				continue;
			}
			rows.push_back(r);
		}
		row = end;
		if (rows.empty()) {
			continue;
		}
		const idx_t n = rows.size();
		vector<double> data(n * p);
		for (idx_t i = 0; i < n; i++) {
			for (idx_t j = 0; j < p; j++) {
				data[j * n + i] = x[entries[rows[i]].offset + j];
			}
		}
		AnofoxEffectsInput in {};
		in.terms = terms.ffi.data();
		in.n_terms = terms.ffi.size();
		in.data = data.data();
		in.n_rows = n;
		in.level = 0.95;
		in.df = std::nan("");
		vector<double> out(n * 8);
		size_t n_columns = 0;
		AnofoxError error;
		if (!CallWithBuffer(out, [&](double *buf, size_t cap, size_t *len) {
			    return anofox_effects_design(&in, buf, cap, len, &n_columns, &error);
		    })) {
			ThrowFromFfiError(fn, error);
		}
		for (idx_t i = 0; i < n; i++) {
			WriteDoubleList(result, rows[i], out.data() + i * n_columns, n_columns);
		}
	}
}

void RegisterScalar(ExtensionLoader &loader, const char *name, vector<ScalarFunction> overloads,
                    const string &description, const vector<string> &examples,
                    const vector<vector<string>> &parameter_names) {
	ScalarFunctionSet set(name);
	for (auto &f : overloads) {
		MarkFallible(f);
		set.AddFunction(f);
	}
	CreateScalarFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	for (idx_t i = 0; i < overloads.size(); i++) {
		FunctionDescription desc;
		desc.description = description;
		desc.examples = {examples[std::min<idx_t>(i, examples.size() - 1)]};
		desc.categories = {"regression", "effects"};
		desc.parameter_names = parameter_names[i];
		desc.parameter_types = ArgumentTypesOf(overloads[i]);
		info.descriptions.push_back(std::move(desc));
	}
	loader.RegisterFunction(std::move(info));
}

// predict_grid_by: Predict() grids per group in the integration contract's
// `prediction` schema (as <model>_effects_by).
const char *const PREDICT_GRID_BY_SQL =
    "SELECT group_col, group_col::VARCHAR AS id, 'predict_grid' AS model_id, 'predict_grid' AS model_name, "
    "_focal AS focal, (_p).x AS x, NULL::DOUBLE AS y, (_p).yhat AS yhat, (_p).yhat_lower AS yhat_lower, "
    "(_p).yhat_upper AS yhat_upper, level::DOUBLE AS level, NULL::VARCHAR AS split, (_p).se_fit AS se_fit, "
    "term::VARCHAR AS focal_name "
    "FROM (SELECT group_col, _focal, unnest(_g) AS _p FROM (SELECT group_col, "
    "list_position(list_transform(list_filter(any_value(terms), lambda _t: _t.kind NOT IN ('interaction', "
    "'restricted_interaction')), lambda _t: lower(_t.name)), lower(term::VARCHAR))::BIGINT AS _focal, "
    "predict_grid_agg(x_cols, model, terms, term, {'level': level, 'simultaneous': simultaneous, "
    "'values': grid_values}) AS _g FROM query_table(source::VARCHAR) GROUP BY group_col)) "
    "ORDER BY group_col, x";

} // namespace

void RegisterEffectsFunctions(ExtensionLoader &loader) {
	const auto DL = LogicalType::LIST(LogicalType::DOUBLE);
	const auto V = LogicalType::VARCHAR;
	const auto ANY = LogicalType::ANY;
	const auto D = LogicalType::DOUBLE;

	RegisterScalar(
	    loader, "effects_term",
	    {ScalarFunction({V, ANY}, TermType(), Flattened<EffectsTermImpl>, ANOFOX_SCALAR_BIND(EffectsTermBind)),
	     ScalarFunction({V, V, DL}, TermType(), Flattened<EffectsTermImpl>, ANOFOX_SCALAR_BIND(EffectsTermBind))},
	    "A predictor term of an effects model specification: effects_term(name, 'linear' | 'categorical' [, levels]) "
	    "or effects_term(name, spline_spec('rcs' | 'lsp', knots)). Returns {name, kind, knots, norm, levels, a, b}. "
	    "A categorical term is coded with treatment contrasts (a dummy per level but the first); its levels are "
	    "required by effects_design.",
	    {"effects_term('age', spline_spec('rcs', [30.0, 45.0, 60.0]))", "effects_term('sex', 'categorical', [0, 1])"},
	    {{"name", "kind"}, {"name", "kind", "levels"}});

	RegisterScalar(loader, "effects_interaction",
	               {ScalarFunction({V, V}, TermType(), Flattened<EffectsInteractionImpl>),
	                ScalarFunction({V, V, LogicalType::BOOLEAN}, TermType(), Flattened<EffectsInteractionImpl>)},
	               "An interaction term between the predictors a and b: the full tensor product (rms a * b), or with "
	               "restricted = true the restricted interaction a %ia% b (no doubly nonlinear products).",
	               {"effects_interaction('age', 'sex')", "effects_interaction('age', 'bmi', true)"},
	               {{"a", "b"}, {"a", "b", "restricted"}});

	RegisterScalar(loader, "effects_design", {ScalarFunction({DL, ANY}, DL, Flattened<EffectsDesignImpl>)},
	               "Design row (no intercept) of the raw predictor values x (one per predictor term, in term order) "
	               "for an effects term list: the main-effect bases in term order, then the interactions. Fit a model "
	               "on it to use effects_summary_agg, contrast_agg, predict_grid_agg and nomogram_agg.",
	               {"effects_design([age, sex], [effects_term('age', spline_spec('rcs', [30.0, 45.0, 60.0])), "
	                "effects_term('sex', 'categorical', [0, 1])])"},
	               {{"x", "terms"}});

	const vector<LogicalType> base = {DL, ANY, ANY};
	RegisterEffectsAggregate(
	    loader, "effects_summary_agg", base, {"x", "model", "terms"}, ANOFOX_AGG_BIND(SummaryBind),
	    "Partial effects as rms summary(fit): for each numeric predictor the effect of going from its lower to its "
	    "upper quartile, for each categorical predictor every level against the most frequent one, others at their "
	    "medians / modes (datadist of the rows of x). One element per effect: term, label, low, high, diff, "
	    "estimate, std_error, conf_low, conf_high, statistic, p_value, ratio, ratio_low, ratio_high (exp: odds / "
	    "rate ratios, for log / logit links or with ratio := true), conf_level. Options: level (0.95), critical "
	    "('auto': t for linear models, normal otherwise | 'normal' | 't'), ratio.",
	    "effects_summary_agg([age, sex], m, terms)");
	RegisterEffectsAggregate(
	    loader, "contrast_agg", {DL, ANY, ANY, ANY, ANY}, {"x", "model", "terms", "a", "b"},
	    ANOFOX_AGG_BIND(ContrastBind),
	    "Contrast between two predictor settings, as rms contrast(fit, a, b): the difference of the linear "
	    "predictor at a and at b (STRUCTs or MAPs {predictor: value}; unset predictors at their datadist adjust-to "
	    "values) with estimate, std_error, conf_low, conf_high, statistic, p_value, conf_level. Options: level, "
	    "critical.",
	    "contrast_agg([age, sex], m, terms, {'age': 60}, {'age': 45})");
	RegisterEffectsAggregate(
	    loader, "double_contrast_agg", {DL, ANY, ANY, ANY, ANY, ANY, ANY},
	    {"x", "model", "terms", "a", "b", "a2", "b2"}, ANOFOX_AGG_BIND(DoubleContrastBind),
	    "Double-difference contrast (a - b) - (a2 - b2), as rms contrast(fit, a, b, a2, b2), e.g. an interaction "
	    "effect. Same result as contrast_agg. Options: level, critical.",
	    "double_contrast_agg([age, sex], m, terms, {'age': 60, 'sex': 1}, {'age': 45, 'sex': 1}, {'age': 60, "
	    "'sex': 0}, {'age': 45, 'sex': 0})");
	RegisterEffectsAggregate(
	    loader, "predict_grid_agg", {DL, ANY, ANY, V}, {"x", "model", "terms", "term"}, ANOFOX_AGG_BIND(GridBind),
	    "Partial effect of one predictor, as rms Predict(fit, term): the linear predictor over a grid (200 points "
	    "between the datadist prediction limits, the levels of a categorical predictor, or option values) with the "
	    "other predictors at their adjust-to values. One element per point: x, yhat, se_fit, yhat_lower, "
	    "yhat_upper, critical (simultaneous bands only). Options: level, critical, values, simultaneous (bands "
	    "simultaneous over the grid, Predict(conf.type = 'simultaneous')).",
	    "predict_grid_agg([age, sex], m, terms, 'age', {'values': [30, 40, 50]})");
	RegisterEffectsAggregate(
	    loader, "nomogram_agg", {DL, D, ANY, ANY}, {"x", "lp", "model", "terms"}, ANOFOX_AGG_BIND(NomogramBind),
	    "Nomogram of a main-effects model, as rms nomogram(fit): per predictor the tick values, their terms of the "
	    "linear predictor (xbeta) and points; the points scale, the linear predictor at zero points (intercept), the "
	    "total-points and linear-predictor axes and, with fun, a function axis (fun_at, fun_points). lp is the "
	    "fit's linear predictor of each row. Options: maxscale (100), nint (10), fun ('none' | 'plogis' | 'exp').",
	    "nomogram_agg([age, sex], predict(m, effects_design([age, sex], terms), {'type': 'link'}), m, terms, "
	    "{'fun': 'plogis'})");

	auto macro = CreateContractTableMacro(
	    "predict_grid_by", {"source", "group_col", "x_cols", "model", "terms", "term"},
	    {{"grid_values", "NULL"}, {"simultaneous", "false"}, {"level", "0.95"}}, PREDICT_GRID_BY_SQL,
	    "Table macro: the partial effect of predictor term per group of source (predict_grid_agg, rms Predict) in "
	    "the integration contract's `prediction` schema: the group column, id, model_id and model_name "
	    "('predict_grid'), focal (the predictor's 1-based position), x, y (NULL), yhat, yhat_lower, yhat_upper, "
	    "level, split (NULL), se_fit, focal_name (term). model and terms may be columns of source (e.g. joined per "
	    "group) or constants; grid_values replaces the default grid, simultaneous gives simultaneous bands.",
	    "predict_grid_by('patients_with_model', site, [age, sex], m, terms, 'age')",
	    {"regression", "effects", "table-macro"});
	loader.RegisterFunction(*macro);
}

} // namespace duckdb
