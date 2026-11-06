// Penalized maximum likelihood a la rms (#323):
//
//   pentrace_agg(y, x, options) -> LIST of STRUCT rows, one per penalty: the
//     penalty components (simple, nonlinear, interaction, nonlinear_interaction,
//     resolved as rms does), the effective d.f., the model LR chi-square, AIC,
//     BIC, corrected AIC and is_best (rms::pentrace's results.all and penalty).
//     model 'ols' (rms ols), 'logistic' (binary lrm) or 'ordinal' (lrm / orm
//     with a cumulative link).
//   penalized_ols_fit_agg(y, x [, options]) -> STRUCT: rms::ols(penalty = ):
//     coefficients, vcov (var.penalty 'simple' or 'sandwich'), Sigma, R2,
//     Model L.R., effective d.f. (total, per coefficient, per term).
//
// The term map (terms, nonlinear, interactions as wald_anova, #311, plus
// categorical) assigns every column its penalty type. Everything is computed by
// anofox_regression (solvers::pentrace) through the FFI; this file collects the
// rows of a group, parses the options and lays the result out. Rows with a NULL y,
// a NULL x or a NULL list element are skipped; a group without usable rows gives
// NULL.

#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "aggregate_finalize_guard.hpp"
#include "canonical_order.hpp"
#include "error_dispatch.hpp"
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

namespace {

constexpr const char *kPentrace = "pentrace_agg";
constexpr const char *kPenalizedOls = "penalized_ols_fit_agg";

//! The term specification of the columns (as wald_anova's terms and options).
struct PenaltyTermSpec {
	bool given = false;
	vector<int64_t> term_of_column;
	vector<string> names;
	vector<bool> nonlinear; // empty: all linear
	vector<std::pair<string, std::pair<string, string>>> interactions;
	vector<string> categorical;

	bool operator==(const PenaltyTermSpec &o) const {
		return given == o.given && term_of_column == o.term_of_column && names == o.names &&
		       nonlinear == o.nonlinear && interactions == o.interactions && categorical == o.categorical;
	}
};

struct PentraceBindData : public FunctionData {
	bool trace = true;
	PenaltyTermSpec spec;
	// pentrace_agg
	vector<double> simple, nonlinear, interaction, nonlinear_interaction;
	int32_t model = 0;
	int32_t link = 0;
	int32_t criterion = 0;
	// penalized_ols_fit_agg (NaN: rms fallback)
	double pen_simple = 0.0;
	double pen_nonlinear = std::numeric_limits<double>::quiet_NaN();
	double pen_interaction = std::numeric_limits<double>::quiet_NaN();
	double pen_nonlinear_interaction = std::numeric_limits<double>::quiet_NaN();
	bool sandwich = false;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PentraceBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PentraceBindData>();
		auto same = [](double a, double b) {
			return a == b || (std::isnan(a) && std::isnan(b));
		};
		return trace == o.trace && spec == o.spec && simple == o.simple && nonlinear == o.nonlinear &&
		       interaction == o.interaction && nonlinear_interaction == o.nonlinear_interaction &&
		       model == o.model && link == o.link && criterion == o.criterion && same(pen_simple, o.pen_simple) &&
		       same(pen_nonlinear, o.pen_nonlinear) && same(pen_interaction, o.pen_interaction) &&
		       same(pen_nonlinear_interaction, o.pen_nonlinear_interaction) && sandwich == o.sandwich;
	}
};

//===--------------------------------------------------------------------===//
// Options
//===--------------------------------------------------------------------===//

template <class F>
void VisitBag(const char *fn, const char *what, const Value &bag, F &&visit) {
	if (bag.IsNull()) {
		return;
	}
	if (bag.type().id() == LogicalTypeId::STRUCT) {
		auto &vals = StructValue::GetChildren(bag);
		for (idx_t i = 0; i < vals.size(); i++) {
			visit(StringUtil::Lower(NameString(StructType::GetChildName(bag.type(), i))), UnwrapVariant(vals[i]));
		}
	} else if (bag.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(bag)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv.size() != 2 || kv[0].IsNull()) {
				throw InvalidInputException("%s: invalid MAP entry in %s", fn, what);
			}
			visit(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else {
		throw InvalidInputException("%s: %s must be a MAP or STRUCT, got %s", fn, what, bag.type().ToString());
	}
}

double AsDouble(const char *fn, const string &key, const Value &v) {
	if (v.IsNull()) {
		throw InvalidInputException("%s: option '%s' must not be NULL", fn, key);
	}
	try {
		return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
	}
}

//! A number or a list of numbers.
vector<double> AsDoubles(const char *fn, const string &key, const Value &v) {
	vector<double> out;
	if (v.IsNull()) {
		return out;
	}
	if (v.type().id() == LogicalTypeId::LIST || v.type().id() == LogicalTypeId::ARRAY) {
		auto &children =
		    v.type().id() == LogicalTypeId::LIST ? ListValue::GetChildren(v) : ArrayValue::GetChildren(v);
		for (auto &c : children) {
			out.push_back(AsDouble(fn, key, UnwrapVariant(c)));
		}
		return out;
	}
	out.push_back(AsDouble(fn, key, v));
	return out;
}

vector<string> AsStrings(const char *fn, const string &key, const Value &v, bool allow_null_elements) {
	vector<string> out;
	if (v.IsNull()) {
		return out;
	}
	if (v.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("%s: option '%s' must be a VARCHAR[]", fn, key);
	}
	for (auto &c : ListValue::GetChildren(v)) {
		auto u = UnwrapVariant(c);
		if (u.IsNull()) {
			if (!allow_null_elements) {
				throw InvalidInputException("%s: option '%s' has a NULL element", fn, key);
			}
			out.emplace_back();
			continue;
		}
		out.push_back(u.ToString());
	}
	return out;
}

std::pair<string, string> ParseFactors(const char *fn, const string &label, const Value &v) {
	if (v.IsNull() || v.type().id() != LogicalTypeId::LIST || ListValue::GetChildren(v).size() != 2) {
		throw InvalidInputException("%s: interaction '%s' must map to a list of its two factor labels", fn, label);
	}
	auto &f = ListValue::GetChildren(v);
	if (f[0].IsNull() || f[1].IsNull()) {
		throw InvalidInputException("%s: interaction '%s' has a NULL factor label", fn, label);
	}
	return {UnwrapVariant(f[0]).ToString(), UnwrapVariant(f[1]).ToString()};
}

//! The term keys shared by both aggregates; false if `key` is not one of them.
bool ParseTermOption(const char *fn, const string &key, const Value &val, PenaltyTermSpec &spec) {
	if (key == "terms") {
		if (val.IsNull()) {
			return true;
		}
		if (val.type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("%s: option 'terms' must be a VARCHAR[] with one label per column", fn);
		}
		spec.given = true;
		std::unordered_map<string, int64_t> index;
		for (auto &c : ListValue::GetChildren(val)) {
			auto u = UnwrapVariant(c);
			if (u.IsNull()) {
				throw InvalidInputException("%s: option 'terms' has a NULL label; every column needs a term (its "
				                            "penalty type comes from the term)",
				                            fn);
			}
			auto label = u.ToString();
			auto it = index.find(label);
			if (it == index.end()) {
				it = index.emplace(label, static_cast<int64_t>(spec.names.size())).first;
				spec.names.push_back(label);
			}
			spec.term_of_column.push_back(it->second);
		}
		if (spec.names.empty()) {
			throw InvalidInputException("%s: option 'terms' is empty", fn);
		}
		return true;
	}
	if (key == "nonlinear") {
		if (val.IsNull()) {
			return true;
		}
		if (val.type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("%s: option 'nonlinear' must be a BOOLEAN[] with one flag per column", fn);
		}
		spec.nonlinear.clear();
		for (auto &b : ListValue::GetChildren(val)) {
			auto u = UnwrapVariant(b);
			spec.nonlinear.push_back(!u.IsNull() && u.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>());
		}
		return true;
	}
	if (key == "interactions") {
		if (val.IsNull()) {
			return true;
		}
		if (val.type().id() == LogicalTypeId::MAP) {
			for (auto &entry : MapValue::GetChildren(val)) {
				auto &kv = StructValue::GetChildren(entry);
				auto label = kv[0].ToString();
				spec.interactions.emplace_back(label, ParseFactors(fn, label, UnwrapVariant(kv[1])));
			}
		} else if (val.type().id() == LogicalTypeId::STRUCT) {
			auto &vals = StructValue::GetChildren(val);
			for (idx_t i = 0; i < vals.size(); i++) {
				auto label = NameString(StructType::GetChildName(val.type(), i));
				spec.interactions.emplace_back(label, ParseFactors(fn, label, UnwrapVariant(vals[i])));
			}
		} else {
			throw InvalidInputException("%s: option 'interactions' must be a MAP from an interaction label to its "
			                            "two factor labels, e.g. MAP {'age * sex': ['age', 'sex']}",
			                            fn);
		}
		return true;
	}
	if (key == "categorical") {
		spec.categorical = AsStrings(fn, key, val, false);
		return true;
	}
	return false;
}

void ValidateTermSpec(const char *fn, const PenaltyTermSpec &spec) {
	if ((!spec.interactions.empty() || !spec.categorical.empty()) && !spec.given) {
		throw InvalidInputException("%s: options 'interactions' and 'categorical' need term labels ('terms')", fn);
	}
	if (spec.given && !spec.nonlinear.empty() && spec.nonlinear.size() != spec.term_of_column.size()) {
		throw InvalidInputException("%s: nonlinear has %llu flags but terms has %llu labels", fn,
		                            (unsigned long long)spec.nonlinear.size(),
		                            (unsigned long long)spec.term_of_column.size());
	}
	auto known = [&](const string &name) {
		for (auto &n : spec.names) {
			if (n == name) {
				return true;
			}
		}
		return false;
	};
	for (auto &ia : spec.interactions) {
		for (auto *name : {&ia.first, &ia.second.first, &ia.second.second}) {
			if (!known(*name)) {
				throw InvalidInputException("%s: interaction label '%s' is not one of the term labels", fn, *name);
			}
		}
	}
	for (auto &c : spec.categorical) {
		if (!known(c)) {
			throw InvalidInputException("%s: categorical label '%s' is not one of the term labels", fn, c);
		}
	}
}

int32_t ParseLink(const char *fn, const Value &v) {
	auto s = StringUtil::Lower(v.ToString());
	if (s == "logistic" || s == "logit") {
		return 0;
	}
	if (s == "probit") {
		return 1;
	}
	if (s == "loglog") {
		return 2;
	}
	if (s == "cloglog") {
		return 3;
	}
	if (s == "cauchit") {
		return 4;
	}
	throw InvalidInputException("%s: option 'link' must be 'logistic', 'probit', 'loglog', 'cloglog' or 'cauchit', "
	                            "got '%s'",
	                            fn, v.ToString());
}

//! penalties: a list of (simple) penalties, or a MAP / STRUCT of lists by type.
void ParsePenalties(PentraceBindData &bind, const Value &v) {
	if (v.IsNull()) {
		return;
	}
	if (v.type().id() == LogicalTypeId::STRUCT || v.type().id() == LogicalTypeId::MAP) {
		VisitBag(kPentrace, "option 'penalties'", v, [&](const string &key, const Value &val) {
			if (key == "simple") {
				bind.simple = AsDoubles(kPentrace, "penalties.simple", val);
			} else if (key == "nonlinear") {
				bind.nonlinear = AsDoubles(kPentrace, "penalties.nonlinear", val);
			} else if (key == "interaction") {
				bind.interaction = AsDoubles(kPentrace, "penalties.interaction", val);
			} else if (key == "nonlinear_interaction" || key == "nonlinear.interaction") {
				bind.nonlinear_interaction = AsDoubles(kPentrace, "penalties.nonlinear_interaction", val);
			} else {
				throw InvalidInputException("%s: unknown penalty type '%s' (supported: simple, nonlinear, interaction, "
				                            "nonlinear_interaction)",
				                            kPentrace, key);
			}
		});
		if (bind.simple.empty()) {
			bind.simple = {0.0};
		}
		return;
	}
	bind.simple = AsDoubles(kPentrace, "penalties", v);
}

//! penalty: a number, or a MAP / STRUCT of numbers by type.
void ParsePenalty(PentraceBindData &bind, const Value &v) {
	if (v.IsNull()) {
		return;
	}
	if (v.type().id() == LogicalTypeId::STRUCT || v.type().id() == LogicalTypeId::MAP) {
		VisitBag(kPenalizedOls, "option 'penalty'", v, [&](const string &key, const Value &val) {
			if (key == "simple") {
				bind.pen_simple = AsDouble(kPenalizedOls, "penalty.simple", val);
			} else if (key == "nonlinear") {
				bind.pen_nonlinear = AsDouble(kPenalizedOls, "penalty.nonlinear", val);
			} else if (key == "interaction") {
				bind.pen_interaction = AsDouble(kPenalizedOls, "penalty.interaction", val);
			} else if (key == "nonlinear_interaction" || key == "nonlinear.interaction") {
				bind.pen_nonlinear_interaction = AsDouble(kPenalizedOls, "penalty.nonlinear_interaction", val);
			} else {
				throw InvalidInputException("%s: unknown penalty type '%s' (supported: simple, nonlinear, interaction, "
				                            "nonlinear_interaction)",
				                            kPenalizedOls, key);
			}
		});
		return;
	}
	bind.pen_simple = AsDouble(kPenalizedOls, "penalty", v);
}

void ParseOptions(PentraceBindData &bind, const Value &options) {
	const char *fn = bind.trace ? kPentrace : kPenalizedOls;
	bool link_given = false;
	VisitBag(fn, "options", options, [&](const string &key, const Value &val) {
		if (ParseTermOption(fn, key, val, bind.spec)) {
			return;
		}
		if (bind.trace) {
			if (key == "penalties" || key == "penalty") {
				ParsePenalties(bind, val);
			} else if (key == "model") {
				auto m = StringUtil::Lower(val.IsNull() ? string("ols") : val.ToString());
				if (m == "ols" || m == "gaussian") {
					bind.model = 0;
				} else if (m == "logistic" || m == "binary" || m == "lrm") {
					bind.model = 1;
				} else if (m == "ordinal" || m == "orm") {
					bind.model = 2;
				} else {
					throw InvalidInputException("%s: option 'model' must be 'ols', 'logistic' or 'ordinal', got '%s'",
					                            fn, val.ToString());
				}
			} else if (key == "link" || key == "family") {
				if (!val.IsNull()) {
					bind.link = ParseLink(fn, val);
					link_given = true;
				}
			} else if (key == "criterion" || key == "which") {
				auto c = StringUtil::Lower(val.IsNull() ? string("aic_c") : val.ToString());
				if (c == "aic_c" || c == "aic.c" || c == "aicc") {
					bind.criterion = 0;
				} else if (c == "aic") {
					bind.criterion = 1;
				} else if (c == "bic") {
					bind.criterion = 2;
				} else {
					throw InvalidInputException("%s: option 'criterion' must be 'aic_c', 'aic' or 'bic', got '%s'", fn,
					                            val.ToString());
				}
			} else {
				throw InvalidInputException("%s: unknown option '%s' (supported: penalties, model, link, criterion, "
				                            "terms, nonlinear, interactions, categorical)",
				                            fn, key);
			}
			return;
		}
		if (key == "penalty") {
			ParsePenalty(bind, val);
		} else if (key == "var_penalty" || key == "var.penalty") {
			auto s = StringUtil::Lower(val.IsNull() ? string("simple") : val.ToString());
			if (s == "simple") {
				bind.sandwich = false;
			} else if (s == "sandwich") {
				bind.sandwich = true;
			} else {
				throw InvalidInputException("%s: option 'var_penalty' must be 'simple' or 'sandwich', got '%s'", fn,
				                            val.ToString());
			}
		} else {
			throw InvalidInputException("%s: unknown option '%s' (supported: penalty, var_penalty, terms, nonlinear, "
			                            "interactions, categorical)",
			                            fn, key);
		}
	});
	if (bind.trace && link_given && bind.model == 0) {
		throw InvalidInputException("%s: option 'link' applies to model 'logistic' and 'ordinal'", fn);
	}
	ValidateTermSpec(fn, bind.spec);
}

//===--------------------------------------------------------------------===//
// Result types
//===--------------------------------------------------------------------===//

LogicalType PentraceRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("simple", LogicalType::DOUBLE));
	c.push_back(make_pair("nonlinear", LogicalType::DOUBLE));
	c.push_back(make_pair("interaction", LogicalType::DOUBLE));
	c.push_back(make_pair("nonlinear_interaction", LogicalType::DOUBLE));
	c.push_back(make_pair("df", LogicalType::DOUBLE));
	c.push_back(make_pair("lr_chisq", LogicalType::DOUBLE));
	c.push_back(make_pair("aic", LogicalType::DOUBLE));
	c.push_back(make_pair("bic", LogicalType::DOUBLE));
	c.push_back(make_pair("aic_c", LogicalType::DOUBLE));
	c.push_back(make_pair("is_best", LogicalType::BOOLEAN));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType PenalizedOlsType() {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", dlist));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("std_errors", dlist));
	c.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
	c.push_back(make_pair("vcov", dlist));
	c.push_back(make_pair("var_penalty", LogicalType::VARCHAR));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("sigma", LogicalType::DOUBLE));
	c.push_back(make_pair("lr_chisq", LogicalType::DOUBLE));
	c.push_back(make_pair("effective_df", LogicalType::DOUBLE));
	c.push_back(make_pair("effective_df_diagonal", dlist));
	c.push_back(make_pair("terms", LogicalType::LIST(LogicalType::VARCHAR)));
	c.push_back(make_pair("effective_df_by_term", dlist));
	c.push_back(make_pair("simple", LogicalType::DOUBLE));
	c.push_back(make_pair("nonlinear", LogicalType::DOUBLE));
	c.push_back(make_pair("interaction", LogicalType::DOUBLE));
	c.push_back(make_pair("nonlinear_interaction", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("model_type", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//

//! The C view of the term specification for `p` columns (kept alive by this object).
struct TermsCall {
	vector<int64_t> term_of_column;
	vector<string> names;
	vector<const char *> name_ptrs;
	std::unique_ptr<bool[]> nonlinear;
	std::unique_ptr<bool[]> categorical;
	vector<int64_t> factors;
	AnofoxPenaltyTerms c {};

	TermsCall(const char *fn, const PenaltyTermSpec &spec, idx_t p) {
		if (spec.given) {
			if (spec.term_of_column.size() != p) {
				throw InvalidInputException("%s: terms has %llu labels but x has %llu columns", fn,
				                            (unsigned long long)spec.term_of_column.size(), (unsigned long long)p);
			}
			term_of_column = spec.term_of_column;
			names = spec.names;
		} else {
			for (idx_t j = 0; j < p; j++) {
				term_of_column.push_back(static_cast<int64_t>(j));
				names.push_back("x" + std::to_string(j + 1));
			}
		}
		if (!spec.nonlinear.empty() && spec.nonlinear.size() != p) {
			throw InvalidInputException("%s: nonlinear has %llu flags but x has %llu columns", fn,
			                            (unsigned long long)spec.nonlinear.size(), (unsigned long long)p);
		}
		const idx_t n_terms = names.size();
		auto index_of = [&](const string &name) {
			for (idx_t t = 0; t < n_terms; t++) {
				if (names[t] == name) {
					return static_cast<int64_t>(t);
				}
			}
			return int64_t(-1);
		};
		factors.assign(2 * n_terms, -1);
		for (auto &ia : spec.interactions) {
			auto t = index_of(ia.first);
			factors[2 * t] = index_of(ia.second.first);
			factors[2 * t + 1] = index_of(ia.second.second);
		}
		nonlinear.reset(new bool[p > 0 ? p : 1]);
		for (idx_t j = 0; j < p; j++) {
			nonlinear[j] = !spec.nonlinear.empty() && spec.nonlinear[j];
		}
		categorical.reset(new bool[n_terms > 0 ? n_terms : 1]);
		for (idx_t t = 0; t < n_terms; t++) {
			categorical[t] = false;
		}
		for (auto &name : spec.categorical) {
			categorical[index_of(name)] = true;
		}
		for (auto &n : names) {
			name_ptrs.push_back(n.c_str());
		}
		c.term_of_column = term_of_column.data();
		c.nonlinear = nonlinear.get();
		c.term_names = name_ptrs.data();
		c.n_terms = n_terms;
		c.interaction_factors = factors.data();
		c.categorical = categorical.get();
	}
};

//! Reserve `n` rows of a LIST(STRUCT) result for `row` and return the first child index.
idx_t ReserveRows(Vector &result, idx_t row, idx_t n) {
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + n);
	ListVector::SetListSize(result, offset + n);
	FlatDataMut<list_entry_t>(result)[row] = {offset, n};
	return offset;
}

void WritePentrace(const AnofoxPentraceResult &r, Vector &result, idx_t row) {
	idx_t out = ReserveRows(result, row, r.n_rows);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t i = 0; i < r.n_rows; i++, out++) {
		auto &e = r.rows[i];
		const double values[] = {e.simple, e.nonlinear, e.interaction, e.nonlinear_interaction, e.df,
		                         e.lr_chisq, e.aic, e.bic, e.aic_c};
		for (idx_t k = 0; k < 9; k++) {
			XyRowsSetDouble(StructEntry(f[k]), out, values[k]);
		}
		FlatDataMut<bool>(StructEntry(f[9]))[out] = i == r.best;
		FlatDataMut<int64_t>(StructEntry(f[10]))[out] = static_cast<int64_t>(r.n_observations);
	}
}

void WritePenalizedOls(const PentraceBindData &bind, const TermsCall &terms, const AnofoxPenalizedOlsResult &r,
                       Vector &result, idx_t row) {
	auto &f = StructVector::GetEntries(result);
	const idx_t p = r.n_features;
	const idx_t k = p + 1;
	const double *vcov = bind.sandwich ? r.vcov_sandwich : r.vcov;
	vector<double> se(p);
	for (idx_t j = 0; j < p; j++) {
		se[j] = std::sqrt(vcov[(j + 1) * k + j + 1]);
	}
	XyRowsSetDoubleList(StructEntry(f[0]), row, r.coefficients, p);
	XyRowsSetDouble(StructEntry(f[1]), row, r.intercept);
	XyRowsSetDoubleList(StructEntry(f[2]), row, se.data(), p);
	XyRowsSetDouble(StructEntry(f[3]), row, std::sqrt(vcov[0]));
	XyRowsSetDoubleList(StructEntry(f[4]), row, vcov, k * k);
	StructEntry(f[5]).SetValue(row, Value(bind.sandwich ? "sandwich" : "simple"));
	XyRowsSetDouble(StructEntry(f[6]), row, r.r_squared);
	XyRowsSetDouble(StructEntry(f[7]), row, r.sigma);
	XyRowsSetDouble(StructEntry(f[8]), row, r.lr_chisq);
	XyRowsSetDouble(StructEntry(f[9]), row, r.effective_df);
	XyRowsSetDoubleList(StructEntry(f[10]), row, r.effective_df_diagonal, k);
	vector<Value> names;
	for (auto &n : terms.names) {
		names.emplace_back(n);
	}
	StructEntry(f[11]).SetValue(row, Value::LIST(LogicalType::VARCHAR, std::move(names)));
	XyRowsSetDoubleList(StructEntry(f[12]), row, r.effective_df_by_term, r.n_terms);
	XyRowsSetDouble(StructEntry(f[13]), row, r.simple);
	XyRowsSetDouble(StructEntry(f[14]), row, r.nonlinear);
	XyRowsSetDouble(StructEntry(f[15]), row, r.interaction);
	XyRowsSetDouble(StructEntry(f[16]), row, r.nonlinear_interaction);
	FlatDataMut<int64_t>(StructEntry(f[17]))[row] = static_cast<int64_t>(r.n_observations);
	FlatDataMut<int64_t>(StructEntry(f[18]))[row] = static_cast<int64_t>(p);
	StructEntry(f[19]).SetValue(row, Value("penalized_ols"));
}

void PentraceFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                      idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PentraceBindData>();
	const char *name = bind.trace ? kPentrace : kPenalizedOls;
	XyRowsStatePointers states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty() || state.n_features == 0) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Thread order must not change the sums of the fit.
		SortRowsCanonically(state.y_values, state.x_columns);
		TermsCall terms(name, bind.spec, state.n_features);
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto x = XyRowsColumnArrays(state);
		AnofoxError error;
		bool ok;
		if (bind.trace) {
			AnofoxPenaltyGrid grid {};
			grid.simple = bind.simple.data();
			grid.n_simple = bind.simple.size();
			grid.nonlinear = bind.nonlinear.empty() ? nullptr : bind.nonlinear.data();
			grid.n_nonlinear = bind.nonlinear.size();
			grid.interaction = bind.interaction.empty() ? nullptr : bind.interaction.data();
			grid.n_interaction = bind.interaction.size();
			grid.nonlinear_interaction =
			    bind.nonlinear_interaction.empty() ? nullptr : bind.nonlinear_interaction.data();
			grid.n_nonlinear_interaction = bind.nonlinear_interaction.size();
			AnofoxPentraceResult r {};
			ok = anofox_pentrace(y, x.data(), x.size(), &terms.c, grid, bind.model, bind.link, bind.criterion, &r,
			                     &error);
			if (ok) {
				WritePentrace(r, result, row);
				anofox_free_pentrace(&r);
			}
		} else {
			AnofoxPenalizedOlsResult r {};
			ok = anofox_penalized_ols(y, x.data(), x.size(), &terms.c, bind.pen_simple, bind.pen_nonlinear,
			                          bind.pen_interaction, bind.pen_nonlinear_interaction, &r, &error);
			if (ok) {
				WritePenalizedOls(bind, terms, r, result, row);
				anofox_free_penalized_ols(&r);
			}
		}
		if (!ok) {
			ThrowUnlessDegenerate(name, error);
			FlatVector::SetNull(result, row, true);
		}
	}
}

//===--------------------------------------------------------------------===//
// Bind and registration
//===--------------------------------------------------------------------===//

template <bool TRACE>
unique_ptr<FunctionData> PentraceBind(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &arguments) {
	const char *name = TRACE ? kPentrace : kPenalizedOls;
	auto bind = make_uniq<PentraceBindData>();
	bind->trace = TRACE;
	if (arguments.size() > 2) {
		ParseOptions(*bind, EvaluateConstantOptions(context, *arguments[2], name));
	}
	if (TRACE && bind->simple.empty()) {
		throw InvalidInputException("%s: option 'penalties' is required, e.g. {'penalties': [0, 1, 2, 4, 8]} or "
		                            "{'penalties': {'simple': [0, 2], 'nonlinear': [0, 2, 4, 8]}}",
		                            name);
	}
	SetReturnTypeOf(function, TRACE ? LogicalType::LIST(PentraceRowType()) : PenalizedOlsType());
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

AggregateFunction MakeAgg(const char *name, vector<LogicalType> args, bind_aggregate_function_t bind) {
	return AggregateFunction(name, std::move(args), LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
	                         ANOFOX_AGG_INIT(XyRowsInitialize), XyRowsUpdate, XyRowsCombine,
	                         ANOFOX_GUARDED_FINALIZE(PentraceFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
	                         bind, XyRowsDestroy);
}

FunctionDescription Describe(const string &description, const string &example, vector<string> names,
                             vector<LogicalType> types) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = {"regression"};
	d.parameter_names = std::move(names);
	d.parameter_types = std::move(types);
	return d;
}

} // namespace

void RegisterPentraceAggregateFunctions(ExtensionLoader &loader) {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	const string terms_doc =
	    " Columns map to terms with terms (one label per column; default every column its own linear term), "
	    "nonlinear (BOOLEAN[], the nonlinear columns of a spline basis), interactions (MAP label -> [factor, factor]) "
	    "and categorical (labels of the terms whose columns are dummies), as wald_anova.";
	{
		AggregateFunctionSet set(kPentrace);
		set.AddFunction(MakeAgg(kPentrace, {LogicalType::DOUBLE, dlist, LogicalType::ANY},
		                        ANOFOX_AGG_BIND(PentraceBind<true>)));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		info.descriptions.push_back(Describe(
		    "rms::pentrace per group: fits the model at every penalty (rms's differential penalty by column type on "
		    "standardised coefficients) and returns a LIST of rows (simple, nonlinear, interaction, "
		    "nonlinear_interaction, df = effective d.f., lr_chisq, aic = LR - 2 df, bic = LR - df log n, aic_c = "
		    "corrected AIC, is_best, n_observations). Options: penalties (DOUBLE[] or a STRUCT of lists by type, "
		    "every combination with complex.more), model ('ols' default, 'logistic', 'ordinal'), link (logistic, "
		    "probit, loglog, cloglog, cauchit), criterion ('aic_c' default, 'aic', 'bic')." +
		        terms_doc,
		    "pentrace_agg(y, [x1, x2], {'penalties': [0, 1, 2, 4, 8]})", {"y", "x", "options"},
		    {LogicalType::DOUBLE, dlist, LogicalType::ANY}));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set(kPenalizedOls);
		set.AddFunction(MakeAgg(kPenalizedOls, {LogicalType::DOUBLE, dlist}, ANOFOX_AGG_BIND(PentraceBind<false>)));
		set.AddFunction(MakeAgg(kPenalizedOls, {LogicalType::DOUBLE, dlist, LogicalType::ANY},
		                        ANOFOX_AGG_BIND(PentraceBind<false>)));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string desc =
		    "rms::ols(penalty = ) per group: penalized least squares with an unpenalized intercept. Returns "
		    "coefficients, intercept, std_errors, vcov (row-major, intercept first; var_penalty 'simple' default or "
		    "'sandwich'), r_squared, sigma (sqrt((SSE + penalty) / n)), lr_chisq, effective_df (total, per "
		    "coefficient and per term), the penalty used, n_observations. Options: penalty (a number or a STRUCT "
		    "{simple, nonlinear, interaction, nonlinear_interaction}), var_penalty." +
		    terms_doc;
		info.descriptions.push_back(
		    Describe(desc, "penalized_ols_fit_agg(y, [x1, x2])", {"y", "x"}, {LogicalType::DOUBLE, dlist}));
		info.descriptions.push_back(Describe(desc, "penalized_ols_fit_agg(y, [x1, x2], {'penalty': 2.0})",
		                                     {"y", "x", "options"}, {LogicalType::DOUBLE, dlist, LogicalType::ANY}));
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
