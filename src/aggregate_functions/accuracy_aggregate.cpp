#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "telemetry.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Predictive accuracy indexes (#308), rms / Hmisc:
//
//   concordance_agg(y, x [, weight])        survival::concordance pair counts, C, Dxy, gamma, tau-a
//   somers2_agg(y_true, prob [, weight])    Hmisc::somers2
//   rcorr_cens_agg(time, event, x)          Hmisc::rcorr.cens
//   lrm_stats_agg(y_true, lp, df [, weight]) the stats vector of rms::lrm
//   val_prob_agg(y_true, prob)              rms::val.prob(pl = FALSE)
//   gini_md_agg(x)                          Hmisc::GiniMd (the g-index of a linear predictor)
//   r2_measures(lr, p, n [, ess])           Hmisc::R2Measures (scalar)
//
// The statistics come from anofox_regression::diagnostics::accuracy. y_true is
// BOOLEAN or a numeric 0/1 (anything else is an error); event is BOOLEAN or
// numeric (non-zero = observed). Rows with a NULL in any argument are skipped;
// no rows left gives NULL. Rows with a NaN prediction are dropped by the rank
// indexes; a value that is undefined (e.g. C with one class) is NULL.
//===--------------------------------------------------------------------===//

enum class AccuracyKind : uint8_t { CONCORDANCE, SOMERS2, RCORR_CENS, LRM_STATS, VAL_PROB, GINI_MD };

struct AccuracyState {
	//! y_true / y / time / x (gini_md_agg).
	vector<double> a;
	//! Prediction (prob, lp, x); unused by gini_md_agg.
	vector<double> b;
	//! Weight or event; empty when not given.
	vector<double> c;

	void Reset() {
		a.clear();
		b.clear();
		c.clear();
	}
};

struct AccuracyBindData : public FunctionData {
	AccuracyKind kind;
	string name;
	double df = 0.0;

	AccuracyBindData(AccuracyKind kind_p, string name_p) : kind(kind_p), name(std::move(name_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<AccuracyBindData>(kind, name);
		result->df = df;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<AccuracyBindData>();
		return kind == o.kind && df == o.df;
	}
};

static LogicalType AccuracyResultType(AccuracyKind kind) {
	child_list_t<LogicalType> c;
	auto d = [&](const char *name) {
		c.push_back(make_pair(name, LogicalType::DOUBLE));
	};
	auto i = [&](const char *name) {
		c.push_back(make_pair(name, LogicalType::BIGINT));
	};
	switch (kind) {
	case AccuracyKind::CONCORDANCE:
		for (auto name : {"concordant", "discordant", "tied_x", "tied_y", "tied_xy", "c", "dxy", "gamma", "tau_a"}) {
			d(name);
		}
		break;
	case AccuracyKind::SOMERS2:
		d("c");
		d("dxy");
		d("n");
		break;
	case AccuracyKind::RCORR_CENS:
		d("c");
		d("dxy");
		d("sd");
		i("n");
		i("missing");
		i("uncensored");
		d("relevant_pairs");
		d("concordant");
		d("uncertain");
		break;
	case AccuracyKind::LRM_STATS:
		i("n");
		for (auto name : {"sum_weights", "model_lr", "df", "p_value", "c", "dxy", "gamma", "tau_a", "r2", "r2_n",
		                  "r2_adj_n", "ess", "r2_ess", "r2_adj_ess", "brier", "g", "gr", "gp"}) {
			d(name);
		}
		break;
	case AccuracyKind::VAL_PROB:
		for (auto name : {"dxy", "c", "r2", "d", "d_chisq", "d_p", "u", "u_chisq", "u_p", "q", "brier", "intercept",
		                  "slope", "emax", "e90", "eavg", "s_z", "s_p"}) {
			d(name);
		}
		break;
	case AccuracyKind::GINI_MD:
		return LogicalType::DOUBLE;
	}
	return LogicalType::STRUCT(std::move(c));
}

static void AccuracyInitialize(data_ptr_t state_p) {
	new (state_p) AccuracyState();
}

static void AccuracyDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AccuracyState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~AccuracyState();
	}
}

//! A binary label (BOOLEAN, or numeric 0/1) or an event indicator (non-zero = observed).
static double ReadLabel(const UnifiedVectorFormat &data, idx_t idx, bool is_bool, bool event, const string &fn) {
	if (is_bool) {
		return UnifiedVectorFormat::GetData<bool>(data)[idx] ? 1.0 : 0.0;
	}
	double v = UnifiedVectorFormat::GetData<double>(data)[idx];
	if (event) {
		return v != 0.0 ? 1.0 : 0.0;
	}
	if (v != 0.0 && v != 1.0) {
		throw InvalidInputException("%s: y_true must be BOOLEAN or 0/1, got %s", fn, std::to_string(v));
	}
	return v;
}

static void AccuracyUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<AccuracyBindData>();
	const auto kind = bind_data.kind;

	// Column roles: a, b and an optional c (weight or event).
	idx_t ia = 0, ib = 1, ic = DConstants::INVALID_INDEX;
	bool binary_a = kind == AccuracyKind::SOMERS2 || kind == AccuracyKind::LRM_STATS || kind == AccuracyKind::VAL_PROB;
	bool event_c = false;
	if (kind == AccuracyKind::RCORR_CENS) {
		// (time, event, x)
		ia = 0;
		ib = 2;
		ic = 1;
		event_c = true;
	} else if (kind == AccuracyKind::LRM_STATS) {
		// (y_true, lp, df [, weight]); df is a bound constant
		if (input_count >= 4) {
			ic = 3;
		}
	} else if (kind == AccuracyKind::GINI_MD) {
		ib = DConstants::INVALID_INDEX;
	} else if (input_count >= 3) {
		ic = 2;
	}

	UnifiedVectorFormat a_data, b_data, c_data, sdata;
	inputs[ia].ToUnifiedFormat(count, a_data);
	if (ib != DConstants::INVALID_INDEX) {
		inputs[ib].ToUnifiedFormat(count, b_data);
	}
	if (ic != DConstants::INVALID_INDEX) {
		inputs[ic].ToUnifiedFormat(count, c_data);
	}
	const bool a_bool = inputs[ia].GetType().id() == LogicalTypeId::BOOLEAN;
	const bool c_bool = ic != DConstants::INVALID_INDEX && inputs[ic].GetType().id() == LogicalTypeId::BOOLEAN;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AccuracyState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto a_idx = a_data.sel->get_index(i);
		if (!a_data.validity.RowIsValid(a_idx)) {
			continue;
		}
		idx_t b_idx = 0, c_idx = 0;
		if (ib != DConstants::INVALID_INDEX) {
			b_idx = b_data.sel->get_index(i);
			if (!b_data.validity.RowIsValid(b_idx)) {
				continue;
			}
		}
		if (ic != DConstants::INVALID_INDEX) {
			c_idx = c_data.sel->get_index(i);
			if (!c_data.validity.RowIsValid(c_idx)) {
				continue;
			}
		}
		auto &state = *states[sdata.sel->get_index(i)];
		if (binary_a) {
			state.a.push_back(ReadLabel(a_data, a_idx, a_bool, false, bind_data.name));
		} else {
			state.a.push_back(UnifiedVectorFormat::GetData<double>(a_data)[a_idx]);
		}
		if (ib != DConstants::INVALID_INDEX) {
			state.b.push_back(UnifiedVectorFormat::GetData<double>(b_data)[b_idx]);
		}
		if (ic != DConstants::INVALID_INDEX) {
			if (event_c) {
				state.c.push_back(ReadLabel(c_data, c_idx, c_bool, true, bind_data.name));
			} else {
				state.c.push_back(UnifiedVectorFormat::GetData<double>(c_data)[c_idx]);
			}
		}
	}
}

static void AccuracyCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (AccuracyState **)source_data.data;
	auto targets = (AccuracyState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		target.a.insert(target.a.end(), source.a.begin(), source.a.end());
		target.b.insert(target.b.end(), source.b.begin(), source.b.end());
		target.c.insert(target.c.end(), source.c.begin(), source.c.end());
	}
}

//! Write `v` to a DOUBLE child, NULL when NaN (an undefined index).
static inline void SetDoubleOrNull(Vector &vec, idx_t idx, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(vec, idx, true);
	} else {
		FlatDataMut<double>(vec)[idx] = v;
	}
}

//! Sequential writer of the STRUCT children of one result row.
struct StructRowWriter {
	StructEntries &entries;
	idx_t row;
	idx_t k = 0;

	StructRowWriter(Vector &result, idx_t row_p) : entries(StructVector::GetEntries(result)), row(row_p) {
	}
	void D(double v) {
		SetDoubleOrNull(StructEntry(entries[k++]), row, v);
	}
	void I(size_t v) {
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(v);
	}
};

static void AccuracyFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	auto &bind_data = aggr_input_data.bind_data->Cast<AccuracyBindData>();
	const string &name = bind_data.name;
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AccuracyState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (state.a.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		{
			CanonicalRows rows;
			rows.Key(state.a);
			if (!state.b.empty()) {
				rows.Key(state.b);
			}
			if (!state.c.empty()) {
				rows.Key(state.c);
			}
			rows.Sort();
		}
		auto arr = [](vector<double> &v) {
			return AnofoxDataArray {v.data(), nullptr, static_cast<size_t>(v.size())};
		};
		AnofoxDataArray a = arr(state.a), b = arr(state.b), c = arr(state.c);
		AnofoxError error;
		bool ok = false;
		auto w = [&]() {
			return StructRowWriter(result, row);
		};

		switch (bind_data.kind) {
		case AccuracyKind::CONCORDANCE: {
			AnofoxConcordanceResult r;
			ok = anofox_concordance(b, a, c, &r, &error);
			if (ok) {
				auto out = w();
				for (double v :
				     {r.concordant, r.discordant, r.tied_x, r.tied_y, r.tied_xy, r.c, r.dxy, r.gamma, r.tau_a}) {
					out.D(v);
				}
			}
			break;
		}
		case AccuracyKind::SOMERS2: {
			AnofoxSomers2Result r;
			ok = anofox_somers2(b, a, c, &r, &error);
			if (ok) {
				auto out = w();
				out.D(r.c);
				out.D(r.dxy);
				out.D(r.n);
			}
			break;
		}
		case AccuracyKind::RCORR_CENS: {
			AnofoxRcorrCensResult r;
			ok = anofox_rcorr_cens(b, a, c, &r, &error);
			if (ok) {
				auto out = w();
				out.D(r.c);
				out.D(r.dxy);
				out.D(r.sd);
				out.I(r.n);
				out.I(r.missing);
				out.I(r.uncensored);
				out.D(r.relevant_pairs);
				out.D(r.concordant);
				out.D(r.uncertain);
			}
			break;
		}
		case AccuracyKind::LRM_STATS: {
			AnofoxLrmStatsResult r;
			ok = anofox_lrm_stats(a, b, c, bind_data.df, &r, &error);
			if (ok) {
				auto out = w();
				out.I(r.n);
				for (double v : {r.sum_weights, r.model_lr, r.df, r.p_value, r.c, r.dxy, r.gamma, r.tau_a, r.r2,
				                 r.r2_measures.r2, r.r2_measures.r2_adj, r.ess, r.r2_measures.r2_ess,
				                 r.r2_measures.r2_adj_ess, r.brier, r.g, r.gr, r.gp}) {
					out.D(v);
				}
			}
			break;
		}
		case AccuracyKind::VAL_PROB: {
			AnofoxValProbResult r;
			ok = anofox_val_prob(b, a, &r, &error);
			if (ok) {
				auto out = w();
				for (double v : {r.dxy, r.c, r.r2, r.d, r.d_chisq, r.d_p, r.u, r.u_chisq, r.u_p, r.q, r.brier,
				                 r.intercept, r.slope, r.emax, r.e90, r.eavg, r.s_z, r.s_p}) {
					out.D(v);
				}
			}
			break;
		}
		case AccuracyKind::GINI_MD: {
			double v = 0.0;
			ok = anofox_gini_md(a, &v, &error);
			if (ok) {
				SetDoubleOrNull(result, row, v);
			}
			break;
		}
		}
		state.Reset();
		if (!ok) {
			ThrowUnlessDegenerate(name.c_str(), error);
			FlatVector::SetNull(result, row, true);
		}
	}
}

//! Evaluate a constant scalar argument; it must not vary per row.
static Value AccuracyConstantArgument(ClientContext &context, Expression &expr, const string &fn, const char *param) {
	RequireResolvedParameters(expr);
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: %s must be a constant", fn, string(param));
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		throw InvalidInputException("%s: %s must not be NULL", fn, string(param));
	}
	return value;
}

template <AccuracyKind KIND>
static unique_ptr<FunctionData> AccuracyBind(ClientContext &context, AggregateBindFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<AccuracyBindData>(KIND, NameOf(function));
	const string &fn = result->name;
	if (KIND == AccuracyKind::LRM_STATS) {
		auto df = AccuracyConstantArgument(context, *arguments[2], fn, "df").GetValue<double>();
		if (!(std::isfinite(df) && df >= 0.0)) {
			throw InvalidInputException("%s: df must be a finite number >= 0, got %s", fn, std::to_string(df));
		}
		result->df = df;
	}
	SetReturnTypeOf(function, AccuracyResultType(KIND));
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(result);
}

struct AccuracyOverload {
	vector<LogicalType> args;
	vector<string> names;
};

static void RegisterAccuracyAggregate(ExtensionLoader &loader, const char *name, bind_aggregate_function_t bind,
                                      const vector<AccuracyOverload> &overloads, const char *description,
                                      const char *example) {
	AggregateFunctionSet func_set(AsDuckName(name));
	vector<FunctionDescription> descriptions;
	for (auto &o : overloads) {
		AggregateFunction fn(name, o.args, LogicalType::ANY, AggregateFunction::StateSize<AccuracyState>,
		                     ANOFOX_AGG_INIT(AccuracyInitialize),
		                     ANOFOX_GUARDED_UPDATE(AccuracyUpdate, AccuracyDestroy, AccuracyInitialize),
		                     AccuracyCombine,
		                     ANOFOX_GUARDED_FINALIZE(AccuracyFinalize, AccuracyDestroy, AccuracyInitialize), nullptr,
		                     bind, AccuracyDestroy);
		func_set.AddFunction(fn);

		FunctionDescription d;
		d.description = description;
		d.examples = {example};
		d.categories = {"classification", "diagnostics"};
		d.parameter_names = o.names;
		d.parameter_types = o.args;
		descriptions.push_back(std::move(d));
	}
	CreateAggregateFunctionInfo info(func_set);
	info.descriptions = std::move(descriptions);
	loader.RegisterFunction(info);
}

//===--------------------------------------------------------------------===//
// r2_measures(lr, p, n [, ess]) -> STRUCT(r2, r2_adj, r2_ess, r2_adj_ess)
//===--------------------------------------------------------------------===//

static LogicalType R2MeasuresType() {
	child_list_t<LogicalType> c;
	for (auto name : {"r2", "r2_adj", "r2_ess", "r2_adj_ess"}) {
		c.push_back(make_pair(name, LogicalType::DOUBLE));
	}
	return LogicalType::STRUCT(std::move(c));
}

static unique_ptr<FunctionData> R2MeasuresBind(ClientContext &, ScalarBindFunction &bound_function,
                                               vector<unique_ptr<Expression>> &) {
	SetReturnTypeOf(bound_function, R2MeasuresType());
	PostHogTelemetry::Instance().RecordFunctionCall("r2_measures");
	return nullptr;
}

static void R2MeasuresFunction(DataChunk &args, ExpressionState &, Vector &result) {
	// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
	// asserts this in debug builds).
	const bool all_constant = args.AllConstant();
	const idx_t count = args.size();
	const idx_t n_args = args.ColumnCount();
	vector<UnifiedVectorFormat> data(n_args);
	for (idx_t k = 0; k < n_args; k++) {
		args.data[k].ToUnifiedFormat(count, data[k]);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < count; row++) {
		double v[4] = {0.0, 0.0, 0.0, NAN};
		bool null = false;
		for (idx_t k = 0; k < n_args; k++) {
			auto idx = data[k].sel->get_index(row);
			if (!data[k].validity.RowIsValid(idx)) {
				null = true;
				break;
			}
			v[k] = UnifiedVectorFormat::GetData<double>(data[k])[idx];
		}
		if (null) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxR2MeasuresResult r;
		AnofoxError error;
		if (!anofox_r2_measures(v[0], v[1], v[2], v[3], &r, &error)) {
			ThrowFromFfiError("r2_measures", error);
		}
		StructRowWriter w(result, row);
		w.D(r.r2);
		w.D(r.r2_adj);
		w.D(r.r2_ess);
		w.D(r.r2_adj_ess);
	}
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void RegisterR2Measures(ExtensionLoader &loader) {
	ScalarFunctionSet func_set("r2_measures");
	vector<FunctionDescription> descriptions;
	for (idx_t n_args = 3; n_args <= 4; n_args++) {
		vector<LogicalType> args(n_args, LogicalType::DOUBLE);
		ScalarFunction fn(args, LogicalType::ANY, R2MeasuresFunction, ANOFOX_SCALAR_BIND(R2MeasuresBind));
		MarkFallible(fn);
		func_set.AddFunction(fn);

		FunctionDescription d;
		d.description =
		    "Explained-variation measures of Hmisc::R2Measures (RMS 2nd ed. 4.5) from a model likelihood-ratio "
		    "chi-square lr on p d.f. with n observations: r2 = 1 - exp(-lr/n) (Maddala / Cox-Snell), r2_adj = "
		    "1 - exp(-max(lr - p, 0)/n), and the same two with the effective sample size ess (NULL without ess).";
		d.examples = {n_args == 3 ? "r2_measures(32.8, 2, 40)" : "r2_measures(32.8, 2, 40, 28.1)"};
		d.categories = {"diagnostics"};
		d.parameter_names = {"lr", "p", "n"};
		if (n_args == 4) {
			d.parameter_names.push_back("ess");
		}
		d.parameter_types = args;
		descriptions.push_back(std::move(d));
	}
	CreateScalarFunctionInfo info(std::move(func_set));
	info.descriptions = std::move(descriptions);
	loader.RegisterFunction(std::move(info));
}

void RegisterAccuracyFunctions(ExtensionLoader &loader) {
	const auto D = LogicalType::DOUBLE;
	const auto B = LogicalType::BOOLEAN;

	RegisterAccuracyAggregate(
	    loader, "concordance_agg", ANOFOX_AGG_BIND(AccuracyBind<AccuracyKind::CONCORDANCE>),
	    {{{D, D}, {"y", "x"}}, {{D, D, D}, {"y", "x", "weight"}}},
	    "Concordance of the prediction x with the ordered response y (survival::concordance): pair counts "
	    "concordant, discordant, tied_x, tied_y, tied_xy (weighted by w_i w_j), the concordance probability c, "
	    "Somers' dxy, Goodman-Kruskal gamma and Kendall tau_a (the rank statistics of rms::lrm / orm).",
	    "concordance_agg(y, prediction)");

	RegisterAccuracyAggregate(loader, "somers2_agg", ANOFOX_AGG_BIND(AccuracyBind<AccuracyKind::SOMERS2>),
	                          {{{B, D}, {"y_true", "prob"}},
	                           {{D, D}, {"y_true", "prob"}},
	                           {{B, D, D}, {"y_true", "prob", "weight"}},
	                           {{D, D, D}, {"y_true", "prob", "weight"}}},
	                          "ROC area c and Somers' dxy = 2(c - 1/2) of a prediction for a 0/1 response, with n "
	                          "(the sum of the weights when weighted), as Hmisc::somers2. c and dxy are NULL when "
	                          "only one class is present.",
	                          "somers2_agg(y_true, prob)");

	RegisterAccuracyAggregate(loader, "rcorr_cens_agg", ANOFOX_AGG_BIND(AccuracyBind<AccuracyKind::RCORR_CENS>),
	                          {{{D, B, D}, {"time", "event", "x"}}, {{D, D, D}, {"time", "event", "x"}}},
	                          "Harrell's C index of the prediction x (larger x = longer survival) for right-censored "
	                          "times (event true / non-zero = observed), as Hmisc::rcorr.cens: c, dxy, its standard "
	                          "deviation sd, n, missing, uncensored and the relevant / concordant / uncertain ordered "
	                          "pair counts. Pass the negated risk score to rank a hazard.",
	                          "rcorr_cens_agg(time, event, predicted_time)");

	RegisterAccuracyAggregate(
	    loader, "lrm_stats_agg", ANOFOX_AGG_BIND(AccuracyBind<AccuracyKind::LRM_STATS>),
	    {{{B, D, D}, {"y_true", "lp", "df"}},
	     {{D, D, D}, {"y_true", "lp", "df"}},
	     {{B, D, D, D}, {"y_true", "lp", "df", "weight"}},
	     {{D, D, D, D}, {"y_true", "lp", "df", "weight"}}},
	    "The stats vector of a binary rms::lrm fit from its 0/1 response, its linear predictor lp (with intercept, "
	    "at the maximum likelihood estimate) and the model d.f. df (constant): n, sum_weights, model_lr, df, "
	    "p_value, c, dxy, gamma, tau_a, the Nagelkerke r2, r2_n / r2_adj_n (R2(n), R2(p,n)), the effective sample "
	    "size ess with r2_ess / r2_adj_ess (R2(m), R2(p,m)), brier, the g-index g, gr = exp(g) and gp.",
	    "lrm_stats_agg(y_true, lp, 2)");

	RegisterAccuracyAggregate(loader, "val_prob_agg", ANOFOX_AGG_BIND(AccuracyBind<AccuracyKind::VAL_PROB>),
	                          {{{B, D}, {"y_true", "prob"}}, {{D, D}, {"y_true", "prob"}}},
	                          "External validation of predicted probabilities against 0/1 outcomes, as "
	                          "rms::val.prob(pl = FALSE): dxy, c, r2, the discrimination index d (d_chisq, d_p), the "
	                          "unreliability index u (u_chisq, u_p), the quality index q = d - u, brier, the "
	                          "calibration intercept and slope, emax / e90 / eavg of the lowess calibration curve "
	                          "and Spiegelhalter's s_z / s_p.",
	                          "val_prob_agg(y_true, prob)");

	RegisterAccuracyAggregate(loader, "gini_md_agg", ANOFOX_AGG_BIND(AccuracyBind<AccuracyKind::GINI_MD>),
	                          {{{D}, {"x"}}},
	                          "Gini's mean difference sum_{i != j} |x_i - x_j| / (n (n - 1)) (Hmisc::GiniMd); the "
	                          "g-index when x is a linear predictor. NULL for fewer than 2 values or a NaN value.",
	                          "gini_md_agg(lp)");

	RegisterR2Measures(loader);
}

} // namespace duckdb
