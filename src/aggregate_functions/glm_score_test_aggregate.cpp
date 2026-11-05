// glm_score_test_agg: Rao score test for adding predictors to a GLM, per group (#310),
// R's anova(glm(y ~ x_reduced), glm(y ~ x_reduced + x_add), test = "Rao").
//
//   glm_score_test_agg(y, x_reduced, x_add[, options])
//
// The test is evaluated at the reduced fit only (the larger model is not fitted,
// except to estimate the dispersion of Gamma / Tweedie / Gaussian families, as
// anova does). A row is skipped when y, either list or any list element is NULL.
// x_reduced may be [] (the intercept-only model). The fit and the statistic are
// computed by anofox-regression (inference::glm_score_test); this file only collects
// rows and writes the STRUCT.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "../include/test_struct.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {

constexpr const char *kName = "glm_score_test_agg";

struct ScoreTestState {
	vector<double> y_values;
	vector<vector<double>> reduced;
	vector<vector<double>> add;
	bool initialized = false;
};

struct ScoreTestBindData : public FunctionData {
	bool fit_intercept = true;
	AnofoxLikelihoodFamily family {0, 0, std::nan("")};
	idx_t offset_column = 0;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ScoreTestBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ScoreTestBindData>();
		auto same = [](double a, double b) {
			return a == b || (std::isnan(a) && std::isnan(b));
		};
		return fit_intercept == o.fit_intercept && family.kind == o.family.kind && family.link == o.family.link &&
		       same(family.param, o.family.param) && offset_column == o.offset_column;
	}
};

LogicalType ScoreTestResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("z_values", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("dispersion", LogicalType::DOUBLE));
	c.push_back(make_pair("theta", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void ScoreTestInitialize(data_ptr_t state_p) {
	new (state_p) ScoreTestState();
}

void ScoreTestDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ScoreTestState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~ScoreTestState();
	}
}

//! A LIST(DOUBLE) argument in unified format, with its child values.
struct ListArg {
	UnifiedVectorFormat format;
	const list_entry_t *entries;
	const double *child;
	const ValidityMask *child_validity;

	ListArg(Vector &v, idx_t count) {
		v.ToUnifiedFormat(count, format);
		entries = UnifiedVectorFormat::GetData<list_entry_t>(format);
		auto &child_vec = ListVector::GetEntry(v);
		child = FlatDataMut<double>(child_vec);
		child_validity = &FlatValidityMut(child_vec);
	}

	//! The row's list entry, or nullptr when the list or one of its elements is NULL.
	const list_entry_t *Row(idx_t i) const {
		auto idx = format.sel->get_index(i);
		if (!format.validity.RowIsValid(idx) || ListHasNullElement(*child_validity, entries[idx])) {
			return nullptr;
		}
		return &entries[idx];
	}
};

void CheckWidth(vector<vector<double>> &columns, const list_entry_t &entry, bool first, const char *arg) {
	if (first) {
		columns.resize(entry.length);
	} else if (entry.length != columns.size()) {
		throw InvalidInputException("%s: inconsistent %s length: expected %llu, got %llu", kName, arg,
		                            (unsigned long long)columns.size(), (unsigned long long)entry.length);
	}
}

void ScoreTestUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	ListArg reduced(inputs[1], count);
	ListArg add(inputs[2], count);
	auto states = (ScoreTestState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto r = reduced.Row(i);
		auto a = add.Row(i);
		if (!y_data.validity.RowIsValid(y_idx) || !r || !a) {
			continue;
		}
		CheckWidth(state.reduced, *r, !state.initialized, "x_reduced");
		CheckWidth(state.add, *a, !state.initialized, "x_add");
		state.initialized = true;
		state.y_values.push_back(y_values[y_idx]);
		for (idx_t j = 0; j < r->length; j++) {
			state.reduced[j].push_back(reduced.child[r->offset + j]);
		}
		for (idx_t j = 0; j < a->length; j++) {
			state.add[j].push_back(add.child[a->offset + j]);
		}
	}
}

void AppendColumns(vector<vector<double>> &target, const vector<vector<double>> &source, const char *arg) {
	if (target.size() != source.size()) {
		throw InvalidInputException("%s: inconsistent %s length: expected %llu, got %llu", kName, arg,
		                            (unsigned long long)target.size(), (unsigned long long)source.size());
	}
	for (idx_t j = 0; j < target.size(); j++) {
		target[j].insert(target[j].end(), source[j].begin(), source[j].end());
	}
}

void ScoreTestCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                      idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (ScoreTestState **)sdata.data;
	auto targets = (ScoreTestState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.reduced = CombineTake(source.reduced, aggr_input_data);
			target.add = CombineTake(source.add, aggr_input_data);
			target.initialized = true;
			continue;
		}
		AppendColumns(target.reduced, source.reduced, "x_reduced");
		AppendColumns(target.add, source.add, "x_add");
		target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
	}
}

vector<AnofoxDataArray> ColumnArrays(vector<vector<double>> &columns) {
	vector<AnofoxDataArray> arrays;
	for (auto &col : columns) {
		arrays.push_back({col.data(), nullptr, col.size()});
	}
	return arrays;
}

//! Write a DOUBLE; non-finite (not applicable) becomes NULL.
void SetDouble(Vector &vec, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(vec)[row] = value;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

void SetDoubleList(Vector &list_vec, idx_t row, const vector<double> &values) {
	auto &child = ListVector::GetEntry(list_vec);
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + values.size());
	ListVector::SetListSize(list_vec, offset + values.size());
	auto data = FlatDataMut<double>(child);
	auto &validity = FlatValidityMut(child);
	for (idx_t j = 0; j < values.size(); j++) {
		if (std::isfinite(values[j])) {
			data[offset + j] = values[j];
		} else {
			validity.SetInvalid(offset + j);
		}
	}
	FlatDataMut<list_entry_t>(list_vec)[row] = {offset, (idx_t)values.size()};
}

void ScoreTestFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                       idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ScoreTestBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ScoreTestState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (state.add.empty()) {
			throw InvalidInputException("%s: x_add must have at least one column", kName);
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Keys(state.add).Keys(state.reduced).Key(state.y_values).Sort();
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto reduced = ColumnArrays(state.reduced);
		auto add = ColumnArrays(state.add);
		vector<double> z(add.size(), std::nan(""));
		AnofoxChiSqTestResult res {};
		AnofoxError error;
		if (!anofox_glm_score_test(bind.family, y, reduced.empty() ? nullptr : reduced.data(), reduced.size(),
		                           add.data(), add.size(), bind.fit_intercept, bind.offset_column, z.data(), &res,
		                           &error)) {
			ThrowUnlessDegenerate(kName, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		SetResultString(StructEntry(entries[k++]), row, "Rao score");
		SetDouble(StructEntry(entries[k++]), row, res.statistic);
		SetDouble(StructEntry(entries[k++]), row, res.df);
		SetDouble(StructEntry(entries[k++]), row, res.p_value);
		SetDoubleList(StructEntry(entries[k++]), row, z);
		SetDouble(StructEntry(entries[k++]), row, res.dispersion);
		SetDouble(StructEntry(entries[k++]), row, res.theta);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = res.n_observations;
		WriteTestResultFields(result, row, k, res.unified);
	}
}

//! Reject a constant NULL list: a missing x_add, or NULL instead of [] for x_reduced.
void RequireList(ClientContext &context, Expression &arg, const char *name) {
	if (arg.IsFoldable() && ExpressionExecutor::EvaluateScalar(context, arg).IsNull()) {
		throw InvalidInputException("%s: %s is required (x_reduced := [] is the intercept-only model)", kName,
		                            name);
	}
}

//! The link code of `family` (AnofoxPoissonLink / AnofoxBinomialLink) named by `value`.
int32_t ParseLink(const Value &value, int32_t family_kind) {
	const string v = StringUtil::Lower(value.ToString());
	if (family_kind == 1) {
		if (v == "log") {
			return ANOFOX_POISSON_LINK_LOG;
		}
		if (v == "identity") {
			return ANOFOX_POISSON_LINK_IDENTITY;
		}
		if (v == "sqrt") {
			return ANOFOX_POISSON_LINK_SQRT;
		}
		throw InvalidInputException("%s: invalid Poisson link '%s'; expected 'log', 'identity' or 'sqrt'", kName,
		                            value.ToString());
	}
	if (family_kind == 2) {
		if (v == "logit") {
			return ANOFOX_BINOMIAL_LINK_LOGIT;
		}
		if (v == "probit") {
			return ANOFOX_BINOMIAL_LINK_PROBIT;
		}
		if (v == "cloglog") {
			return ANOFOX_BINOMIAL_LINK_CLOGLOG;
		}
		throw InvalidInputException("%s: invalid binomial link '%s'; expected 'logit', 'probit' or 'cloglog'",
		                            kName, value.ToString());
	}
	// The other families have one link (identity for gaussian, log otherwise).
	if ((family_kind == 0 && v == "identity") || (family_kind != 0 && v == "log")) {
		return 0;
	}
	throw InvalidInputException("%s: link '%s' is not supported for this family (gaussian: identity; negbinom, "
	                            "gamma, tweedie: log)",
	                            kName, value.ToString());
}

unique_ptr<FunctionData> ScoreTestBind(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<ScoreTestBindData>();
	RequireList(context, *args[1], "x_reduced");
	RequireList(context, *args[2], "x_add");
	if (args.size() >= 4) {
		auto opts = RegressionMapOptions::ParseFromExpression(
		    context, *args[3], kName, {"fit_intercept", "family", "link", "power", "theta", "offset"});
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.glmm_family.has_value()) {
			bind->family.kind = opts.glmm_family.value();
		}
		if (opts.link_value.has_value() && !opts.link_value->IsNull()) {
			bind->family.link = ParseLink(opts.link_value.value(), bind->family.kind);
		}
		if (bind->family.kind == 5) {
			bind->family.param = opts.tweedie_power.value_or(1.5);
		} else if (opts.tweedie_power.has_value()) {
			throw InvalidInputException("%s: 'power' applies only to family 'tweedie'", kName);
		}
		if (bind->family.kind == 3) {
			bind->family.param = opts.nb_theta.value_or(std::nan(""));
		} else if (opts.nb_theta.has_value()) {
			throw InvalidInputException("%s: 'theta' applies only to family 'negbinom'", kName);
		}
		if (opts.offset_column.has_value()) {
			bind->offset_column = opts.offset_column.value();
		}
	}
	SetReturnTypeOf(function, ScoreTestResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kName);
	return std::move(bind);
}

} // namespace

void RegisterGlmScoreTestAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set(kName);
	const auto list = LogicalType::LIST(LogicalType::DOUBLE);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, list, list};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    kName, args, LogicalType::ANY, AggregateFunction::StateSize<ScoreTestState>,
		    ANOFOX_AGG_INIT(ScoreTestInitialize),
		    ANOFOX_GUARDED_UPDATE(ScoreTestUpdate, ScoreTestDestroy, ScoreTestInitialize), ScoreTestCombine,
		    ANOFOX_GUARDED_FINALIZE(ScoreTestFinalize, ScoreTestDestroy, ScoreTestInitialize), nullptr,
		    ANOFOX_AGG_BIND(ScoreTestBind), ScoreTestDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    "Rao score test per group for adding the x_add predictors to the GLM of y on x_reduced (R's "
	    "anova(reduced, full, test = 'Rao')), evaluated at the reduced fit. Returns the chi-square statistic, df1, "
	    "p_value and the per-column z_values. Options: family ('gaussian' default, 'binomial', 'poisson', "
	    "'negbinom', 'gamma', 'tweedie'), link, power (tweedie), theta (negbinom; estimated from the reduced fit "
	    "when omitted), fit_intercept, offset (1-based column of x_reduced). x_reduced := [] is the intercept-only "
	    "model.";
	FunctionDescription d1;
	d1.description = description;
	d1.examples = {"glm_score_test_agg(y, [x1], [x2, x3])"};
	d1.categories = {"regression", "hypothesis-test"};
	d1.parameter_names = {"y", "x_reduced", "x_add"};
	d1.parameter_types = {LogicalType::DOUBLE, list, list};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = description;
	d2.examples = {"glm_score_test_agg(y, [x1], [x2, x3], {'family': 'poisson'})"};
	d2.categories = {"regression", "hypothesis-test"};
	d2.parameter_names = {"y", "x_reduced", "x_add", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, list, list, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
