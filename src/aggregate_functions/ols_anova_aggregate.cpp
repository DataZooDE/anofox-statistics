// ols_anova_agg: nested-model F test of two OLS fits per group (#221), R's
// anova(lm(y ~ x_reduced), lm(y ~ x_full)).
//
//   ols_anova_agg(y, x_reduced, x_full[, options])
//
// Both models are fitted on the same rows: a row is skipped when y, either list
// or any list element is NULL. x_reduced may be [] (the intercept-only model,
// i.e. the overall F test). Its columns must lie in the span of x_full, which
// upstream checks; a reduced column outside it is an input error. The fits, the
// RSS values and the F test are computed by anofox-regression
// (inference::nested_f_test); this file only collects rows and writes the STRUCT.

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

namespace duckdb {

namespace {

constexpr const char *kName = "ols_anova_agg";

struct OlsAnovaState {
	vector<double> y_values;
	vector<vector<double>> reduced;
	vector<vector<double>> full;
	bool initialized = false;
};

struct OlsAnovaBindData : public FunctionData {
	bool fit_intercept = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<OlsAnovaBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return fit_intercept == other_p.Cast<OlsAnovaBindData>().fit_intercept;
	}
};

LogicalType OlsAnovaResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("df2", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("rss_reduced", LogicalType::DOUBLE));
	c.push_back(make_pair("rss_full", LogicalType::DOUBLE));
	c.push_back(make_pair("adj_r_squared_reduced", LogicalType::DOUBLE));
	c.push_back(make_pair("adj_r_squared_full", LogicalType::DOUBLE));
	c.push_back(make_pair("delta_adj_r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void OlsAnovaInitialize(data_ptr_t state_p) {
	new (state_p) OlsAnovaState();
}

void OlsAnovaDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OlsAnovaState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~OlsAnovaState();
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

void OlsAnovaUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	ListArg reduced(inputs[1], count);
	ListArg full(inputs[2], count);
	auto states = (OlsAnovaState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto r = reduced.Row(i);
		auto f = full.Row(i);
		if (!y_data.validity.RowIsValid(y_idx) || !r || !f) {
			continue;
		}
		CheckWidth(state.reduced, *r, !state.initialized, "x_reduced");
		CheckWidth(state.full, *f, !state.initialized, "x_full");
		state.initialized = true;
		state.y_values.push_back(y_values[y_idx]);
		for (idx_t j = 0; j < r->length; j++) {
			state.reduced[j].push_back(reduced.child[r->offset + j]);
		}
		for (idx_t j = 0; j < f->length; j++) {
			state.full[j].push_back(full.child[f->offset + j]);
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

void OlsAnovaCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                     idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (OlsAnovaState **)sdata.data;
	auto targets = (OlsAnovaState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.reduced = CombineTake(source.reduced, aggr_input_data);
			target.full = CombineTake(source.full, aggr_input_data);
			target.initialized = true;
			continue;
		}
		AppendColumns(target.reduced, source.reduced, "x_reduced");
		AppendColumns(target.full, source.full, "x_full");
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

void OlsAnovaFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                      idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<OlsAnovaBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OlsAnovaState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto reduced = ColumnArrays(state.reduced);
		auto full = ColumnArrays(state.full);
		AnofoxOlsAnovaResult res;
		AnofoxError error;
		if (!anofox_ols_anova(y, reduced.empty() ? nullptr : reduced.data(), reduced.size(), full.data(),
		                      full.size(), bind.fit_intercept, &res, &error)) {
			ThrowUnlessDegenerate(kName, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		SetResultString(StructEntry(entries[k++]), row, "nested F");
		for (double v : {res.statistic, res.df1, res.df2, res.p_value, res.rss_reduced, res.rss_full,
		                 res.adj_r_squared_reduced, res.adj_r_squared_full, res.delta_adj_r_squared}) {
			SetDouble(StructEntry(entries[k++]), row, v);
		}
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = (int64_t)res.n_observations;
		WriteTestResultFields(result, row, k, res.unified);
	}
}

//! Reject a constant NULL list: a missing x_full, or NULL instead of [] for x_reduced.
void RequireList(ClientContext &context, Expression &arg, const char *name) {
	if (arg.IsFoldable() && ExpressionExecutor::EvaluateScalar(context, arg).IsNull()) {
		throw InvalidInputException("%s: %s is required (x_reduced := [] is the intercept-only model)", kName,
		                            name);
	}
}

unique_ptr<FunctionData> OlsAnovaBind(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<OlsAnovaBindData>();
	RequireList(context, *args[1], "x_reduced");
	RequireList(context, *args[2], "x_full");
	if (args.size() >= 4) {
		auto opts = RegressionMapOptions::ParseFromExpression(context, *args[3], kName, {"fit_intercept"});
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
	}
	SetReturnTypeOf(function, OlsAnovaResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kName);
	return std::move(bind);
}

} // namespace

void RegisterOlsAnovaAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set(kName);
	const auto list = LogicalType::LIST(LogicalType::DOUBLE);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, list, list};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    kName, args, LogicalType::ANY, AggregateFunction::StateSize<OlsAnovaState>,
		    ANOFOX_AGG_INIT(OlsAnovaInitialize), ANOFOX_GUARDED_UPDATE(OlsAnovaUpdate, OlsAnovaDestroy, OlsAnovaInitialize),
		    OlsAnovaCombine, ANOFOX_GUARDED_FINALIZE(OlsAnovaFinalize, OlsAnovaDestroy, OlsAnovaInitialize), nullptr,
		    ANOFOX_AGG_BIND(OlsAnovaBind), OlsAnovaDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    "Nested-model F test per group (R's anova(lm(y ~ x_reduced), lm(y ~ x_full))): does the full OLS model fit "
	    "significantly better than the reduced one? Returns the F statistic, df1, df2, p_value, both RSS values and "
	    "both adjusted R^2 values. x_reduced := [] is the intercept-only model; its columns must appear in x_full.";
	FunctionDescription d1;
	d1.description = description;
	d1.examples = {"ols_anova_agg(y, [price], [price, promo, temp])"};
	d1.categories = {"regression", "hypothesis-test"};
	d1.parameter_names = {"y", "x_reduced", "x_full"};
	d1.parameter_types = {LogicalType::DOUBLE, list, list};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = description;
	d2.examples = {"ols_anova_agg(y, [price], [price, promo, temp], {'fit_intercept': false})"};
	d2.categories = {"regression", "hypothesis-test"};
	d2.parameter_names = {"y", "x_reduced", "x_full", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, list, list, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
