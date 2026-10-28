// ols_anova_table_agg: per-term ANOVA table of an OLS fit per group (#231), R's
// anova(lm(y ~ x)) (type I, sequential) or car::Anova(fit, type = 2) (type II,
// partial).
//
//   ols_anova_table_agg(y, x[, options])
//
// Returns LIST(STRUCT(term, method, statistic, df1, df2, p_value, sum_sq,
// mean_sq, residual_sum_sq, residual_mean_sq, n_observations, <unified test
// fields>)), one element per term. A row is skipped when y, the list or any list
// element is NULL. By default every x column is its own term, labelled x1, x2,
// ...; the terms option labels the columns, and columns sharing a label form one
// term (for example the dummies of a factor). The fits and the table are
// computed by anofox-regression (inference::anova_table); this file only collects
// rows, maps the labels to term indices and writes the list.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
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

constexpr const char *kName = "ols_anova_table_agg";

struct OlsAnovaTableState {
	vector<double> y_values;
	vector<vector<double>> x;
	bool initialized = false;
};

struct OlsAnovaTableBindData : public FunctionData {
	bool fit_intercept = true;
	int32_t anova_type = 1;
	//! One label per x column; empty: x1, x2, ...
	vector<string> terms;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<OlsAnovaTableBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<OlsAnovaTableBindData>();
		return fit_intercept == o.fit_intercept && anova_type == o.anova_type && terms == o.terms;
	}
};

LogicalType OlsAnovaTableRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("df2", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("sum_sq", LogicalType::DOUBLE));
	c.push_back(make_pair("mean_sq", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_sum_sq", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_mean_sq", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void OlsAnovaTableInitialize(data_ptr_t state_p) {
	new (state_p) OlsAnovaTableState();
}

void OlsAnovaTableDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OlsAnovaTableState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~OlsAnovaTableState();
	}
}

void ThrowWidthMismatch(idx_t expected, idx_t got) {
	throw InvalidInputException("%s: inconsistent x length: expected %llu, got %llu", kName,
	                            (unsigned long long)expected, (unsigned long long)got);
}

void OlsAnovaTableUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto entries = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child_vec = ListVector::GetEntry(inputs[1]);
	auto child = FlatDataMut<double>(child_vec);
	auto &child_validity = FlatValidityMut(child_vec);
	auto states = (OlsAnovaTableState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx) ||
		    ListHasNullElement(child_validity, entries[x_idx])) {
			continue;
		}
		auto &entry = entries[x_idx];
		if (!state.initialized) {
			state.x.resize(entry.length);
			state.initialized = true;
		} else if (entry.length != state.x.size()) {
			ThrowWidthMismatch(state.x.size(), entry.length);
		}
		state.y_values.push_back(y_values[y_idx]);
		for (idx_t j = 0; j < entry.length; j++) {
			state.x[j].push_back(child[entry.offset + j]);
		}
	}
}

void OlsAnovaTableCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                          idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (OlsAnovaTableState **)sdata.data;
	auto targets = (OlsAnovaTableState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.x = CombineTake(source.x, aggr_input_data);
			target.initialized = true;
			continue;
		}
		if (target.x.size() != source.x.size()) {
			ThrowWidthMismatch(target.x.size(), source.x.size());
		}
		for (idx_t j = 0; j < target.x.size(); j++) {
			target.x[j].insert(target.x[j].end(), source.x[j].begin(), source.x[j].end());
		}
		target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
	}
}

//! Write a DOUBLE; non-finite (not applicable) becomes NULL.
void SetDouble(Vector &vec, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(vec)[row] = value;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

//! The term labels in table order (first appearance) and each column's term.
void ResolveTerms(const vector<string> &labels, idx_t n_columns, vector<string> &term_names,
                  vector<uint64_t> &term_of_column) {
	if (!labels.empty() && labels.size() != n_columns) {
		throw InvalidInputException("%s: terms has %llu labels for %llu x columns", kName,
		                            (unsigned long long)labels.size(), (unsigned long long)n_columns);
	}
	for (idx_t j = 0; j < n_columns; j++) {
		string label = labels.empty() ? "x" + std::to_string(j + 1) : labels[j];
		idx_t t = 0;
		while (t < term_names.size() && term_names[t] != label) {
			t++;
		}
		if (t == term_names.size()) {
			term_names.push_back(label);
		}
		term_of_column.push_back(static_cast<uint64_t>(t));
	}
}

void OlsAnovaTableFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<OlsAnovaTableBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (OlsAnovaTableState **)sdata.data;
	const char *method = bind.anova_type == 2 ? "type II ANOVA" : "type I ANOVA";

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		if (state.x.empty()) {
			throw InvalidInputException("%s: x must have at least one column", kName);
		}
		vector<string> term_names;
		vector<uint64_t> term_of_column;
		ResolveTerms(bind.terms, state.x.size(), term_names, term_of_column);

		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		vector<AnofoxDataArray> x;
		for (auto &col : state.x) {
			x.push_back({col.data(), nullptr, col.size()});
		}
		vector<AnofoxAnovaTerm> terms(term_names.size());
		AnofoxOlsAnovaTableResult res;
		AnofoxError error;
		if (!anofox_ols_anova_table(y, x.data(), x.size(), term_of_column.data(), terms.size(), bind.fit_intercept,
		                            bind.anova_type, terms.data(), &res, &error)) {
			ThrowUnlessDegenerate(kName, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		AnofoxTestSummaryFFI unified {};
		unified.estimate = unified.conf_low = unified.conf_high = unified.conf_level = unified.effect_size = NAN;
		unified.n = static_cast<int64_t>(res.n_observations);

		auto n_rows = static_cast<idx_t>(terms.size());
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_rows);
		auto &rows = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(rows);
		for (idx_t r = 0; r < n_rows; r++) {
			auto &term = terms[r];
			auto out = list_offset + r;
			idx_t k = 0;
			SetResultString(StructEntry(fields[k++]), out, term_names[r].c_str());
			SetResultString(StructEntry(fields[k++]), out, method);
			for (double v : {term.statistic, term.df, res.residual_df, term.p_value, term.sum_sq, term.mean_sq,
			                 res.residual_sum_sq, res.residual_mean_sq}) {
				SetDouble(StructEntry(fields[k++]), out, v);
			}
			FlatDataMut<int64_t>(StructEntry(fields[k++]))[out] = static_cast<int64_t>(res.n_observations);
			WriteTestResultFields(rows, out, k, unified);
		}
		ListVector::SetListSize(result, list_offset + n_rows);
		ListVector::GetData(result)[result_idx] = list_entry_t(list_offset, n_rows);
	}
}

unique_ptr<FunctionData> OlsAnovaTableBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<OlsAnovaTableBindData>();
	if (args.size() >= 3) {
		auto opts = AnovaTableMapOptions::ParseFromValue(EvaluateConstantOptions(context, *args[2], kName), kName);
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.anova_type.has_value()) {
			bind->anova_type = opts.anova_type.value();
		}
		if (opts.terms.has_value()) {
			bind->terms = opts.terms.value();
		}
	}
	SetReturnTypeOf(function, LogicalType::LIST(OlsAnovaTableRowType()));
	PostHogTelemetry::Instance().RecordFunctionCall(kName);
	return std::move(bind);
}

} // namespace

void RegisterOlsAnovaTableAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set(kName);
	const auto list = LogicalType::LIST(LogicalType::DOUBLE);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, list};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    kName, args, LogicalType::ANY, AggregateFunction::StateSize<OlsAnovaTableState>,
		    ANOFOX_AGG_INIT(OlsAnovaTableInitialize),
		    ANOFOX_GUARDED_UPDATE(OlsAnovaTableUpdate, OlsAnovaTableDestroy, OlsAnovaTableInitialize),
		    OlsAnovaTableCombine,
		    ANOFOX_GUARDED_FINALIZE(OlsAnovaTableFinalize, OlsAnovaTableDestroy, OlsAnovaTableInitialize), nullptr,
		    ANOFOX_AGG_BIND(OlsAnovaTableBind), OlsAnovaTableDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    "Per-term ANOVA table of an OLS fit per group: R's anova(lm(y ~ x)) (type I, sequential sums of squares in "
	    "the column order) or car::Anova(fit, type = 2) (type II, each term adjusted for all others). Returns one "
	    "list element per term: term, statistic (F), df1, df2 (residual df), p_value, sum_sq, mean_sq and the "
	    "residual sum of squares. Options: type ('I' default, 'II'), terms (one label per x column; equal labels "
	    "form one term, e.g. factor dummies), fit_intercept.";
	FunctionDescription d1;
	d1.description = description;
	d1.examples = {"ols_anova_table_agg(y, [price, promo, temp])"};
	d1.categories = {"regression", "hypothesis-test"};
	d1.parameter_names = {"y", "x"};
	d1.parameter_types = {LogicalType::DOUBLE, list};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = description;
	d2.examples = {"ols_anova_table_agg(y, [price, promo, temp], {'type': 'II', 'terms': ['price', 'promo', 'temp']})"};
	d2.categories = {"regression", "hypothesis-test"};
	d2.parameter_names = {"y", "x", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, list, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
