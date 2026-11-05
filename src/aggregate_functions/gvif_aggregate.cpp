// gvif_agg: generalized variance inflation factors (Fox & Monette 1992) per
// term, as car::vif reports them for a model with multi-column terms (#307).
//
//   gvif_agg(x[, options])
//
// Returns LIST(STRUCT(term, df, gvif, gvif_adj, n_observations)), one element
// per term; gvif_adj is GVIF^(1/(2 df)). A row is skipped when the list or any
// list element is NULL; rows with a NaN are left out of the computation. By
// default every x column is its own term (labelled x1, x2, ...; gvif is then
// the ordinary VIF); the terms option labels the columns, and columns sharing a
// label form one term (e.g. the dummies of a factor), as in
// ols_anova_table_agg. The GVIF is computed by anofox-regression
// (diagnostics::generalized_vif); this file only collects rows, maps the labels
// to term indices and writes the list.

#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/canonical_order.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr const char *kName = "gvif_agg";

struct GvifState {
	vector<vector<double>> x;
	bool initialized = false;
};

struct GvifBindData : public FunctionData {
	//! One label per x column; empty: x1, x2, ...
	vector<string> terms;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<GvifBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return terms == other_p.Cast<GvifBindData>().terms;
	}
};

LogicalType GvifRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("df", LogicalType::BIGINT));
	c.push_back(make_pair("gvif", LogicalType::DOUBLE));
	c.push_back(make_pair("gvif_adj", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

void GvifInitialize(data_ptr_t state_p) {
	new (state_p) GvifState();
}

void GvifDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GvifState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~GvifState();
	}
}

void ThrowWidthMismatch(idx_t expected, idx_t got) {
	throw InvalidInputException("%s: inconsistent x length: expected %llu, got %llu", kName,
	                            (unsigned long long)expected, (unsigned long long)got);
}

void GvifUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat x_data, sdata;
	inputs[0].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto entries = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &child_vec = ListVector::GetEntry(inputs[0]);
	auto child = FlatDataMut<double>(child_vec);
	auto &child_validity = FlatValidityMut(child_vec);
	auto states = (GvifState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx) || ListHasNullElement(child_validity, entries[x_idx])) {
			continue;
		}
		auto &entry = entries[x_idx];
		if (!state.initialized) {
			state.x.resize(entry.length);
			state.initialized = true;
		} else if (entry.length != state.x.size()) {
			ThrowWidthMismatch(state.x.size(), entry.length);
		}
		for (idx_t j = 0; j < entry.length; j++) {
			state.x[j].push_back(child[entry.offset + j]);
		}
	}
}

void GvifCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (GvifState **)sdata.data;
	auto targets = (GvifState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
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
	}
}

//! The term labels in first-appearance order and each column's term.
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

void GvifFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                  idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<GvifBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GvifState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		// As vif_agg: NULL without two columns or three rows.
		if (!state.initialized || state.x.size() < 2 || state.x[0].size() < 3) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		vector<string> term_names;
		vector<uint64_t> term_of_column;
		ResolveTerms(bind.terms, state.x.size(), term_names, term_of_column);

		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Keys(state.x).Sort();

		vector<AnofoxDataArray> x;
		for (auto &col : state.x) {
			x.push_back({col.data(), nullptr, col.size()});
		}
		vector<AnofoxGvifTerm> terms(term_names.size());
		size_t n_observations = 0;
		AnofoxError error;
		if (!anofox_compute_gvif(x.data(), x.size(), term_of_column.data(), terms.size(), terms.data(),
		                         &n_observations, &error)) {
			ThrowUnlessDegenerate(kName, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		auto n_rows = static_cast<idx_t>(terms.size());
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_rows);
		auto &rows = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(rows);
		for (idx_t r = 0; r < n_rows; r++) {
			auto out = list_offset + r;
			SetResultString(StructEntry(fields[0]), out, term_names[r].c_str());
			FlatDataMut<int64_t>(StructEntry(fields[1]))[out] = static_cast<int64_t>(terms[r].df);
			FlatDataMut<double>(StructEntry(fields[2]))[out] = terms[r].gvif;
			FlatDataMut<double>(StructEntry(fields[3]))[out] = terms[r].gvif_adj;
			FlatDataMut<int64_t>(StructEntry(fields[4]))[out] = static_cast<int64_t>(n_observations);
		}
		ListVector::SetListSize(result, list_offset + n_rows);
		ListVector::GetData(result)[result_idx] = list_entry_t(list_offset, n_rows);
	}
}

unique_ptr<FunctionData> GvifBind(ClientContext &context, AggregateBindFunction &function,
                                  vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<GvifBindData>();
	if (args.size() >= 2) {
		auto opts = GvifMapOptions::ParseFromValue(EvaluateConstantOptions(context, *args[1], kName), kName);
		if (opts.terms.has_value()) {
			bind->terms = opts.terms.value();
		}
	}
	SetReturnTypeOf(function, LogicalType::LIST(GvifRowType()));
	PostHogTelemetry::Instance().RecordFunctionCall(kName);
	return std::move(bind);
}

} // namespace

void RegisterGvifAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set(kName);
	const auto list = LogicalType::LIST(LogicalType::DOUBLE);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {list};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		set.AddFunction(AggregateFunction(
		    kName, args, LogicalType::ANY, AggregateFunction::StateSize<GvifState>, ANOFOX_AGG_INIT(GvifInitialize),
		    ANOFOX_GUARDED_UPDATE(GvifUpdate, GvifDestroy, GvifInitialize), GvifCombine,
		    ANOFOX_GUARDED_FINALIZE(GvifFinalize, GvifDestroy, GvifInitialize), nullptr, ANOFOX_AGG_BIND(GvifBind),
		    GvifDestroy));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    "Generalized variance inflation factors (Fox & Monette 1992) per term, as car::vif reports them: one list "
	    "element per term with term, df (number of columns), gvif and gvif_adj = GVIF^(1/(2 df)), comparable across "
	    "terms of different size. Options: terms (one label per x column; equal labels form one term, e.g. the "
	    "dummies of a factor). Without terms every column is its own term and gvif is the ordinary VIF.";
	FunctionDescription d1;
	d1.description = description;
	d1.examples = {"gvif_agg([x1, x2, x3])"};
	d1.categories = {"regression-diagnostics"};
	d1.parameter_names = {"x"};
	d1.parameter_types = {list};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = description;
	d2.examples = {"gvif_agg([age, region_b, region_c], {'terms': ['age', 'region', 'region']})"};
	d2.categories = {"regression-diagnostics"};
	d2.parameter_names = {"x", "options"};
	d2.parameter_types = {list, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
