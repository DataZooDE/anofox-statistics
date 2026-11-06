// Collinearity and separation diagnostics (#306):
//
//   condition_diagnostic_agg(x [, options])   Belsley-Kuh-Welsch condition indices and
//                                             variance-decomposition proportions (perturb::colldiag)
//   separation_check_agg(y, x)                per-predictor separation of a binary y
//   count_sparsity_agg(y, x)                  all-zero responses on an active indicator (count y)
//   detect_separation_agg(y, x [, options])   exact LP separation check (detectseparation)
//
// A row is skipped when y, the x list or any list element is NULL or not finite.
// Options: {'fit_intercept': true} (the default) adds the intercept column to the
// design of condition_diagnostic_agg and detect_separation_agg. Every statistic is
// computed by anofox-regression (diagnostics::condition_diagnostic,
// variance_decomposition_proportions, check_binary_separation,
// check_count_sparsity, detect_separation); this file only collects rows and
// writes the STRUCT.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "../include/xy_rows_aggregate.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {

constexpr const char *kConditionName = "condition_diagnostic_agg";
constexpr const char *kSeparationName = "separation_check_agg";
constexpr const char *kSparsityName = "count_sparsity_agg";
constexpr const char *kDetectName = "detect_separation_agg";

enum class DiagKind : uint8_t { CONDITION, SEPARATION, SPARSITY, DETECT };

struct DiagBindData : public FunctionData {
	DiagKind kind = DiagKind::CONDITION;
	bool fit_intercept = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<DiagBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<DiagBindData>();
		return kind == o.kind && fit_intercept == o.fit_intercept;
	}
};

const char *KindName(DiagKind kind) {
	switch (kind) {
	case DiagKind::CONDITION:
		return kConditionName;
	case DiagKind::SEPARATION:
		return kSeparationName;
	case DiagKind::SPARSITY:
		return kSparsityName;
	case DiagKind::DETECT:
	default:
		return kDetectName;
	}
}

LogicalType ResultType(DiagKind kind) {
	const auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	switch (kind) {
	case DiagKind::CONDITION:
		c.push_back(make_pair("condition_number", LogicalType::DOUBLE));
		c.push_back(make_pair("condition_number_xtx", LogicalType::DOUBLE));
		c.push_back(make_pair("scaled_condition_number", LogicalType::DOUBLE));
		c.push_back(make_pair("singular_values", dlist));
		c.push_back(make_pair("condition_indices", dlist));
		c.push_back(make_pair("variance_proportions", LogicalType::LIST(dlist)));
		c.push_back(make_pair("severity", LogicalType::VARCHAR));
		c.push_back(make_pair("warning", LogicalType::VARCHAR));
		break;
	case DiagKind::SEPARATION:
	case DiagKind::SPARSITY:
		c.push_back(make_pair("has_separation", LogicalType::BOOLEAN));
		c.push_back(make_pair("separated_predictors", LogicalType::LIST(LogicalType::BIGINT)));
		c.push_back(make_pair("separation_types", LogicalType::LIST(LogicalType::VARCHAR)));
		c.push_back(make_pair("warning", LogicalType::VARCHAR));
		break;
	case DiagKind::DETECT:
		c.push_back(make_pair("separated", LogicalType::BOOLEAN));
		c.push_back(make_pair("infinite", dlist));
		c.push_back(make_pair("direction", dlist));
		break;
	}
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(c));
}

//! Update for (x LIST(DOUBLE)): collects the x rows into XyRowsState::x_columns.
void XOnlyUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat x_data, sdata;
	inputs[0].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[0]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	auto states = (XyRowsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		auto entry = x_list_data[x_idx];
		if (ListHasNullElement(x_child_validity, entry)) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(state.n_features);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            entry.length);
		}
		for (idx_t j = 0; j < state.n_features; j++) {
			state.x_columns[j].push_back(x_child_data[entry.offset + j]);
		}
	}
}

//! Number of collected rows (x_columns may be empty for x = []).
idx_t RowCount(const XyRowsState &state, bool has_y) {
	if (has_y) {
		return state.y_values.size();
	}
	return state.x_columns.empty() ? 0 : state.x_columns[0].size();
}

//! Drop the rows holding a non-finite value (NaN / +-inf) in y or x.
void DropNonFiniteRows(XyRowsState &state, bool has_y) {
	idx_t n = RowCount(state, has_y);
	idx_t kept = 0;
	for (idx_t i = 0; i < n; i++) {
		bool ok = !has_y || std::isfinite(state.y_values[i]);
		for (idx_t j = 0; ok && j < state.x_columns.size(); j++) {
			ok = std::isfinite(state.x_columns[j][i]);
		}
		if (!ok) {
			continue;
		}
		if (has_y) {
			state.y_values[kept] = state.y_values[i];
		}
		for (auto &col : state.x_columns) {
			col[kept] = col[i];
		}
		kept++;
	}
	if (has_y) {
		state.y_values.resize(kept);
	}
	for (auto &col : state.x_columns) {
		col.resize(kept);
	}
}

//! Write a LIST(DOUBLE) value keeping +-inf (a singular design, a diverging
//! coefficient); NaN becomes a NULL element.
void SetDoubleList(Vector &list_vec, idx_t row, const double *values, idx_t n) {
	auto list_data = FlatDataMut<list_entry_t>(list_vec);
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + n);
	auto &child = ListVector::GetEntry(list_vec);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	for (idx_t j = 0; j < n; j++) {
		if (std::isnan(values[j])) {
			child_validity.SetInvalid(offset + j);
		} else {
			child_data[offset + j] = values[j];
		}
	}
	list_data[row].offset = offset;
	list_data[row].length = n;
	ListVector::SetListSize(list_vec, offset + n);
}

//! Write a DOUBLE keeping +-inf; NaN becomes NULL.
void SetDouble(Vector &vec, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<double>(vec)[row] = value;
	}
}

const char *SeverityName(int32_t code) {
	switch (code) {
	case 0:
		return "well_conditioned";
	case 1:
		return "moderate";
	case 2:
		return "high";
	default:
		return "severe";
	}
}

const char *SeparationTypeName(int32_t code) {
	switch (code) {
	case 1:
		return "complete";
	case 2:
		return "quasi";
	case 3:
		return "monotonic_response";
	default:
		return "none";
	}
}

void WriteCondition(StructEntries &entries, idx_t row, const AnofoxConditionDiagnostic &r) {
	idx_t k = 0;
	SetDouble(StructEntry(entries[k++]), row, r.condition_number);
	SetDouble(StructEntry(entries[k++]), row, r.condition_number_xtx);
	SetDouble(StructEntry(entries[k++]), row, r.scaled_condition_number);
	SetDoubleList(StructEntry(entries[k++]), row, r.singular_values, r.n_components);
	SetDoubleList(StructEntry(entries[k++]), row, r.condition_indices, r.n_components);
	// LIST(LIST(DOUBLE)): one inner list per component.
	auto &outer = StructEntry(entries[k++]);
	auto outer_data = FlatDataMut<list_entry_t>(outer);
	auto outer_offset = ListVector::GetListSize(outer);
	ListVector::Reserve(outer, outer_offset + r.n_components);
	auto &inner = ListVector::GetEntry(outer);
	for (idx_t c = 0; c < r.n_components; c++) {
		SetDoubleList(inner, outer_offset + c, r.variance_proportions + c * r.n_terms, r.n_terms);
	}
	outer_data[row].offset = outer_offset;
	outer_data[row].length = r.n_components;
	ListVector::SetListSize(outer, outer_offset + r.n_components);
	SetResultString(StructEntry(entries[k++]), row, SeverityName(r.severity));
	SetResultString(StructEntry(entries[k++]), row, r.warning);
}

void WriteSeparationCheck(StructEntries &entries, idx_t row, const AnofoxSeparationCheck &r) {
	idx_t k = 0;
	FlatDataMut<bool>(StructEntry(entries[k++]))[row] = r.has_separation;
	// 1-based positions in x, as DuckDB list indexing.
	auto &pred = StructEntry(entries[k++]);
	auto pred_data = FlatDataMut<list_entry_t>(pred);
	auto pred_offset = ListVector::GetListSize(pred);
	ListVector::Reserve(pred, pred_offset + r.n_separated);
	auto pred_child = FlatDataMut<int64_t>(ListVector::GetEntry(pred));
	for (idx_t j = 0; j < r.n_separated; j++) {
		pred_child[pred_offset + j] = (int64_t)r.separated_predictors[j] + 1;
	}
	pred_data[row] = {pred_offset, (idx_t)r.n_separated};
	ListVector::SetListSize(pred, pred_offset + r.n_separated);

	auto &types = StructEntry(entries[k++]);
	auto types_data = FlatDataMut<list_entry_t>(types);
	auto types_offset = ListVector::GetListSize(types);
	ListVector::Reserve(types, types_offset + r.n_separated);
	auto &types_child = ListVector::GetEntry(types);
	for (idx_t j = 0; j < r.n_separated; j++) {
		SetResultString(types_child, types_offset + j, SeparationTypeName(r.separation_types[j]));
	}
	types_data[row] = {types_offset, (idx_t)r.n_separated};
	ListVector::SetListSize(types, types_offset + r.n_separated);
	SetResultString(StructEntry(entries[k++]), row, r.warning);
}

void WriteLinearSeparation(StructEntries &entries, idx_t row, const AnofoxLinearSeparation &r) {
	idx_t k = 0;
	FlatDataMut<bool>(StructEntry(entries[k++]))[row] = r.separated;
	SetDoubleList(StructEntry(entries[k++]), row, r.infinite, r.n_terms);
	SetDoubleList(StructEntry(entries[k++]), row, r.direction, r.n_terms);
}

void DiagFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                  idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<DiagBindData>();
	const char *name = KindName(bind.kind);
	const bool has_y = bind.kind != DiagKind::CONDITION;
	XyRowsStatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	const idx_t n_fields = entries.size();

	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		DropNonFiniteRows(state, has_y);
		idx_t n_obs = RowCount(state, has_y);
		if (n_obs == 0) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		if (has_y) {
			CanonicalRows().Keys(state.x_columns).Key(state.y_values).Sort();
		} else {
			CanonicalRows().Keys(state.x_columns).Sort();
		}
		auto x = XyRowsColumnArrays(state);
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		const AnofoxDataArray *x_ptr = x.empty() ? nullptr : x.data();
		AnofoxError error;
		bool ok = false;
		switch (bind.kind) {
		case DiagKind::CONDITION: {
			AnofoxConditionDiagnostic r;
			ok = anofox_condition_diagnostic(x_ptr, x.size(), bind.fit_intercept, &r, &error);
			if (ok) {
				WriteCondition(entries, row, r);
				anofox_free_condition_diagnostic(&r);
			}
			break;
		}
		case DiagKind::SEPARATION:
		case DiagKind::SPARSITY: {
			AnofoxSeparationCheck r;
			ok = anofox_separation_check(bind.kind == DiagKind::SEPARATION ? 0 : 1, y, x_ptr, x.size(), &r,
			                             &error);
			if (ok) {
				WriteSeparationCheck(entries, row, r);
				anofox_free_separation_check(&r);
			}
			break;
		}
		case DiagKind::DETECT: {
			AnofoxLinearSeparation r;
			ok = anofox_detect_separation(y, x_ptr, x.size(), bind.fit_intercept, &r, &error);
			if (ok) {
				WriteLinearSeparation(entries, row, r);
				anofox_free_linear_separation(&r);
			}
			break;
		}
		}
		if (!ok) {
			ThrowUnlessDegenerate(name, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(entries[n_fields - 1]))[row] = (int64_t)n_obs;
	}
}

template <DiagKind KIND>
unique_ptr<FunctionData> DiagBind(ClientContext &context, AggregateBindFunction &function,
                                  vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<DiagBindData>();
	bind->kind = KIND;
	const char *name = KindName(KIND);
	const idx_t n_data_args = KIND == DiagKind::CONDITION ? 1 : 2;
	if (args.size() > n_data_args) {
		auto opts = RegressionMapOptions::ParseFromExpression(context, *args[n_data_args], name, {"fit_intercept"});
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
	}
	SetReturnTypeOf(function, ResultType(KIND));
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

void RegisterDiag(ExtensionLoader &loader, DiagKind kind, bool with_y, bool has_options, bind_aggregate_function_t bind,
                  const string &description, const string &example, const string &options_example) {
	const char *name = KindName(kind);
	const auto list = LogicalType::LIST(LogicalType::DOUBLE);
	AggregateFunctionSet set(name);
	vector<LogicalType> base = with_y ? vector<LogicalType> {LogicalType::DOUBLE, list} : vector<LogicalType> {list};
	vector<string> base_names = with_y ? vector<string> {"y", "x"} : vector<string> {"x"};
	for (bool with_options : {false, true}) {
		if (with_options && !has_options) {
			continue;
		}
		auto args = base;
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		if (with_y) {
			set.AddFunction(AggregateFunction(
			    name, args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
			    ANOFOX_AGG_INIT(XyRowsInitialize), ANOFOX_GUARDED_UPDATE(XyRowsUpdate, XyRowsDestroy, XyRowsInitialize),
			    XyRowsCombine, ANOFOX_GUARDED_FINALIZE(DiagFinalize, XyRowsDestroy, XyRowsInitialize), nullptr, bind,
			    XyRowsDestroy));
		} else {
			set.AddFunction(AggregateFunction(
			    name, args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
			    ANOFOX_AGG_INIT(XyRowsInitialize), ANOFOX_GUARDED_UPDATE(XOnlyUpdate, XyRowsDestroy, XyRowsInitialize),
			    XyRowsCombine, ANOFOX_GUARDED_FINALIZE(DiagFinalize, XyRowsDestroy, XyRowsInitialize), nullptr, bind,
			    XyRowsDestroy));
		}
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription d1;
	d1.description = description;
	d1.examples = {example};
	d1.categories = {"regression-diagnostics"};
	d1.parameter_names = base_names;
	d1.parameter_types = base;
	info.descriptions.push_back(std::move(d1));
	if (has_options) {
		FunctionDescription d2;
		d2.description = description;
		d2.examples = {options_example};
		d2.categories = {"regression-diagnostics"};
		d2.parameter_names = base_names;
		d2.parameter_names.push_back("options");
		d2.parameter_types = base;
		d2.parameter_types.push_back(LogicalType::ANY);
		info.descriptions.push_back(std::move(d2));
	}
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterCollinearitySeparationAggregateFunctions(ExtensionLoader &loader) {
	RegisterDiag(loader, DiagKind::CONDITION, false, true, ANOFOX_AGG_BIND(DiagBind<DiagKind::CONDITION>),
	             "Belsley-Kuh-Welsch collinearity diagnostics of the design [1, x] (perturb::colldiag, "
	             "olsrr::ols_eigen_cindex): condition indices and variance-decomposition proportions of the "
	             "unit-column-length design, plus kappa(X, exact = TRUE) of the unscaled design.",
	             "condition_diagnostic_agg([x1, x2, x3])",
	             "condition_diagnostic_agg([x1, x2, x3], {'fit_intercept': false})");
	RegisterDiag(loader, DiagKind::SEPARATION, true, false, ANOFOX_AGG_BIND(DiagBind<DiagKind::SEPARATION>),
	             "Per-predictor check for complete / quasi-separation of a binary (0/1) response, which makes "
	             "logistic coefficients diverge. Use detect_separation_agg for separation by a combination of "
	             "predictors.",
	             "separation_check_agg(y, [x1, x2])", "");
	RegisterDiag(loader, DiagKind::SPARSITY, true, false, ANOFOX_AGG_BIND(DiagBind<DiagKind::SPARSITY>),
	             "Count-data sparsity check for Poisson / negative binomial models: flags indicator predictors "
	             "whose active rows all have a zero response when more than 90% of the counts are zero.",
	             "count_sparsity_agg(y, [x1, x2])", "");
	RegisterDiag(loader, DiagKind::DETECT, true, true, ANOFOX_AGG_BIND(DiagBind<DiagKind::DETECT>),
	             "Exact check for (quasi-)complete separation of a binary (0/1) response by the linear program "
	             "of Konis (2007), as R's detectseparation::detect_separation: whether the logistic MLE exists, "
	             "and which coefficients diverge to +-Inf.",
	             "detect_separation_agg(y, [x1, x2])",
	             "detect_separation_agg(y, [x1, x2], {'fit_intercept': false})");
}

} // namespace duckdb
