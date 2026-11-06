// Non-parametric survival analysis (#156):
//
//   kaplan_meier_agg(time, event [, conf_level [, conf_type]])
//       -> [{time, surv, conf_low, conf_high, n_risk, n_event, n_censor, std_err}]
//   log_rank_test_agg(time, event, group)   -- unified test shape (#159)
//
// event is BOOLEAN or numeric (non-zero = event observed, 0 = right-censored).
// Rows with a NULL time, event (or group) are skipped. The estimates come from
// anofox-statistics (R survival::survfit, survival::survdiff).
//
// Extended forms (#322), from anofox-regression's survival module:
//
//   kaplan_meier_agg(time, event, options) / (time, event, columns DOUBLE[], options)
//       -> [{time, surv, conf_low, conf_high, n_risk, n_event, n_censor, std_err, cumhaz, std_chaz}]
//          (counts DOUBLE: weighted sums)
//   log_rank_test_agg(time, event, group, options) / (time, event, group, columns DOUBLE[], options)
//
// Case weights, entry (left-truncation) times, cluster ids (KM) and strata
// (log-rank) are columns of `columns` named by their 1-based position in the
// options, as cox_fit_agg's strata / entry / cluster are columns of x. The
// positional forms above are unchanged; log_rank_test_agg only leaves
// anofox-statistics when rho != 0 or strata are given.

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/test_struct.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "two_group.hpp"
#include "canonical_order.hpp"

namespace duckdb {

// Must match KaplanMeierResultFFI (crates/anofox-stats-ffi/src/types.rs): a
// double, eight pointers, two sizes.
static_assert(offsetof(AnofoxKaplanMeierResult, time) == sizeof(double) &&
                  offsetof(AnofoxKaplanMeierResult, n) == sizeof(double) + 8 * sizeof(double *) + sizeof(size_t),
              "AnofoxKaplanMeierResult must match KaplanMeierResultFFI in crates/anofox-stats-ffi/src/types.rs");
// Must match SurvfitDataFFI / SurvfitResultFFI (crates/anofox-stats-ffi/src/survival_extras_ffi.rs).
static_assert(offsetof(AnofoxSurvfitData, n_rows) == 5 * sizeof(double *) &&
                  offsetof(AnofoxSurvfitResult, len) == 10 * sizeof(double *),
              "AnofoxSurvfit* must match crates/anofox-stats-ffi/src/survival_extras_ffi.rs");

//! Walk a MAP / STRUCT options value as (lower-case key, value) pairs.
template <class FN>
static void VisitSurvivalOptions(const Value &options, FN &&fn) {
	const Value v = UnwrapVariant(options);
	if (v.IsNull()) {
		return;
	}
	if (v.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(v)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv.size() != 2 || kv[0].IsNull()) {
				throw InvalidInputException("Invalid MAP option entry");
			}
			fn(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else if (v.type().id() == LogicalTypeId::STRUCT) {
		auto &children = StructValue::GetChildren(v);
		auto &types = StructType::GetChildTypes(v.type());
		for (idx_t i = 0; i < types.size(); i++) {
			fn(StringUtil::Lower(NameString(types[i].first)), UnwrapVariant(children[i]));
		}
	} else {
		throw InvalidInputException("Expected MAP or STRUCT type for options, got %s", v.type().ToString());
	}
}

static double SurvivalOptionNumber(const char *fn, const string &key, const Value &v) {
	try {
		return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
	}
}

static bool SurvivalOptionBool(const char *fn, const string &key, const Value &v) {
	try {
		return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a boolean, got '%s'", fn, key, v.ToString());
	}
}

//! A 1-based position in `columns`.
static idx_t SurvivalOptionColumn(const char *fn, const string &key, const Value &v, bool has_columns) {
	if (!has_columns) {
		throw InvalidInputException("%s: option '%s' names a column of the columns argument; pass the per-row "
		                            "values as a DOUBLE[] before the options",
		                            fn, key);
	}
	double d = SurvivalOptionNumber(fn, key, v);
	if (!(d >= 1.0) || d != std::floor(d) || d > 1e9) {
		throw InvalidInputException("%s: option '%s' must be a 1-based position in columns, got '%s'", fn, key,
		                            v.ToString());
	}
	return static_cast<idx_t>(d);
}

//! Append the rows' `columns` list (NaN for a NULL element) to a row-major
//! buffer; returns false for a NULL list (the row is skipped).
static bool ReadSurvivalColumns(const UnifiedVectorFormat &list_data, Vector &list_vector, idx_t i, idx_t &n_columns,
                                bool &initialized, vector<double> &out, const char *fn) {
	auto idx = list_data.sel->get_index(i);
	if (!list_data.validity.RowIsValid(idx)) {
		return false;
	}
	auto entry = UnifiedVectorFormat::GetData<list_entry_t>(list_data)[idx];
	if (!initialized) {
		n_columns = entry.length;
		initialized = true;
	} else if (entry.length != n_columns) {
		throw InvalidInputException("%s: inconsistent length of columns: expected %llu, got %llu", fn,
		                            (unsigned long long)n_columns, (unsigned long long)entry.length);
	}
	auto &child = ListVector::GetEntry(list_vector);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	for (idx_t j = 0; j < entry.length; j++) {
		auto c = entry.offset + j;
		out.push_back(child_validity.RowIsValid(c) ? child_data[c] : std::numeric_limits<double>::quiet_NaN());
	}
	return true;
}

//! Column `pos` (1-based) of a row-major buffer with `n_columns` columns.
static vector<double> SurvivalColumn(const vector<double> &rows, idx_t n_columns, idx_t pos) {
	const idx_t n = n_columns == 0 ? 0 : rows.size() / n_columns;
	vector<double> col(n);
	for (idx_t r = 0; r < n; r++) {
		col[r] = rows[r * n_columns + pos - 1];
	}
	return col;
}

static void CheckSurvivalColumns(const char *fn, idx_t n_columns, std::initializer_list<idx_t> used) {
	for (auto c : used) {
		if (c > n_columns) {
			throw InvalidInputException("%s: option column %llu is beyond the %llu values of columns", fn,
			                            (unsigned long long)c, (unsigned long long)n_columns);
		}
	}
}

//! Event indicator of row `idx` as 1.0 / 0.0, from a BOOLEAN or DOUBLE column.
static double ReadEvent(const UnifiedVectorFormat &data, idx_t idx, bool is_bool) {
	if (is_bool) {
		return UnifiedVectorFormat::GetData<bool>(data)[idx] ? 1.0 : 0.0;
	}
	return UnifiedVectorFormat::GetData<double>(data)[idx] != 0.0 ? 1.0 : 0.0;
}

//===--------------------------------------------------------------------===//
// kaplan_meier_agg
//===--------------------------------------------------------------------===//
struct KaplanMeierState {
	vector<double> time;
	vector<double> event;
	// Extended form with a columns argument: row-major, n_columns per row.
	vector<double> columns;
	idx_t n_columns = 0;
	bool columns_initialized = false;
};

struct KaplanMeierBindData : public FunctionData {
	double conf_level = 0.95;
	int32_t conf_type = 0; // 0 log (R default), 1 log-log, 2 plain
	// #322: the options forms go through anofox-regression's survfit.
	bool extended = false;
	bool has_columns = false;
	int32_t hazard = 0;  // 0 Nelson-Aalen, 1 Fleming-Harrington
	int32_t robust = -1; // -1 survfit's default, 0 Greenwood, 1 robust
	bool timefix = true;
	idx_t weight_col = 0, entry_col = 0, cluster_col = 0; // 1-based in columns; 0 = none

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<KaplanMeierBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<KaplanMeierBindData>();
		return conf_level == o.conf_level && conf_type == o.conf_type && extended == o.extended &&
		       has_columns == o.has_columns && hazard == o.hazard && robust == o.robust && timefix == o.timefix &&
		       weight_col == o.weight_col && entry_col == o.entry_col && cluster_col == o.cluster_col;
	}
};

static LogicalType KaplanMeierExtendedResultType() {
	child_list_t<LogicalType> children;
	for (auto name : {"time", "surv", "conf_low", "conf_high", "n_risk", "n_event", "n_censor", "std_err", "cumhaz",
	                  "std_chaz"}) {
		children.push_back(make_pair(name, LogicalType::DOUBLE));
	}
	return LogicalType::LIST(LogicalType::STRUCT(std::move(children)));
}

static LogicalType KaplanMeierResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("time", LogicalType::DOUBLE));
	children.push_back(make_pair("surv", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	children.push_back(make_pair("n_risk", LogicalType::BIGINT));
	children.push_back(make_pair("n_event", LogicalType::BIGINT));
	children.push_back(make_pair("n_censor", LogicalType::BIGINT));
	children.push_back(make_pair("std_err", LogicalType::DOUBLE));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(children)));
}

static void KaplanMeierInitialize(data_ptr_t state_p) {
	new (state_p) KaplanMeierState();
}

static void KaplanMeierDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (KaplanMeierState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~KaplanMeierState();
	}
}

static void KaplanMeierUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                              idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<KaplanMeierBindData>();
	UnifiedVectorFormat time_data, event_data, columns_data, sdata;
	inputs[0].ToUnifiedFormat(count, time_data);
	inputs[1].ToUnifiedFormat(count, event_data);
	if (bind.has_columns) {
		inputs[2].ToUnifiedFormat(count, columns_data);
	}
	state_vector.ToUnifiedFormat(count, sdata);
	auto times = UnifiedVectorFormat::GetData<double>(time_data);
	const bool event_is_bool = inputs[1].GetType().id() == LogicalTypeId::BOOLEAN;
	auto states = (KaplanMeierState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto t_idx = time_data.sel->get_index(i);
		auto e_idx = event_data.sel->get_index(i);
		if (!time_data.validity.RowIsValid(t_idx) || !event_data.validity.RowIsValid(e_idx) ||
		    std::isnan(times[t_idx])) {
			continue;
		}
		auto &state = *states[sdata.sel->get_index(i)];
		if (bind.has_columns && !ReadSurvivalColumns(columns_data, inputs[2], i, state.n_columns,
		                                             state.columns_initialized, state.columns, "kaplan_meier_agg")) {
			continue;
		}
		state.time.push_back(times[t_idx]);
		state.event.push_back(ReadEvent(event_data, e_idx, event_is_bool));
	}
}

static void KaplanMeierCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                               idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (KaplanMeierState **)source_data.data;
	auto targets = (KaplanMeierState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.columns_initialized && target.columns_initialized && source.n_columns != target.n_columns) {
			throw InvalidInputException("kaplan_meier_agg: inconsistent length of columns during combine");
		}
		if (source.columns_initialized && !target.columns_initialized) {
			target.n_columns = source.n_columns;
			target.columns_initialized = true;
		}
		if (target.time.empty()) {
			target.time = CombineTake(source.time, aggr_input_data);
			target.event = CombineTake(source.event, aggr_input_data);
			target.columns = CombineTake(source.columns, aggr_input_data);
		} else {
			target.time.insert(target.time.end(), source.time.begin(), source.time.end());
			target.event.insert(target.event.end(), source.event.begin(), source.event.end());
			target.columns.insert(target.columns.end(), source.columns.begin(), source.columns.end());
		}
	}
}

static void SetSurvNumber(Vector &vec, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<double>(vec)[row] = value;
	}
}

//! The options forms (#322): anofox-regression's survfit.
static void KaplanMeierExtendedFinalize(Vector &state_vector, const KaplanMeierBindData &bind, Vector &result,
                                        idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (KaplanMeierState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.time.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		vector<double> weight, entry, cluster;
		if (bind.has_columns) {
			CheckSurvivalColumns("kaplan_meier_agg", state.n_columns,
			                     {bind.weight_col, bind.entry_col, bind.cluster_col});
			// Canonical row order: bitwise-identical results whatever the thread count (#226).
			vector<vector<double>> cols(state.n_columns);
			for (idx_t j = 0; j < state.n_columns; j++) {
				cols[j] = SurvivalColumn(state.columns, state.n_columns, j + 1);
			}
			CanonicalRows().Keys(cols).Key(state.time).Key(state.event).Sort();
			auto pick = [&](idx_t pos, vector<double> &out) {
				if (pos != 0) {
					out = cols[pos - 1];
				}
			};
			pick(bind.weight_col, weight);
			pick(bind.entry_col, entry);
			pick(bind.cluster_col, cluster);
		} else {
			CanonicalRows().Key(state.time).Key(state.event).Sort();
		}
		AnofoxSurvfitData data {};
		data.time = state.time.data();
		data.event = state.event.data();
		data.entry = bind.entry_col != 0 ? entry.data() : nullptr;
		data.weights = bind.weight_col != 0 ? weight.data() : nullptr;
		data.cluster = bind.cluster_col != 0 ? cluster.data() : nullptr;
		data.n_rows = static_cast<size_t>(state.time.size());
		AnofoxSurvfitOptions options {bind.conf_level, bind.conf_type, bind.hazard, bind.robust, bind.timefix};
		AnofoxSurvfitResult km {};
		AnofoxError error;
		if (!anofox_survfit(&data, options, &km, &error)) {
			ThrowUnlessDegenerate("kaplan_meier_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto len = static_cast<idx_t>(km.len);
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + len);
		auto &fields = StructVector::GetEntries(ListVector::GetEntry(result));
		const double *columns[] = {km.time,    km.surv,    km.conf_low, km.conf_high, km.n_risk,
		                           km.n_event, km.n_censor, km.std_err, km.cumhaz,    km.std_chaz};
		for (idx_t k = 0; k < len; k++) {
			for (idx_t c = 0; c < 10; c++) {
				SetSurvNumber(StructEntry(fields[c]), list_offset + k, columns[c][k]);
			}
		}
		ListVector::SetListSize(result, list_offset + len);
		ListVector::GetData(result)[row] = list_entry_t(list_offset, len);
		anofox_free_survfit_result(&km);
	}
}

static void KaplanMeierFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<KaplanMeierBindData>();
	if (bind.extended) {
		KaplanMeierExtendedFinalize(state_vector, bind, result, count, offset);
		return;
	}
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (KaplanMeierState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.time.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Key(state.time).Key(state.event).Sort();
		AnofoxDataArray time_array {state.time.data(), nullptr, static_cast<size_t>(state.time.size())};
		AnofoxDataArray event_array {state.event.data(), nullptr, static_cast<size_t>(state.event.size())};
		AnofoxKaplanMeierResult km;
		AnofoxError error;
		if (!anofox_kaplan_meier(time_array, event_array, bind.conf_type, bind.conf_level, &km, &error)) {
			ThrowUnlessDegenerate("kaplan_meier_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto len = static_cast<idx_t>(km.len);
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + len);
		auto &fields = StructVector::GetEntries(ListVector::GetEntry(result));
		for (idx_t k = 0; k < len; k++) {
			auto out = list_offset + k;
			FlatDataMut<double>(StructEntry(fields[0]))[out] = km.time[k];
			SetSurvNumber(StructEntry(fields[1]), out, km.survival[k]);
			SetSurvNumber(StructEntry(fields[2]), out, km.conf_low[k]);
			SetSurvNumber(StructEntry(fields[3]), out, km.conf_high[k]);
			FlatDataMut<int64_t>(StructEntry(fields[4]))[out] = static_cast<int64_t>(km.n_risk[k]);
			FlatDataMut<int64_t>(StructEntry(fields[5]))[out] = static_cast<int64_t>(km.n_event[k]);
			FlatDataMut<int64_t>(StructEntry(fields[6]))[out] = static_cast<int64_t>(km.n_censor[k]);
			SetSurvNumber(StructEntry(fields[7]), out, km.std_err[k]);
		}
		ListVector::SetListSize(result, list_offset + len);
		ListVector::GetData(result)[row] = list_entry_t(list_offset, len);
		anofox_free_kaplan_meier_result(&km);
	}
}

static Value SurvivalConstant(ClientContext &context, Expression &expr, const char *what) {
	RequireResolvedParameters(expr);
	if (!expr.IsFoldable()) {
		throw InvalidInputException("kaplan_meier_agg: %s must be a constant", what);
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		throw InvalidInputException("kaplan_meier_agg: %s must not be NULL", what);
	}
	return value;
}

static int32_t ParseConfType(const string &raw) {
	auto v = StringUtil::Lower(raw);
	if (v == "log") {
		return 0;
	} else if (v == "log-log" || v == "loglog" || v == "log_log") {
		return 1;
	} else if (v == "plain") {
		return 2;
	}
	throw InvalidInputException("kaplan_meier_agg: unknown conf_type '%s' (expected log, log-log or plain)", raw);
}

static void ParseKaplanMeierOptions(const Value &value, KaplanMeierBindData &bind) {
	const char *fn = "kaplan_meier_agg";
	VisitSurvivalOptions(value, [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "conf_level" || key == "confidence_level") {
			double d = SurvivalOptionNumber(fn, key, v);
			if (!(d > 0.0 && d < 1.0)) {
				throw InvalidInputException("kaplan_meier_agg: conf_level must be in (0, 1), got %g", d);
			}
			bind.conf_level = d;
		} else if (key == "conf_type") {
			bind.conf_type = ParseConfType(v.ToString());
		} else if (key == "hazard" || key == "ctype") {
			auto h = StringUtil::Lower(v.ToString());
			if (h == "nelson-aalen" || h == "nelson_aalen" || h == "nelsonaalen" || h == "1") {
				bind.hazard = 0;
			} else if (h == "fleming-harrington" || h == "fleming_harrington" || h == "flemingharrington" ||
			           h == "2") {
				bind.hazard = 1;
			} else {
				throw InvalidInputException("kaplan_meier_agg: unknown hazard '%s' (expected 'nelson-aalen' or "
				                            "'fleming-harrington')",
				                            v.ToString());
			}
		} else if (key == "robust") {
			bind.robust = SurvivalOptionBool(fn, key, v) ? 1 : 0;
		} else if (key == "timefix") {
			bind.timefix = SurvivalOptionBool(fn, key, v);
		} else if (key == "weight" || key == "weights") {
			bind.weight_col = SurvivalOptionColumn(fn, key, v, bind.has_columns);
		} else if (key == "entry" || key == "start") {
			bind.entry_col = SurvivalOptionColumn(fn, key, v, bind.has_columns);
		} else if (key == "cluster" || key == "id") {
			bind.cluster_col = SurvivalOptionColumn(fn, key, v, bind.has_columns);
		} else {
			throw InvalidInputException("kaplan_meier_agg: unknown option '%s'. Supported options: conf_level, "
			                            "conf_type, hazard, robust, timefix, weight, entry, cluster",
			                            key);
		}
	});
	idx_t cols[] = {bind.weight_col, bind.entry_col, bind.cluster_col};
	for (idx_t a = 0; a < 3; a++) {
		for (idx_t b = a + 1; b < 3; b++) {
			if (cols[a] != 0 && cols[a] == cols[b]) {
				throw InvalidInputException("kaplan_meier_agg: weight, entry and cluster must name different "
				                            "columns");
			}
		}
	}
}

static unique_ptr<FunctionData> KaplanMeierBind(ClientContext &context, AggregateBindFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<KaplanMeierBindData>();
	// (time, event, options) / (time, event, columns, options): the extended forms (#322).
	// A numeric third argument is the positional conf_level of the original form.
	const bool third_numeric = arguments.size() >= 3 && ReturnTypeOf(*arguments[2]).IsNumeric();
	if (arguments.size() == 3 && !third_numeric) {
		bind->extended = true;
	} else if (arguments.size() == 4 && ReturnTypeOf(*arguments[2]).id() == LogicalTypeId::LIST) {
		bind->extended = true;
		bind->has_columns = true;
	}
	if (bind->extended) {
		auto &options = *arguments[arguments.size() - 1];
		ParseKaplanMeierOptions(EvaluateConstantOptions(context, options, "kaplan_meier_agg"), *bind);
		SetReturnTypeOf(function, KaplanMeierExtendedResultType());
		PostHogTelemetry::Instance().RecordFunctionCall("kaplan_meier_agg");
		return std::move(bind);
	}
	if (arguments.size() >= 3) {
		bind->conf_level = SurvivalConstant(context, *arguments[2], "conf_level").GetValue<double>();
		if (!(bind->conf_level > 0.0 && bind->conf_level < 1.0)) {
			throw InvalidInputException("kaplan_meier_agg: conf_level must be in (0, 1), got %g", bind->conf_level);
		}
	}
	if (arguments.size() >= 4) {
		bind->conf_type = ParseConfType(SurvivalConstant(context, *arguments[3], "conf_type").ToString());
	}
	SetReturnTypeOf(function, KaplanMeierResultType());
	PostHogTelemetry::Instance().RecordFunctionCall("kaplan_meier_agg");
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// log_rank_test_agg
//===--------------------------------------------------------------------===//
//! Relocatable: GroupLabel keeps strings behind a pointer.
struct LogRankState {
	vector<double> time;
	vector<double> event;
	vector<uint64_t> codes;
	vector<GroupLabel> labels;
	// Extended form with a columns argument: row-major, n_columns per row.
	vector<double> columns;
	idx_t n_columns = 0;
	bool columns_initialized = false;

	uint64_t Slot(const GroupLabel &label) {
		for (idx_t k = 0; k < labels.size(); k++) {
			if (labels[k] == label) {
				return k;
			}
		}
		labels.push_back(label);
		return labels.size() - 1;
	}

	//! Renumber the groups by sorted label and sort the rows (by group, time,
	//! event, then the extra columns), so the result is bitwise independent of
	//! the row order (#226).
	void Canonicalize(vector<vector<double>> &extra) {
		vector<idx_t> order(labels.size());
		std::iota(order.begin(), order.end(), idx_t(0));
		std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return labels[a] < labels[b]; });
		vector<uint64_t> rank(labels.size());
		vector<GroupLabel> sorted(labels.size());
		for (idx_t r = 0; r < order.size(); r++) {
			rank[order[r]] = r;
			sorted[r] = labels[order[r]];
		}
		labels = std::move(sorted);
		vector<double> group_key(codes.size());
		for (idx_t k = 0; k < codes.size(); k++) {
			codes[k] = rank[codes[k]];
			group_key[k] = static_cast<double>(codes[k]);
		}
		CanonicalRows().Key(group_key).Key(time).Key(event).Keys(extra).Carry(codes).Sort();
	}
};

static void LogRankInitialize(data_ptr_t state_p) {
	new (state_p) LogRankState();
}

static void LogRankDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (LogRankState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~LogRankState();
	}
}

struct LogRankBindData : public FunctionData {
	// #322: rho != 0 or strata go through anofox-regression's survdiff.
	double rho = 0.0;
	bool has_columns = false;
	idx_t strata_col = 0; // 1-based in columns; 0 = none

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LogRankBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<LogRankBindData>();
		return rho == o.rho && has_columns == o.has_columns && strata_col == o.strata_col;
	}
};

static void LogRankUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                          idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<LogRankBindData>();
	UnifiedVectorFormat time_data, event_data, group_data, columns_data, sdata;
	inputs[0].ToUnifiedFormat(count, time_data);
	inputs[1].ToUnifiedFormat(count, event_data);
	inputs[2].ToUnifiedFormat(count, group_data);
	if (bind.has_columns) {
		inputs[3].ToUnifiedFormat(count, columns_data);
	}
	state_vector.ToUnifiedFormat(count, sdata);
	auto times = UnifiedVectorFormat::GetData<double>(time_data);
	const bool event_is_bool = inputs[1].GetType().id() == LogicalTypeId::BOOLEAN;
	const bool group_is_string = inputs[2].GetType().id() == LogicalTypeId::VARCHAR;
	auto states = (LogRankState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto t_idx = time_data.sel->get_index(i);
		auto e_idx = event_data.sel->get_index(i);
		auto g_idx = group_data.sel->get_index(i);
		if (!time_data.validity.RowIsValid(t_idx) || !event_data.validity.RowIsValid(e_idx) ||
		    !group_data.validity.RowIsValid(g_idx) || std::isnan(times[t_idx])) {
			continue;
		}
		auto &state = *states[sdata.sel->get_index(i)];
		if (bind.has_columns && !ReadSurvivalColumns(columns_data, inputs[3], i, state.n_columns,
		                                             state.columns_initialized, state.columns, "log_rank_test_agg")) {
			continue;
		}
		state.time.push_back(times[t_idx]);
		state.event.push_back(ReadEvent(event_data, e_idx, event_is_bool));
		state.codes.push_back(state.Slot(ReadGroupLabel(group_data, g_idx, group_is_string)));
	}
}

static void LogRankCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                           idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (LogRankState **)source_data.data;
	auto targets = (LogRankState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.time.empty()) {
			continue;
		}
		if (source.columns_initialized && target.columns_initialized && source.n_columns != target.n_columns) {
			throw InvalidInputException("log_rank_test_agg: inconsistent length of columns during combine");
		}
		if (source.columns_initialized && !target.columns_initialized) {
			target.n_columns = source.n_columns;
			target.columns_initialized = true;
		}
		if (target.time.empty()) {
			target.time = CombineTake(source.time, aggr_input_data);
			target.event = CombineTake(source.event, aggr_input_data);
			target.codes = CombineTake(source.codes, aggr_input_data);
			target.labels = CombineTake(source.labels, aggr_input_data);
			target.columns = CombineTake(source.columns, aggr_input_data);
			continue;
		}
		target.columns.insert(target.columns.end(), source.columns.begin(), source.columns.end());
		vector<uint64_t> remap(source.labels.size());
		for (idx_t k = 0; k < source.labels.size(); k++) {
			remap[k] = target.Slot(source.labels[k]);
		}
		target.time.insert(target.time.end(), source.time.begin(), source.time.end());
		target.event.insert(target.event.end(), source.event.begin(), source.event.end());
		for (auto code : source.codes) {
			target.codes.push_back(remap[code]);
		}
	}
}

static void LogRankFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<LogRankBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (LogRankState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.labels.size() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		vector<vector<double>> cols;
		if (bind.has_columns) {
			CheckSurvivalColumns("log_rank_test_agg", state.n_columns, {bind.strata_col});
			for (idx_t j = 0; j < state.n_columns; j++) {
				cols.push_back(SurvivalColumn(state.columns, state.n_columns, j + 1));
			}
		}
		state.Canonicalize(cols);
		AnofoxDataArray time_array {state.time.data(), nullptr, static_cast<size_t>(state.time.size())};
		AnofoxDataArray event_array {state.event.data(), nullptr, static_cast<size_t>(state.event.size())};
		AnofoxTestSummaryFFI summary;
		AnofoxError error;
		// The original test (anofox-statistics) unless rho or strata need survdiff (#322).
		const bool legacy = bind.rho == 0.0 && bind.strata_col == 0;
		const bool ok = legacy ? anofox_log_rank(time_array, event_array, state.codes.data(), &summary, &error)
		                       : anofox_survdiff(time_array, event_array, state.codes.data(),
		                                         bind.strata_col != 0 ? cols[bind.strata_col - 1].data() : nullptr,
		                                         bind.rho, &summary, &error);
		if (!ok) {
			ThrowUnlessDegenerate("log_rank_test_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteTestResultFields(result, row, 0, summary);
	}
}

static unique_ptr<FunctionData> LogRankBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<LogRankBindData>();
	const char *fn = "log_rank_test_agg";
	if (arguments.size() >= 4) {
		bind->has_columns = arguments.size() == 5;
		auto options = EvaluateConstantOptions(context, *arguments[arguments.size() - 1], fn);
		VisitSurvivalOptions(options, [&](const string &key, const Value &v) {
			if (v.IsNull()) {
				return;
			}
			if (key == "rho") {
				double d = SurvivalOptionNumber(fn, key, v);
				if (!std::isfinite(d)) {
					throw InvalidInputException("log_rank_test_agg: rho must be finite, got %g", d);
				}
				bind->rho = d;
			} else if (key == "strata") {
				bind->strata_col = SurvivalOptionColumn(fn, key, v, bind->has_columns);
			} else {
				throw InvalidInputException("log_rank_test_agg: unknown option '%s'. Supported options: rho, strata",
				                            key);
			}
		});
	}
	child_list_t<LogicalType> children;
	AppendTestResultFields(children);
	SetReturnTypeOf(function, LogicalType::STRUCT(std::move(children)));
	PostHogTelemetry::Instance().RecordFunctionCall("log_rank_test_agg");
	return std::move(bind);
}

void RegisterSurvivalAggregateFunctions(ExtensionLoader &loader) {
	{
		AggregateFunctionSet set("kaplan_meier_agg");
		for (auto event_type : {LogicalType::DOUBLE, LogicalType::BOOLEAN}) {
			for (idx_t extra = 0; extra <= 2; extra++) {
				vector<LogicalType> args = {LogicalType::DOUBLE, event_type};
				if (extra >= 1) {
					args.push_back(LogicalType::DOUBLE);
				}
				if (extra >= 2) {
					args.push_back(LogicalType::VARCHAR);
				}
				set.AddFunction(AggregateFunction(
				    "kaplan_meier_agg", args, LogicalType::ANY, AggregateFunction::StateSize<KaplanMeierState>,
				    ANOFOX_AGG_INIT(KaplanMeierInitialize),
				    ANOFOX_GUARDED_UPDATE(KaplanMeierUpdate, KaplanMeierDestroy, KaplanMeierInitialize),
				    KaplanMeierCombine,
				    ANOFOX_GUARDED_FINALIZE(KaplanMeierFinalize, KaplanMeierDestroy, KaplanMeierInitialize), nullptr,
				    ANOFOX_AGG_BIND(KaplanMeierBind), KaplanMeierDestroy));
			}
			// #322: (time, event, options) and (time, event, columns DOUBLE[], options).
			for (bool with_columns : {false, true}) {
				vector<LogicalType> args = {LogicalType::DOUBLE, event_type};
				if (with_columns) {
					args.push_back(LogicalType::LIST(LogicalType::DOUBLE));
				}
				args.push_back(LogicalType::ANY);
				set.AddFunction(AggregateFunction(
				    "kaplan_meier_agg", args, LogicalType::ANY, AggregateFunction::StateSize<KaplanMeierState>,
				    ANOFOX_AGG_INIT(KaplanMeierInitialize),
				    ANOFOX_GUARDED_UPDATE(KaplanMeierUpdate, KaplanMeierDestroy, KaplanMeierInitialize),
				    KaplanMeierCombine,
				    ANOFOX_GUARDED_FINALIZE(KaplanMeierFinalize, KaplanMeierDestroy, KaplanMeierInitialize), nullptr,
				    ANOFOX_AGG_BIND(KaplanMeierBind), KaplanMeierDestroy));
			}
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description =
		    "Kaplan-Meier survival curve of right-censored durations (event true/1 = observed, false/0 = censored), "
		    "as R survival::survfit: one row per distinct time with surv, the pointwise confidence interval "
		    "conf_low/conf_high (Greenwood; conf_type 'log' [default], 'log-log' or 'plain'; conf_level default "
		    "0.95), n_risk, n_event, n_censor and the Greenwood std_err of surv.";
		d.examples = {"kaplan_meier_agg(duration, event, 0.95, 'log-log')"};
		d.categories = {"survival"};
		d.parameter_names = {"time", "event", "conf_level", "conf_type"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::DOUBLE, LogicalType::VARCHAR};
		info.descriptions.push_back(d);
		FunctionDescription d2 = d;
		d2.description =
		    "Kaplan-Meier curve with a MAP / STRUCT of options (R survfit, via anofox-regression): conf_level, "
		    "conf_type, hazard ('nelson-aalen' [default] or 'fleming-harrington'), robust, timefix; with a DOUBLE[] "
		    "of per-row columns, weight, entry (left truncation, Surv(entry, time, event)) and cluster name their "
		    "1-based positions. Adds cumhaz and std_chaz; the counts are DOUBLE (weighted sums).";
		d2.examples = {"kaplan_meier_agg(duration, event, [entry_day, w], {'entry': 1, 'weight': 2})"};
		d2.parameter_names = {"time", "event", "columns", "options"};
		d2.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE),
		                      LogicalType::ANY};
		info.descriptions.push_back(d2);
		// One description per arity: duckdb_functions() matches several descriptions by parameter count.
		FunctionDescription d0 = d;
		d0.parameter_names = {"time", "event"};
		d0.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY};
		d0.examples = {"kaplan_meier_agg(duration, event)"};
		info.descriptions.push_back(std::move(d0));
		FunctionDescription d3 = d2;
		d3.description = "Kaplan-Meier survival curve (R survival::survfit) with a confidence level (DOUBLE, the "
		                 "original form) or a MAP / STRUCT of options (conf_level, conf_type, hazard, robust, "
		                 "timefix; adds cumhaz and std_chaz).";
		d3.examples = {"kaplan_meier_agg(duration, event, {'hazard': 'fleming-harrington'})"};
		d3.parameter_names = {"time", "event", "options"};
		d3.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::ANY};
		info.descriptions.push_back(std::move(d3));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set("log_rank_test_agg");
		for (auto event_type : {LogicalType::DOUBLE, LogicalType::BOOLEAN}) {
			for (auto group_type : {LogicalType::VARCHAR, LogicalType::BIGINT}) {
				set.AddFunction(AggregateFunction(
				    "log_rank_test_agg", {LogicalType::DOUBLE, event_type, group_type}, LogicalType::ANY,
				    AggregateFunction::StateSize<LogRankState>, ANOFOX_AGG_INIT(LogRankInitialize),
				    ANOFOX_GUARDED_UPDATE(LogRankUpdate, LogRankDestroy, LogRankInitialize), LogRankCombine,
				    ANOFOX_GUARDED_FINALIZE(LogRankFinalize, LogRankDestroy, LogRankInitialize), nullptr, ANOFOX_AGG_BIND(LogRankBind),
				    LogRankDestroy));
				// #322: (time, event, group, options) and (time, event, group, columns DOUBLE[], options).
				for (bool with_columns : {false, true}) {
					vector<LogicalType> args = {LogicalType::DOUBLE, event_type, group_type};
					if (with_columns) {
						args.push_back(LogicalType::LIST(LogicalType::DOUBLE));
					}
					args.push_back(LogicalType::ANY);
					set.AddFunction(AggregateFunction(
					    "log_rank_test_agg", args, LogicalType::ANY, AggregateFunction::StateSize<LogRankState>,
					    ANOFOX_AGG_INIT(LogRankInitialize),
					    ANOFOX_GUARDED_UPDATE(LogRankUpdate, LogRankDestroy, LogRankInitialize), LogRankCombine,
					    ANOFOX_GUARDED_FINALIZE(LogRankFinalize, LogRankDestroy, LogRankInitialize), nullptr,
					    ANOFOX_AGG_BIND(LogRankBind), LogRankDestroy));
				}
			}
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = "Log-rank test of equal survival across two or more groups (R survival::survdiff, rho = 0): "
		                "statistic (chi-squared), df1 (groups - 1), p_value, n.";
		d.examples = {"log_rank_test_agg(duration, event, treatment)"};
		d.categories = {"survival", "hypothesis-testing"};
		d.parameter_names = {"time", "event", "group"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::ANY};
		info.descriptions.push_back(d);
		FunctionDescription d2 = d;
		d2.description = "G-rho test of equal survival across groups (R survival::survdiff(rho), via "
		                 "anofox-regression when rho != 0 or strata are given): option rho (0 = log-rank, 1 = "
		                 "Peto-Peto); with a DOUBLE[] of per-row columns, strata names the 1-based position of a "
		                 "stratum code (survdiff(... + strata(s))).";
		d2.examples = {"log_rank_test_agg(duration, event, treatment, [site], {'rho': 1, 'strata': 1})"};
		d2.parameter_names = {"time", "event", "group", "columns", "options"};
		d2.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::ANY,
		                      LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
		info.descriptions.push_back(d2);
		FunctionDescription d4 = d2;
		d4.description = "G-rho test of equal survival across groups (R survival::survdiff(rho)): option rho "
		                 "(0 = log-rank, 1 = Peto-Peto).";
		d4.examples = {"log_rank_test_agg(duration, event, treatment, {'rho': 1})"};
		d4.parameter_names = {"time", "event", "group", "options"};
		d4.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::ANY, LogicalType::ANY};
		info.descriptions.push_back(std::move(d4));
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
