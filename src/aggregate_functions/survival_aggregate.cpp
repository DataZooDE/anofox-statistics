// Non-parametric survival analysis (#156):
//
//   kaplan_meier_agg(time, event [, conf_level [, conf_type]])
//       -> [{time, surv, conf_low, conf_high, n_risk, n_event, n_censor, std_err}]
//   log_rank_test_agg(time, event, group)   -- unified test shape (#159)
//
// event is BOOLEAN or numeric (non-zero = event observed, 0 = right-censored).
// Rows with a NULL time, event (or group) are skipped. The estimates come from
// anofox-statistics (R survival::survfit, survival::survdiff).

#include <algorithm>
#include <cmath>
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
};

struct KaplanMeierBindData : public FunctionData {
	double conf_level = 0.95;
	int32_t conf_type = 0; // 0 log (R default), 1 log-log, 2 plain

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<KaplanMeierBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<KaplanMeierBindData>();
		return conf_level == o.conf_level && conf_type == o.conf_type;
	}
};

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

static void KaplanMeierUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat time_data, event_data, sdata;
	inputs[0].ToUnifiedFormat(count, time_data);
	inputs[1].ToUnifiedFormat(count, event_data);
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
		if (target.time.empty()) {
			target.time = CombineTake(source.time, aggr_input_data);
			target.event = CombineTake(source.event, aggr_input_data);
		} else {
			target.time.insert(target.time.end(), source.time.begin(), source.time.end());
			target.event.insert(target.event.end(), source.event.begin(), source.event.end());
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

static void KaplanMeierFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (KaplanMeierState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<KaplanMeierBindData>();

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
	if (!expr.IsFoldable()) {
		throw InvalidInputException("kaplan_meier_agg: %s must be a constant", what);
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		throw InvalidInputException("kaplan_meier_agg: %s must not be NULL", what);
	}
	return value;
}

static unique_ptr<FunctionData> KaplanMeierBind(ClientContext &context, AggregateBindFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<KaplanMeierBindData>();
	if (arguments.size() >= 3) {
		bind->conf_level = SurvivalConstant(context, *arguments[2], "conf_level").GetValue<double>();
		if (!(bind->conf_level > 0.0 && bind->conf_level < 1.0)) {
			throw InvalidInputException("kaplan_meier_agg: conf_level must be in (0, 1), got %g", bind->conf_level);
		}
	}
	if (arguments.size() >= 4) {
		auto raw = SurvivalConstant(context, *arguments[3], "conf_type").ToString();
		auto v = StringUtil::Lower(raw);
		if (v == "log") {
			bind->conf_type = 0;
		} else if (v == "log-log" || v == "loglog" || v == "log_log") {
			bind->conf_type = 1;
		} else if (v == "plain") {
			bind->conf_type = 2;
		} else {
			throw InvalidInputException("kaplan_meier_agg: unknown conf_type '%s' (expected log, log-log or plain)",
			                            raw);
		}
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
	//! event), so the result is bitwise independent of the row order (#226).
	void Canonicalize() {
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
		CanonicalRows().Key(group_key).Key(time).Key(event).Carry(codes).Sort();
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

static void LogRankUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat time_data, event_data, group_data, sdata;
	inputs[0].ToUnifiedFormat(count, time_data);
	inputs[1].ToUnifiedFormat(count, event_data);
	inputs[2].ToUnifiedFormat(count, group_data);
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
		if (target.time.empty()) {
			target.time = CombineTake(source.time, aggr_input_data);
			target.event = CombineTake(source.event, aggr_input_data);
			target.codes = CombineTake(source.codes, aggr_input_data);
			target.labels = CombineTake(source.labels, aggr_input_data);
			continue;
		}
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

static void LogRankFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
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
		state.Canonicalize();
		AnofoxDataArray time_array {state.time.data(), nullptr, static_cast<size_t>(state.time.size())};
		AnofoxDataArray event_array {state.event.data(), nullptr, static_cast<size_t>(state.event.size())};
		AnofoxTestSummaryFFI summary;
		AnofoxError error;
		if (!anofox_log_rank(time_array, event_array, state.codes.data(), &summary, &error)) {
			ThrowUnlessDegenerate("log_rank_test_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteTestResultFields(result, row, 0, summary);
	}
}

static unique_ptr<FunctionData> LogRankBind(ClientContext &, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &) {
	child_list_t<LogicalType> children;
	AppendTestResultFields(children);
	SetReturnTypeOf(function, LogicalType::STRUCT(std::move(children)));
	PostHogTelemetry::Instance().RecordFunctionCall("log_rank_test_agg");
	return nullptr;
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
		info.descriptions.push_back(std::move(d));
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
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
