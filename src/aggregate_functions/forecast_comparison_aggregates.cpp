// Multiple-model forecast comparison aggregates (#295):
//   spa_test_agg(benchmark_loss, model_losses DOUBLE[] [, options])
//   mspe_adjusted_spa_agg(benchmark_error, model_errors DOUBLE[] [, options])
//   model_confidence_set_agg(model_losses DOUBLE[] [, options])
// The statistics come from anofox-statistics (forecast::spa_test,
// forecast::mspe_adjusted_spa, forecast::model_confidence_set_with_method);
// this file only collects the series and marshals options and results.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/test_struct.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/result_fields.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "list_input.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// State: one series per model (and the benchmark for the SPA tests)
//===--------------------------------------------------------------------===//
struct ForecastComparisonState {
	vector<double> benchmark;
	vector<vector<double>> models;
	bool initialized;

	ForecastComparisonState() : initialized(false) {
	}

	void Reset() {
		benchmark.clear();
		models.clear();
		initialized = false;
	}

	idx_t Rows() const {
		return models.empty() ? 0 : models[0].size();
	}
};

static void ForecastComparisonInitialize(data_ptr_t state_p) {
	new (state_p) ForecastComparisonState();
}

static void ForecastComparisonDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ForecastComparisonState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.~ForecastComparisonState();
	}
}

static void CheckModelCount(const char *fn_name, idx_t expected, idx_t got) {
	if (expected != got) {
		throw InvalidInputException("%s: every row must hold the same number of models, expected %llu, got %llu",
		                            fn_name, (unsigned long long)expected, (unsigned long long)got);
	}
}

//! Collect one row: rows with a NULL/NaN benchmark, a NULL list or a NULL/NaN
//! element are skipped, as the sibling forecast tests skip incomplete rows.
static void ForecastComparisonUpdateImpl(const char *fn_name, bool has_benchmark, Vector inputs[],
                                         Vector &state_vector, idx_t count) {
	Vector &list_input = inputs[has_benchmark ? 1 : 0];

	UnifiedVectorFormat bench_data, list_data;
	if (has_benchmark) {
		inputs[0].ToUnifiedFormat(count, bench_data);
	}
	list_input.ToUnifiedFormat(count, list_data);
	auto benches = has_benchmark ? UnifiedVectorFormat::GetData<double>(bench_data) : nullptr;
	auto lists = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child = ListVector::GetEntry(list_input);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ForecastComparisonState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;

		double bench = 0.0;
		if (has_benchmark) {
			auto b_idx = bench_data.sel->get_index(i);
			if (!bench_data.validity.RowIsValid(b_idx)) {
				continue;
			}
			bench = benches[b_idx];
			if (std::isnan(bench)) {
				continue;
			}
		}
		auto l_idx = list_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(l_idx)) {
			continue;
		}
		auto entry = lists[l_idx];
		if (ListHasNullElement(child_validity, entry)) {
			continue;
		}
		bool has_nan = false;
		for (idx_t j = 0; j < entry.length; j++) {
			if (std::isnan(child_data[entry.offset + j])) {
				has_nan = true;
				break;
			}
		}
		if (has_nan) {
			continue;
		}

		if (state.models.empty()) {
			if (entry.length == 0) {
				throw InvalidInputException("%s: the list of models is empty", fn_name);
			}
			state.models.resize(entry.length);
		}
		CheckModelCount(fn_name, state.models.size(), entry.length);

		if (has_benchmark) {
			state.benchmark.push_back(bench);
		}
		for (idx_t j = 0; j < entry.length; j++) {
			state.models[j].push_back(child_data[entry.offset + j]);
		}
	}
}

static void SpaAggUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	ForecastComparisonUpdateImpl("spa_test_agg", true, inputs, state_vector, count);
}

static void MspeSpaAggUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	ForecastComparisonUpdateImpl("mspe_adjusted_spa_agg", true, inputs, state_vector, count);
}

static void McsAggUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	ForecastComparisonUpdateImpl("model_confidence_set_agg", false, inputs, state_vector, count);
}

static void ForecastComparisonCombine(Vector &source_vector, Vector &target_vector,
                                      AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (ForecastComparisonState **)source_data.data;
	auto targets = (ForecastComparisonState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized || target.models.empty()) {
			target.benchmark = CombineTake(source.benchmark, aggr_input_data);
			target.models = CombineTake(source.models, aggr_input_data);
			target.initialized = true;
			continue;
		}
		if (source.models.empty()) {
			continue;
		}
		CheckModelCount("forecast comparison aggregate", target.models.size(), source.models.size());
		target.benchmark.insert(target.benchmark.end(), source.benchmark.begin(), source.benchmark.end());
		for (idx_t j = 0; j < target.models.size(); j++) {
			target.models[j].insert(target.models[j].end(), source.models[j].begin(), source.models[j].end());
		}
	}
}

static AnofoxDataArray ToDataArray(vector<double> &v) {
	AnofoxDataArray a;
	a.data = v.data();
	a.validity = nullptr;
	a.len = v.size();
	return a;
}

//===--------------------------------------------------------------------===//
// Bind data
//===--------------------------------------------------------------------===//
struct ForecastComparisonBindData : public FunctionData {
	double alpha = 0.1;
	idx_t n_bootstrap = 1000;
	double block_length = 0.0; // <= 0: T^(1/3)
	uint64_t seed = ANOFOX_DEFAULT_MC_SEED;
	int32_t statistic = 0; // MCS: 0 range, 1 max
	int32_t bootstrap = 0; // MCS: 0 stationary, 1 moving block, 2 circular block, 3 IID

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ForecastComparisonBindData>();
		copy->alpha = alpha;
		copy->n_bootstrap = n_bootstrap;
		copy->block_length = block_length;
		copy->seed = seed;
		copy->statistic = statistic;
		copy->bootstrap = bootstrap;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ForecastComparisonBindData>();
		return alpha == other.alpha && n_bootstrap == other.n_bootstrap && block_length == other.block_length &&
		       seed == other.seed && statistic == other.statistic && bootstrap == other.bootstrap;
	}
};

static void CheckBootstrapOptions(const char *fn_name, const ForecastComparisonBindData &bd) {
	if (bd.n_bootstrap == 0) {
		throw InvalidInputException("%s: n_bootstrap must be positive", fn_name);
	}
	if (!std::isfinite(bd.block_length)) {
		throw InvalidInputException("%s: block_length must be finite", fn_name);
	}
}

//===--------------------------------------------------------------------===//
// SPA tests
//===--------------------------------------------------------------------===//
static LogicalType GetSpaAggResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("statistic", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value_lower", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value_consistent", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value_upper", LogicalType::DOUBLE));
	children.push_back(make_pair("best_model", LogicalType::BIGINT));
	children.push_back(make_pair("n_models", LogicalType::BIGINT));
	children.push_back(make_pair("n", LogicalType::BIGINT));
	children.push_back(make_pair("n_bootstrap", LogicalType::BIGINT));
	children.push_back(make_pair("block_length", LogicalType::DOUBLE));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	children.push_back(make_pair("alternative", LogicalType::VARCHAR));
	AppendTestResultFields(children);
	return LogicalType::STRUCT(std::move(children));
}

static void SpaFinalizeImpl(const char *fn_name, int32_t kind, const char *method, Vector &state_vector,
                            AggregateInputData &aggr_input_data, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ForecastComparisonState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);
	auto &bind_data = aggr_input_data.bind_data->Cast<ForecastComparisonBindData>();

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;

		if (!state.initialized || state.Rows() < 3) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		vector<AnofoxDataArray> models;
		models.reserve(state.models.size());
		for (auto &m : state.models) {
			models.push_back(ToDataArray(m));
		}

		AnofoxSpaOptions options;
		options.n_bootstrap = bind_data.n_bootstrap;
		options.block_length = bind_data.block_length;
		options.seed = bind_data.seed;
		options.has_seed = true;

		AnofoxSpaResult r;
		AnofoxError error;
		bool ok = anofox_spa_test(kind, ToDataArray(state.benchmark), models.data(), models.size(), options, &r,
		                          &error);
		if (!ok) {
			ThrowUnlessDegenerate(fn_name, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		idx_t s = 0;
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.statistic;
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.p_value_consistent;
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.p_value_lower;
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.p_value_consistent;
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.p_value_upper;
		if (r.best_model < 0) {
			SetResultNull(StructEntry(struct_entries[s++]), result_idx);
		} else {
			// 1-based, like DuckDB list positions in model_losses
			FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = r.best_model + 1;
		}
		FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = static_cast<int64_t>(r.n_models);
		FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = static_cast<int64_t>(r.n);
		FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = static_cast<int64_t>(r.n_bootstrap);
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.block_length;
		SetResultString(StructEntry(struct_entries[s++]), result_idx, method);
		// H1: at least one competing model has a lower expected loss than the benchmark.
		SetResultString(StructEntry(struct_entries[s++]), result_idx, "greater");
		WriteTestResultFields(result, result_idx, s, r.unified);

		state.Reset();
	}
}

static void SpaAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	SpaFinalizeImpl("spa_test_agg", 0, "Superior predictive ability test", state_vector, aggr_input_data, result,
	                count, offset);
}

static void MspeSpaAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                               idx_t count, idx_t offset) {
	SpaFinalizeImpl("mspe_adjusted_spa_agg", 1, "MSPE-adjusted SPA test", state_vector, aggr_input_data, result,
	                count, offset);
}

static unique_ptr<FunctionData> SpaBindImpl(const char *fn_name, ClientContext &context,
                                            AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, GetSpaAggResultType());
	auto bind_data = make_uniq<ForecastComparisonBindData>();
	if (arguments.size() >= 3) {
		Value options_val = EvaluateConstantOptions(context, *arguments[2], fn_name);
		auto opts = SpaMapOptions::ParseFromValue(options_val, fn_name);
		if (opts.n_bootstrap.has_value()) {
			bind_data->n_bootstrap = opts.n_bootstrap.value();
		}
		if (opts.block_length.has_value()) {
			bind_data->block_length = opts.block_length.value();
		}
		if (opts.seed.has_value()) {
			bind_data->seed = opts.seed.value();
		}
	}
	CheckBootstrapOptions(fn_name, *bind_data);
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(bind_data);
}

static unique_ptr<FunctionData> SpaAggBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	return SpaBindImpl("spa_test_agg", context, function, arguments);
}

static unique_ptr<FunctionData> MspeSpaAggBind(ClientContext &context, AggregateBindFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	return SpaBindImpl("mspe_adjusted_spa_agg", context, function, arguments);
}

//===--------------------------------------------------------------------===//
// Model Confidence Set
//===--------------------------------------------------------------------===//
static LogicalType GetMcsStepType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("model", LogicalType::BIGINT));
	children.push_back(make_pair("p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("mcs_p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("eliminated", LogicalType::BOOLEAN));
	return LogicalType::STRUCT(std::move(children));
}

static LogicalType GetMcsAggResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("included_models", LogicalType::LIST(LogicalType::BIGINT)));
	children.push_back(make_pair("eliminated_models", LogicalType::LIST(LogicalType::BIGINT)));
	children.push_back(make_pair("mcs_p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("mcs_p_values", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("elimination_order", LogicalType::LIST(LogicalType::BIGINT)));
	children.push_back(make_pair("elimination_steps", LogicalType::LIST(GetMcsStepType())));
	children.push_back(make_pair("statistic_type", LogicalType::VARCHAR));
	children.push_back(make_pair("alpha", LogicalType::DOUBLE));
	children.push_back(make_pair("n_models", LogicalType::BIGINT));
	children.push_back(make_pair("n", LogicalType::BIGINT));
	children.push_back(make_pair("n_bootstrap", LogicalType::BIGINT));
	children.push_back(make_pair("block_length", LogicalType::DOUBLE));
	children.push_back(make_pair("bootstrap_method", LogicalType::VARCHAR));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(children));
}

static const char *McsBootstrapName(int32_t bootstrap) {
	switch (bootstrap) {
	case 1:
		return "moving_block";
	case 2:
		return "circular_block";
	case 3:
		return "iid";
	default:
		return "stationary";
	}
}

static void WriteDoubleList(Vector &list_vec, idx_t row, const vector<double> &values) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + values.size());
	auto &child = ListVector::GetEntry(list_vec);
	auto data = FlatDataMut<double>(child);
	for (idx_t j = 0; j < values.size(); j++) {
		data[offset + j] = values[j];
	}
	ListVector::SetListSize(list_vec, offset + values.size());
	ListVector::GetData(list_vec)[row] = {offset, (idx_t)values.size()};
}

static void WriteBigintList(Vector &list_vec, idx_t row, const vector<int64_t> &values) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + values.size());
	auto &child = ListVector::GetEntry(list_vec);
	auto data = FlatDataMut<int64_t>(child);
	for (idx_t j = 0; j < values.size(); j++) {
		data[offset + j] = values[j];
	}
	ListVector::SetListSize(list_vec, offset + values.size());
	ListVector::GetData(list_vec)[row] = {offset, (idx_t)values.size()};
}

static void McsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ForecastComparisonState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);
	auto &bind_data = aggr_input_data.bind_data->Cast<ForecastComparisonBindData>();

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;

		if (!state.initialized || state.Rows() < 3) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		const idx_t k = state.models.size();
		vector<AnofoxDataArray> models;
		models.reserve(k);
		for (auto &m : state.models) {
			models.push_back(ToDataArray(m));
		}

		AnofoxMcsOptions options;
		options.alpha = bind_data.alpha;
		options.n_bootstrap = bind_data.n_bootstrap;
		options.block_length = bind_data.block_length;
		options.seed = bind_data.seed;
		options.statistic = bind_data.statistic;
		options.bootstrap = bind_data.bootstrap;
		options.has_seed = true;

		vector<uint64_t> included(k);
		vector<AnofoxMcsStep> steps(k);
		vector<uint64_t> order(k);
		vector<double> mcs_p_values(k);
		AnofoxMcsResult r;
		AnofoxError error;
		bool ok = anofox_model_confidence_set(models.data(), k, options, included.data(), steps.data(), order.data(),
		                                      mcs_p_values.data(), &r, &error);
		if (!ok) {
			ThrowUnlessDegenerate("model_confidence_set_agg", error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		// Model positions are 1-based, like DuckDB list positions in model_losses.
		vector<int64_t> included_out, eliminated_out, order_out;
		for (idx_t j = 0; j < k; j++) {
			order_out.push_back(static_cast<int64_t>(order[j]) + 1);
		}
		for (idx_t j = 0; j < r.n_included; j++) {
			included_out.push_back(static_cast<int64_t>(included[j]) + 1);
		}
		for (idx_t j = 0; j < r.n_steps; j++) {
			if (steps[j].eliminated) {
				eliminated_out.push_back(static_cast<int64_t>(steps[j].model) + 1);
			}
		}

		idx_t s = 0;
		WriteBigintList(StructEntry(struct_entries[s++]), result_idx, included_out);
		WriteBigintList(StructEntry(struct_entries[s++]), result_idx, eliminated_out);
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.mcs_p_value;
		WriteDoubleList(StructEntry(struct_entries[s++]), result_idx, mcs_p_values);
		WriteBigintList(StructEntry(struct_entries[s++]), result_idx, order_out);

		{
			auto &steps_vec = StructEntry(struct_entries[s++]);
			auto list_offset = ListVector::GetListSize(steps_vec);
			ListVector::Reserve(steps_vec, list_offset + r.n_steps);
			auto &step_struct = ListVector::GetEntry(steps_vec);
			auto &fields = StructVector::GetEntries(step_struct);
			auto models_out = FlatDataMut<int64_t>(StructEntry(fields[0]));
			auto p_out = FlatDataMut<double>(StructEntry(fields[1]));
			auto mcs_p_out = FlatDataMut<double>(StructEntry(fields[2]));
			auto elim_out = FlatDataMut<bool>(StructEntry(fields[3]));
			for (idx_t j = 0; j < r.n_steps; j++) {
				models_out[list_offset + j] = static_cast<int64_t>(steps[j].model) + 1;
				p_out[list_offset + j] = steps[j].p_value;
				mcs_p_out[list_offset + j] = steps[j].mcs_p_value;
				elim_out[list_offset + j] = steps[j].eliminated;
			}
			ListVector::SetListSize(steps_vec, list_offset + r.n_steps);
			ListVector::GetData(steps_vec)[result_idx] = {list_offset, (idx_t)r.n_steps};
		}

		SetResultString(StructEntry(struct_entries[s++]), result_idx, bind_data.statistic == 1 ? "max" : "range");
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = bind_data.alpha;
		FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = static_cast<int64_t>(r.n_models);
		FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = static_cast<int64_t>(r.n);
		FlatDataMut<int64_t>(StructEntry(struct_entries[s++]))[result_idx] = static_cast<int64_t>(r.n_bootstrap);
		FlatDataMut<double>(StructEntry(struct_entries[s++]))[result_idx] = r.block_length;
		SetResultString(StructEntry(struct_entries[s++]), result_idx, McsBootstrapName(bind_data.bootstrap));
		SetResultString(StructEntry(struct_entries[s++]), result_idx, "Model confidence set");

		state.Reset();
	}
}

static unique_ptr<FunctionData> McsAggBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, GetMcsAggResultType());
	auto bind_data = make_uniq<ForecastComparisonBindData>();
	if (arguments.size() >= 2) {
		Value options_val = EvaluateConstantOptions(context, *arguments[1], "model_confidence_set_agg");
		auto opts = McsMapOptions::ParseFromValue(options_val, "model_confidence_set_agg");
		if (opts.alpha.has_value()) {
			bind_data->alpha = opts.alpha.value();
		}
		if (opts.n_bootstrap.has_value()) {
			bind_data->n_bootstrap = opts.n_bootstrap.value();
		}
		if (opts.block_length.has_value()) {
			bind_data->block_length = opts.block_length.value();
		}
		if (opts.seed.has_value()) {
			bind_data->seed = opts.seed.value();
		}
		if (opts.max_statistic.has_value()) {
			bind_data->statistic = opts.max_statistic.value() ? 1 : 0;
		}
		if (opts.method.has_value()) {
			const string &m = opts.method.value();
			bind_data->bootstrap = m == "moving_block" ? 1 : m == "circular_block" ? 2 : m == "iid" ? 3 : 0;
		}
	}
	if (!(bind_data->alpha > 0.0 && bind_data->alpha < 1.0)) {
		throw InvalidInputException("model_confidence_set_agg: alpha must be in (0, 1), got %g", bind_data->alpha);
	}
	CheckBootstrapOptions("model_confidence_set_agg", *bind_data);
	if ((bind_data->bootstrap == 1 || bind_data->bootstrap == 2) && bind_data->block_length > 0 &&
	    bind_data->block_length != std::floor(bind_data->block_length)) {
		throw InvalidInputException("model_confidence_set_agg: block_length must be a whole number for '%s', got %g",
		                            McsBootstrapName(bind_data->bootstrap), bind_data->block_length);
	}
	PostHogTelemetry::Instance().RecordFunctionCall("model_confidence_set_agg");
	return std::move(bind_data);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
#define ANOFOX_FORECAST_CMP_FN(NAME, ARGS, UPDATE, FINALIZE, BIND)                                                   \
	AggregateFunction(NAME, ARGS, LogicalType::ANY, AggregateFunction::StateSize<ForecastComparisonState>,            \
	                  ANOFOX_AGG_INIT(ForecastComparisonInitialize),                                                 \
	                  ANOFOX_GUARDED_UPDATE(UPDATE, ForecastComparisonDestroy, ForecastComparisonInitialize),        \
	                  ForecastComparisonCombine,                                                                     \
	                  ANOFOX_GUARDED_FINALIZE(FINALIZE, ForecastComparisonDestroy, ForecastComparisonInitialize),    \
	                  nullptr, ANOFOX_AGG_BIND(BIND), ForecastComparisonDestroy)

static void RegisterForecastComparisonSet(ExtensionLoader &loader, const string &name, AggregateFunction with_opts,
                                          AggregateFunction no_opts, const string &description,
                                          const vector<string> &param_names_no_opts,
                                          const vector<LogicalType> &param_types_no_opts, const string &example,
                                          const string &example_no_opts) {
	AggregateFunctionSet func_set(name);
	// Row order is part of the input (time series): honour `agg(... ORDER BY t)`.
	SetOrderDependentOf(with_opts, AggregateOrderDependent::ORDER_DEPENDENT);
	func_set.AddFunction(with_opts);
	SetOrderDependentOf(no_opts, AggregateOrderDependent::ORDER_DEPENDENT);
	func_set.AddFunction(no_opts);
	CreateAggregateFunctionInfo info(std::move(func_set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

	FunctionDescription d1;
	d1.description = description;
	d1.examples = {example};
	d1.categories = {"forecast-evaluation"};
	d1.parameter_names = param_names_no_opts;
	d1.parameter_names.push_back("options");
	d1.parameter_types = param_types_no_opts;
	d1.parameter_types.push_back(LogicalType::ANY);
	info.descriptions.push_back(std::move(d1));

	FunctionDescription d2;
	d2.description = description + " Uses default options.";
	d2.examples = {example_no_opts};
	d2.categories = {"forecast-evaluation"};
	d2.parameter_names = param_names_no_opts;
	d2.parameter_types = param_types_no_opts;
	info.descriptions.push_back(std::move(d2));

	loader.RegisterFunction(std::move(info));
}

void RegisterForecastComparisonAggregateFunctions(ExtensionLoader &loader) {
	const auto dbl = LogicalType::DOUBLE;
	const auto dbl_list = LogicalType::LIST(LogicalType::DOUBLE);

	RegisterForecastComparisonSet(
	    loader, "spa_test_agg",
	    ANOFOX_FORECAST_CMP_FN("spa_test_agg", (vector<LogicalType> {dbl, dbl_list, LogicalType::ANY}),
	                           SpaAggUpdate, SpaAggFinalize, SpaAggBind),
	    ANOFOX_FORECAST_CMP_FN("spa_test_agg", (vector<LogicalType> {dbl, dbl_list}), SpaAggUpdate, SpaAggFinalize,
	                           SpaAggBind),
	    "Hansen's (2005) Superior Predictive Ability test: is any competing model better than the benchmark?",
	    {"benchmark_loss", "model_losses"}, {dbl, dbl_list},
	    "spa_test_agg(benchmark_loss, [loss_a, loss_b] ORDER BY t, {'n_bootstrap': 999, 'block_length': 5, 'seed': 1})",
	    "spa_test_agg(benchmark_loss, [loss_a, loss_b] ORDER BY t)");

	RegisterForecastComparisonSet(
	    loader, "mspe_adjusted_spa_agg",
	    ANOFOX_FORECAST_CMP_FN("mspe_adjusted_spa_agg", (vector<LogicalType> {dbl, dbl_list, LogicalType::ANY}),
	                           MspeSpaAggUpdate, MspeSpaAggFinalize, MspeSpaAggBind),
	    ANOFOX_FORECAST_CMP_FN("mspe_adjusted_spa_agg", (vector<LogicalType> {dbl, dbl_list}), MspeSpaAggUpdate,
	                           MspeSpaAggFinalize, MspeSpaAggBind),
	    "MSPE-adjusted SPA test (Clark-McCracken): SPA bootstrap on Clark-West adjusted differentials of forecast "
	    "errors, for a benchmark nested in each alternative.",
	    {"benchmark_error", "model_errors"}, {dbl, dbl_list},
	    "mspe_adjusted_spa_agg(benchmark_error, [error_a, error_b] ORDER BY t, {'n_bootstrap': 999})",
	    "mspe_adjusted_spa_agg(benchmark_error, [error_a, error_b] ORDER BY t)");

	RegisterForecastComparisonSet(
	    loader, "model_confidence_set_agg",
	    ANOFOX_FORECAST_CMP_FN("model_confidence_set_agg", (vector<LogicalType> {dbl_list, LogicalType::ANY}),
	                           McsAggUpdate, McsAggFinalize, McsAggBind),
	    ANOFOX_FORECAST_CMP_FN("model_confidence_set_agg", (vector<LogicalType> {dbl_list}), McsAggUpdate,
	                           McsAggFinalize, McsAggBind),
	    "Hansen, Lunde and Nason's (2011) Model Confidence Set: the set of models that contains the best one with "
	    "confidence 1 - alpha, with the elimination order and the MCS p-values. Bootstrap 'method': 'stationary' "
	    "(default), 'moving_block' (R MCS::MCSprocedure(k = block_length)), 'circular_block' or 'iid'.",
	    {"model_losses"}, {dbl_list},
	    "model_confidence_set_agg([loss_a, loss_b, loss_c] ORDER BY t, {'alpha': 0.1, 'statistic': 'max', "
	    "'method': 'moving_block', 'block_length': 3})",
	    "model_confidence_set_agg([loss_a, loss_b, loss_c] ORDER BY t)");
}

} // namespace duckdb
