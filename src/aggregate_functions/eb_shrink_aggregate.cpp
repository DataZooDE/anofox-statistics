#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Empirical-Bayes shrinkage.
//
//   anofox_stats_eb_shrink_agg(estimate DOUBLE, se DOUBLE [, options MAP])
//
// Consumes per-group estimates that already exist -- typically the output of a
// GROUP BY fit -- and shrinks each toward the precision-weighted mean by an
// amount the data determines. The `shrunken` LIST comes back in input order, the
// same convention the *_fit_predict_agg functions use, so it can be UNNESTed or
// indexed by ROW_NUMBER().
//===--------------------------------------------------------------------===//
struct EbShrinkState {
	vector<double> estimates;
	vector<double> standard_errors;
	AnofoxTauMethod method;
	double tau_squared;

	EbShrinkState() : method(ANOFOX_TAU_DERSIMONIAN_LAIRD), tau_squared(std::numeric_limits<double>::quiet_NaN()) {
	}

	void Reset() {
		estimates.clear();
		standard_errors.clear();
	}
};

struct EbShrinkBindData : public FunctionData {
	AnofoxTauMethod method = ANOFOX_TAU_DERSIMONIAN_LAIRD;
	double tau_squared = std::numeric_limits<double>::quiet_NaN();

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<EbShrinkBindData>();
		result->method = method;
		result->tau_squared = tau_squared;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<EbShrinkBindData>();
		// NaN == NaN is false, so compare the "unset" case explicitly.
		bool tau_same = (std::isnan(tau_squared) && std::isnan(o.tau_squared)) || tau_squared == o.tau_squared;
		return method == o.method && tau_same;
	}
};

static LogicalType GetEbShrinkResultType() {
	child_list_t<LogicalType> group_children;
	group_children.push_back(make_pair("estimate", LogicalType::DOUBLE));
	group_children.push_back(make_pair("se", LogicalType::DOUBLE));
	group_children.push_back(make_pair("shrunken", LogicalType::DOUBLE));
	group_children.push_back(make_pair("shrunken_se", LogicalType::DOUBLE));
	group_children.push_back(make_pair("weight", LogicalType::DOUBLE));
	auto group_type = LogicalType::STRUCT(std::move(group_children));

	child_list_t<LogicalType> children;
	children.push_back(make_pair("mu", LogicalType::DOUBLE));
	children.push_back(make_pair("mu_se", LogicalType::DOUBLE));
	children.push_back(make_pair("tau_squared", LogicalType::DOUBLE));
	children.push_back(make_pair("i_squared", LogicalType::DOUBLE));
	children.push_back(make_pair("q", LogicalType::DOUBLE));
	children.push_back(make_pair("n_groups", LogicalType::BIGINT));
	children.push_back(make_pair("shrunken", LogicalType::LIST(group_type)));
	return LogicalType::STRUCT(std::move(children));
}

static void EbShrinkInitialize(const AggregateFunction &, data_ptr_t state_p) {
	new (state_p) EbShrinkState();
}

static void EbShrinkDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EbShrinkState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~EbShrinkState();
	}
}

static void EbShrinkUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<EbShrinkBindData>();

	UnifiedVectorFormat est_data, se_data;
	inputs[0].ToUnifiedFormat(count, est_data);
	inputs[1].ToUnifiedFormat(count, se_data);
	auto est_values = UnifiedVectorFormat::GetData<double>(est_data);
	auto se_values = UnifiedVectorFormat::GetData<double>(se_data);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EbShrinkState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.method = bind_data.method;
		state.tau_squared = bind_data.tau_squared;

		auto e_idx = est_data.sel->get_index(i);
		auto s_idx = se_data.sel->get_index(i);
		if (!est_data.validity.RowIsValid(e_idx) || !se_data.validity.RowIsValid(s_idx)) {
			continue;
		}
		state.estimates.push_back(est_values[e_idx]);
		state.standard_errors.push_back(se_values[s_idx]);
	}
}

static void EbShrinkCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (EbShrinkState **)source_data.data;
	auto targets = (EbShrinkState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.estimates.empty()) {
			continue;
		}
		if (target.estimates.empty()) {
			target.method = source.method;
			target.tau_squared = source.tau_squared;
		}
		target.estimates.insert(target.estimates.end(), source.estimates.begin(), source.estimates.end());
		target.standard_errors.insert(target.standard_errors.end(), source.standard_errors.begin(),
		                              source.standard_errors.end());
	}
}

static void EbShrinkFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EbShrinkState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;

		if (state.estimates.size() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		AnofoxDataArray est_array {state.estimates.data(), nullptr, state.estimates.size()};
		AnofoxDataArray se_array {state.standard_errors.data(), nullptr, state.standard_errors.size()};

		AnofoxEbShrinkOptions options {};
		options.method = state.method;
		options.tau_squared = state.tau_squared;

		AnofoxEbShrinkResult res {};
		AnofoxError error;
		if (!anofox_eb_shrink(est_array, se_array, options, &res, &error)) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}

		FlatVector::GetData<double>(*struct_entries[0])[row] = res.mu;
		FlatVector::GetData<double>(*struct_entries[1])[row] = res.mu_se;
		FlatVector::GetData<double>(*struct_entries[2])[row] = res.tau_squared;
		FlatVector::GetData<double>(*struct_entries[3])[row] = res.i_squared;
		FlatVector::GetData<double>(*struct_entries[4])[row] = res.q;
		FlatVector::GetData<int64_t>(*struct_entries[5])[row] = (int64_t)res.n_groups;

		// The per-group LIST(STRUCT(...)).
		auto &list_vec = *struct_entries[6];
		auto list_data = FlatVector::GetData<list_entry_t>(list_vec);
		auto child_offset = ListVector::GetListSize(list_vec);
		ListVector::Reserve(list_vec, child_offset + res.len);
		auto &child_struct = ListVector::GetEntry(list_vec);
		auto &child_entries = StructVector::GetEntries(child_struct);

		auto c_est = FlatVector::GetData<double>(*child_entries[0]);
		auto c_se = FlatVector::GetData<double>(*child_entries[1]);
		auto c_shr = FlatVector::GetData<double>(*child_entries[2]);
		auto c_sse = FlatVector::GetData<double>(*child_entries[3]);
		auto c_w = FlatVector::GetData<double>(*child_entries[4]);

		for (idx_t j = 0; j < res.len; j++) {
			c_est[child_offset + j] = res.estimate[j];
			c_se[child_offset + j] = res.se[j];
			c_shr[child_offset + j] = res.shrunken[j];
			c_sse[child_offset + j] = res.shrunken_se[j];
			c_w[child_offset + j] = res.weight[j];
		}
		ListVector::SetListSize(list_vec, child_offset + res.len);
		list_data[row].offset = child_offset;
		list_data[row].length = res.len;

		anofox_free_eb_shrink_result(&res);
		state.Reset();
	}
}

static unique_ptr<FunctionData> EbShrinkBind(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<EbShrinkBindData>();

	if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
		auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
		if (opts.tau_squared.has_value()) {
			result->tau_squared = opts.tau_squared.value();
		}
		if (opts.tau_method.has_value()) {
			result->method = opts.tau_method.value() ? ANOFOX_TAU_NONE : ANOFOX_TAU_DERSIMONIAN_LAIRD;
		}
	}

	function.return_type = GetEbShrinkResultType();
	PostHogTelemetry::Instance().RecordFunctionCall("eb_shrink_agg");
	return std::move(result);
}

void RegisterEbShrinkAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("anofox_stats_eb_shrink_agg");

	auto basic =
	    AggregateFunction("anofox_stats_eb_shrink_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::ANY,
	                      AggregateFunction::StateSize<EbShrinkState>, EbShrinkInitialize, EbShrinkUpdate,
	                      EbShrinkCombine, EbShrinkFinalize, nullptr, EbShrinkBind, EbShrinkDestroy);
	func_set.AddFunction(basic);

	auto with_opts =
	    AggregateFunction("anofox_stats_eb_shrink_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
	                      LogicalType::ANY, AggregateFunction::StateSize<EbShrinkState>, EbShrinkInitialize,
	                      EbShrinkUpdate, EbShrinkCombine, EbShrinkFinalize, nullptr, EbShrinkBind, EbShrinkDestroy);
	func_set.AddFunction(with_opts);

	CreateAggregateFunctionInfo info(func_set);
	loader.RegisterFunction(info);

	AggregateFunctionSet alias_set("eb_shrink_agg");
	alias_set.AddFunction(basic);
	alias_set.AddFunction(with_opts);
	CreateAggregateFunctionInfo alias_info(alias_set);
	alias_info.alias_of = "anofox_stats_eb_shrink_agg";
	loader.RegisterFunction(alias_info);
}

} // namespace duckdb
