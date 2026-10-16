// Pairwise post-hoc test aggregates (#159): pairwise_t_test_agg, tukey_hsd_agg
// and dunn_test_agg. Each collects (value, group) and returns one row per pair
// of groups as LIST(STRUCT(group1, group2, estimate, statistic, p_value, p_adj,
// conf_low, conf_high, method, df1, alternative, conf_level, p_adjust_method)),
// the `test` schema of the integration contract. The statistics come from the
// upstream anofox-statistics post-hoc functions.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/ffi_enum_converters.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "two_group.hpp"

namespace duckdb {

enum class PosthocKind : uint8_t { PAIRWISE_T, TUKEY_HSD, DUNN };

static std::string PosthocName(PosthocKind kind) {
	switch (kind) {
	case PosthocKind::PAIRWISE_T:
		return "pairwise_t_test_agg";
	case PosthocKind::TUKEY_HSD:
		return "tukey_hsd_agg";
	case PosthocKind::DUNN:
	default:
		return "dunn_test_agg";
	}
}

//! Observations split by group label. Labels are kept in first-seen order; the
//! finalizer sorts them. Relocatable: GroupLabel keeps strings behind a pointer.
struct PosthocState {
	std::vector<GroupLabel> labels;
	std::vector<std::vector<double>> samples;

	idx_t Slot(const GroupLabel &label) {
		for (idx_t k = 0; k < labels.size(); k++) {
			if (labels[k] == label) {
				return k;
			}
		}
		labels.push_back(label);
		samples.emplace_back();
		return labels.size() - 1;
	}
};

struct PosthocBindData : public FunctionData {
	PosthocKind kind = PosthocKind::PAIRWISE_T;
	string p_adjust = "holm";
	bool pool_sd = true;
	AnofoxAlternative alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
	double confidence_level = 0.95;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PosthocBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PosthocBindData>();
		return kind == o.kind && p_adjust == o.p_adjust && pool_sd == o.pool_sd && alternative == o.alternative &&
		       confidence_level == o.confidence_level;
	}
};

static LogicalType GetPosthocRowType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("group1", LogicalType::VARCHAR));
	children.push_back(make_pair("group2", LogicalType::VARCHAR));
	children.push_back(make_pair("estimate", LogicalType::DOUBLE));
	children.push_back(make_pair("statistic", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("p_adj", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	children.push_back(make_pair("df1", LogicalType::DOUBLE));
	children.push_back(make_pair("alternative", LogicalType::VARCHAR));
	children.push_back(make_pair("conf_level", LogicalType::DOUBLE));
	children.push_back(make_pair("p_adjust_method", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(children));
}

static void PosthocInitialize(data_ptr_t state_p) {
	new (state_p) PosthocState();
}

static void PosthocDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PosthocState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~PosthocState();
	}
}

static void PosthocUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat value_data, group_data, sdata;
	inputs[0].ToUnifiedFormat(count, value_data);
	inputs[1].ToUnifiedFormat(count, group_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);
	const bool group_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;
	auto states = (PosthocState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto v_idx = value_data.sel->get_index(i);
		auto g_idx = group_data.sel->get_index(i);
		if (!value_data.validity.RowIsValid(v_idx) || !group_data.validity.RowIsValid(g_idx)) {
			continue;
		}
		double v = values[v_idx];
		if (std::isnan(v)) {
			continue;
		}
		state.samples[state.Slot(ReadGroupLabel(group_data, g_idx, group_is_string))].push_back(v);
	}
}

static void PosthocCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                           idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (PosthocState **)source_data.data;
	auto targets = (PosthocState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		for (idx_t k = 0; k < source.labels.size(); k++) {
			auto &dst = target.samples[target.Slot(source.labels[k])];
			if (dst.empty()) {
				dst = CombineTake(source.samples[k], aggr_input_data);
			} else {
				dst.insert(dst.end(), source.samples[k].begin(), source.samples[k].end());
			}
		}
	}
}

static std::string LabelText(const GroupLabel &label) {
	return label.is_string ? *label.str_value : std::to_string(label.int_value);
}

static void SetName(Vector &vec, idx_t row, const char *value, size_t capacity) {
	auto len = strnlen(value, capacity);
	if (len == 0) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<string_t>(vec)[row] = StringVector::AddString(vec, value, len);
	}
}

static void SetNumber(Vector &vec, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<double>(vec)[row] = value;
	}
}

static void PosthocFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PosthocState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<PosthocBindData>();
	const std::string fn_name = PosthocName(bind.kind);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (state.labels.size() < 2) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		// Group codes in sorted label order, so upstream (which sorts the codes)
		// orders the comparisons by label, as R orders factor levels.
		std::vector<idx_t> order(state.labels.size());
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(),
		          [&](idx_t a, idx_t b) { return state.labels[a] < state.labels[b]; });
		std::vector<double> values;
		std::vector<uint64_t> codes;
		for (idx_t code = 0; code < order.size(); code++) {
			auto &sample = state.samples[order[code]];
			values.insert(values.end(), sample.begin(), sample.end());
			codes.insert(codes.end(), sample.size(), static_cast<uint64_t>(code));
		}

		AnofoxDataArray values_array;
		values_array.data = values.data();
		values_array.validity = nullptr;
		values_array.len = static_cast<size_t>(values.size());

		AnofoxPairwiseResult res;
		AnofoxError error;
		bool ok;
		switch (bind.kind) {
		case PosthocKind::PAIRWISE_T:
			ok = anofox_pairwise_t_test_with_conf_level(values_array, codes.data(), bind.pool_sd, bind.p_adjust.c_str(),
			                                            bind.alternative, bind.confidence_level, &res, &error);
			break;
		case PosthocKind::TUKEY_HSD:
			ok = anofox_tukey_hsd(values_array, codes.data(), bind.confidence_level, &res, &error);
			break;
		case PosthocKind::DUNN:
		default:
			ok = anofox_dunn_test(values_array, codes.data(), bind.p_adjust.c_str(), &res, &error);
			break;
		}
		if (!ok) {
			ThrowUnlessDegenerate(fn_name.c_str(), error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		auto n_rows = static_cast<idx_t>(res.len);
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_rows);
		auto &rows = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(rows);
		for (idx_t r = 0; r < n_rows; r++) {
			auto &row = res.rows[r];
			auto out = list_offset + r;
			auto g1 = LabelText(state.labels[order[static_cast<idx_t>(row.group1)]]);
			auto g2 = LabelText(state.labels[order[static_cast<idx_t>(row.group2)]]);
			FlatDataMut<string_t>(StructEntry(fields[0]))[out] = StringVector::AddString(StructEntry(fields[0]), g1);
			FlatDataMut<string_t>(StructEntry(fields[1]))[out] = StringVector::AddString(StructEntry(fields[1]), g2);
			SetNumber(StructEntry(fields[2]), out, row.estimate);
			SetNumber(StructEntry(fields[3]), out, row.statistic);
			SetNumber(StructEntry(fields[4]), out, row.p_value);
			SetNumber(StructEntry(fields[5]), out, row.p_adj);
			SetNumber(StructEntry(fields[6]), out, row.conf_low);
			SetNumber(StructEntry(fields[7]), out, row.conf_high);
			SetName(StructEntry(fields[8]), out, res.method, ANOFOX_TEST_METHOD_LEN);
			SetNumber(StructEntry(fields[9]), out, row.df);
			SetName(StructEntry(fields[10]), out, res.alternative, ANOFOX_TEST_NAME_LEN);
			SetNumber(StructEntry(fields[11]), out, res.conf_level);
			SetName(StructEntry(fields[12]), out, res.p_adjust_method, ANOFOX_TEST_NAME_LEN);
		}
		ListVector::SetListSize(result, list_offset + n_rows);
		ListVector::GetData(result)[result_idx] = list_entry_t(list_offset, n_rows);
		anofox_free_pairwise_result(&res);
	}
}

template <PosthocKind KIND>
static unique_ptr<FunctionData> PosthocBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, LogicalType::LIST(GetPosthocRowType()));
	auto bind = make_uniq<PosthocBindData>();
	bind->kind = KIND;
	const std::string fn_name = PosthocName(KIND);
	if (arguments.size() >= 3) {
		Value options_val = EvaluateConstantOptions(context, *arguments[2], fn_name);
		auto opts = PosthocMapOptions::ParseFromValue(options_val, fn_name, KIND != PosthocKind::TUKEY_HSD,
		                                              KIND == PosthocKind::PAIRWISE_T, KIND != PosthocKind::DUNN);
		if (opts.p_adjust) {
			bind->p_adjust = *opts.p_adjust;
		}
		if (opts.pool_sd) {
			bind->pool_sd = *opts.pool_sd;
		}
		if (opts.alternative) {
			bind->alternative = ConvertAlternative(*opts.alternative);
		}
		if (opts.confidence_level) {
			bind->confidence_level = *opts.confidence_level;
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(bind);
}

template <PosthocKind KIND>
static void RegisterPosthoc(ExtensionLoader &loader, const char *description, const char *example,
                            const char *options_doc) {
	const std::string name = PosthocName(KIND);
	AggregateFunctionSet set(AsDuckName(name));
	for (auto group_type : {LogicalType::VARCHAR, LogicalType::BIGINT}) {
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::DOUBLE, group_type};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			AggregateFunction fn(
			    AsDuckName(name), args, LogicalType::ANY, AggregateFunction::StateSize<PosthocState>, ANOFOX_AGG_INIT(PosthocInitialize),
			    ANOFOX_GUARDED_UPDATE(PosthocUpdate, PosthocDestroy, PosthocInitialize), PosthocCombine,
			    ANOFOX_GUARDED_FINALIZE(PosthocFinalize, PosthocDestroy, PosthocInitialize), nullptr,
			    ANOFOX_AGG_BIND(PosthocBind<KIND>), PosthocDestroy);
			set.AddFunction(fn);
		}
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription d;
	d.description = std::string(description) + " " + options_doc;
	d.examples = {example};
	d.categories = {"hypothesis-testing", "post-hoc"};
	d.parameter_names = {"value", "group", "options"};
	d.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(std::move(d));
	loader.RegisterFunction(std::move(info));
}

void RegisterPairwisePosthocAggregateFunctions(ExtensionLoader &loader) {
	RegisterPosthoc<PosthocKind::PAIRWISE_T>(
	    loader,
	    "Pairwise t-tests between all groups with multiple-comparison adjustment (R pairwise.t.test). Returns "
	    "one row per pair: group1, group2, estimate (mean of group2 - group1), statistic (t), p_value, p_adj, and "
	    "the confidence interval of estimate (conf_low, conf_high at conf_level) from the same SE and df: per "
	    "comparison with p_adjust none, Bonferroni-adjusted otherwise (as emmeans); one-sided for a one-sided "
	    "alternative.",
	    "pairwise_t_test_agg(value, group, {'p_adjust': 'holm', 'pool_sd': true})",
	    "Options: p_adjust (holm [default], hochberg, hommel, bonferroni, BH/fdr, BY, none), pool_sd (default true), "
	    "alternative (two_sided, less, greater; for group2 - group1), confidence_level (default 0.95).");
	RegisterPosthoc<PosthocKind::TUKEY_HSD>(
	    loader,
	    "Tukey's honest significant difference for all pairwise differences of group means (R "
	    "TukeyHSD(aov(value ~ group))). Returns one row per pair with the simultaneous confidence interval "
	    "(conf_low, conf_high) and family-wise p_adj.",
	    "tukey_hsd_agg(value, group, {'confidence_level': 0.95})", "Options: confidence_level (default 0.95).");
	RegisterPosthoc<PosthocKind::DUNN>(
	    loader,
	    "Dunn's test for all pairwise differences of mean ranks after Kruskal-Wallis (FSA::dunnTest). Returns one "
	    "row per pair: estimate (mean rank of group2 - group1), statistic (z for group2 - group1), p_value, p_adj.",
	    "dunn_test_agg(value, group, {'p_adjust': 'holm'})",
	    "Options: p_adjust (holm [default], hochberg, hommel, bonferroni, BH/fdr, BY, none).");
}

} // namespace duckdb
