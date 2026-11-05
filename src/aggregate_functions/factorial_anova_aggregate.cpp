// Two-way and repeated-measures ANOVA aggregates (#297).
//
//   two_way_anova_agg(y, factor_a, factor_b[, options])
//     LIST(STRUCT(term, method, statistic, df1, df2, p_value, sum_sq, mean_sq,
//     residual_sum_sq, residual_mean_sq, n_observations, <unified test fields>)):
//     one element each for factor A, factor B and the A x B interaction, with
//     Type III sums of squares (R's car::Anova(lm(y ~ a * b), type = 3) with sum
//     contrasts; equal to aov() for balanced designs). Options: terms (labels of
//     the two factors).
//
//   repeated_measures_anova_agg(y, subject, condition[, options])
//     STRUCT of the within-subjects F test (R's aov(y ~ condition +
//     Error(subject / condition))), Mauchly's sphericity test and the
//     Greenhouse-Geisser correction. Options: sphericity (default true).
//
// Factors, subjects and conditions are BIGINT or VARCHAR labels. A row is
// skipped when y is NULL or NaN or a label is NULL. The statistics are computed
// by anofox-statistics (parametric::two_way_anova, repeated_measures_anova);
// this file only collects the rows, codes the labels in sorted order (so the
// result does not depend on the row order) and writes the result.

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
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
#include "../include/two_group.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr const char *kTwoWay = "two_way_anova_agg";
constexpr const char *kRm = "repeated_measures_anova_agg";

//! Write a DOUBLE; non-finite (not applicable) becomes NULL.
void SetDouble(Vector &vec, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(vec)[row] = value;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

void SetBigint(Vector &vec, idx_t row, int64_t value) {
	FlatDataMut<int64_t>(vec)[row] = value;
}

//===--------------------------------------------------------------------===//
// two_way_anova_agg
//===--------------------------------------------------------------------===//

struct TwoWayAnovaState {
	vector<double> values;
	vector<double> codes_a;
	vector<double> codes_b;
	GroupCoder coder_a;
	GroupCoder coder_b;
	bool initialized = false;
};

struct TwoWayAnovaBindData : public FunctionData {
	vector<string> terms = {"factor_a", "factor_b"};

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TwoWayAnovaBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return terms == other_p.Cast<TwoWayAnovaBindData>().terms;
	}
};

LogicalType TwoWayAnovaRowType() {
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

//! Index of the first own (non-unified) field after p_value: sum_sq.
constexpr idx_t kTwoWaySumSqField = 6;

void TwoWayInitialize(data_ptr_t state_p) {
	new (state_p) TwoWayAnovaState();
}

void TwoWayDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (TwoWayAnovaState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~TwoWayAnovaState();
	}
}

void TwoWayUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, a_data, b_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, a_data);
	inputs[2].ToUnifiedFormat(count, b_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y = UnifiedVectorFormat::GetData<double>(y_data);
	const bool a_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;
	const bool b_is_string = inputs[2].GetType().id() == LogicalTypeId::VARCHAR;
	auto states = (TwoWayAnovaState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;
		auto y_idx = y_data.sel->get_index(i);
		auto a_idx = a_data.sel->get_index(i);
		auto b_idx = b_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !a_data.validity.RowIsValid(a_idx) ||
		    !b_data.validity.RowIsValid(b_idx) || std::isnan(y[y_idx])) {
			continue;
		}
		state.values.push_back(y[y_idx]);
		state.codes_a.push_back(state.coder_a.Code(ReadGroupLabel(a_data, a_idx, a_is_string)));
		state.codes_b.push_back(state.coder_b.Code(ReadGroupLabel(b_data, b_idx, b_is_string)));
	}
}

void TwoWayCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (TwoWayAnovaState **)sdata.data;
	auto targets = (TwoWayAnovaState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.values = CombineTake(source.values, aggr_input_data);
			target.codes_a = CombineTake(source.codes_a, aggr_input_data);
			target.codes_b = CombineTake(source.codes_b, aggr_input_data);
			target.coder_a.labels = CombineTake(source.coder_a.labels, aggr_input_data);
			target.coder_b.labels = CombineTake(source.coder_b.labels, aggr_input_data);
			target.initialized = true;
			continue;
		}
		target.values.insert(target.values.end(), source.values.begin(), source.values.end());
		auto remap_a = target.coder_a.Remap(source.coder_a);
		auto remap_b = target.coder_b.Remap(source.coder_b);
		for (auto code : source.codes_a) {
			target.codes_a.push_back(remap_a[static_cast<idx_t>(code)]);
		}
		for (auto code : source.codes_b) {
			target.codes_b.push_back(remap_b[static_cast<idx_t>(code)]);
		}
	}
}

void TwoWayFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                    idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<TwoWayAnovaBindData>();
	const string term_names[3] = {bind.terms[0], bind.terms[1], bind.terms[0] + ":" + bind.terms[1]};
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (TwoWayAnovaState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (!state.initialized || state.values.empty()) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		// Level codes in sorted label order: the result is independent of the row order.
		auto codes_a = state.codes_a;
		auto codes_b = state.codes_b;
		GroupCoder coder_a = state.coder_a;
		GroupCoder coder_b = state.coder_b;
		coder_a.Canonicalize(codes_a);
		coder_b.Canonicalize(codes_b);
		// Rows sorted by (level A, level B, value): the sums are then taken in the
		// same order whatever the row order, so the result is bitwise reproducible.
		vector<idx_t> order(state.values.size());
		std::iota(order.begin(), order.end(), idx_t(0));
		std::sort(order.begin(), order.end(), [&](idx_t l, idx_t r) {
			return std::tie(codes_a[l], codes_b[l], state.values[l]) <
			       std::tie(codes_a[r], codes_b[r], state.values[r]);
		});
		vector<double> values(order.size()), sorted_a(order.size()), sorted_b(order.size());
		for (idx_t k = 0; k < order.size(); k++) {
			values[k] = state.values[order[k]];
			sorted_a[k] = codes_a[order[k]];
			sorted_b[k] = codes_b[order[k]];
		}
		codes_a = std::move(sorted_a);
		codes_b = std::move(sorted_b);

		AnofoxDataArray y = {values.data(), nullptr, values.size()};
		AnofoxDataArray a = {codes_a.data(), nullptr, codes_a.size()};
		AnofoxDataArray b = {codes_b.data(), nullptr, codes_b.size()};
		AnofoxTwoWayAnovaResult res;
		AnofoxError error;
		if (!anofox_two_way_anova(y, a, b, &res, &error)) {
			ThrowUnlessDegenerate(kTwoWay, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		const idx_t n_rows = 3;
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_rows);
		auto &rows = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(rows);
		for (idx_t r = 0; r < n_rows; r++) {
			auto &term = res.terms[r];
			auto out = list_offset + r;
			SetResultString(StructEntry(fields[0]), out, term_names[r].c_str());
			// method, statistic, df1, df2, p_value and the appended unified fields.
			WriteTestResultFields(rows, out, 1, term.unified);
			idx_t k = kTwoWaySumSqField;
			for (double v : {term.sum_sq, term.mean_sq, res.residual_sum_sq, res.residual_mean_sq}) {
				SetDouble(StructEntry(fields[k++]), out, v);
			}
			SetBigint(StructEntry(fields[k++]), out, static_cast<int64_t>(res.n));
		}
		ListVector::SetListSize(result, list_offset + n_rows);
		ListVector::GetData(result)[result_idx] = list_entry_t(list_offset, n_rows);
	}
}

unique_ptr<FunctionData> TwoWayBind(ClientContext &context, AggregateBindFunction &function,
                                    vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<TwoWayAnovaBindData>();
	if (args.size() >= 4) {
		auto opts = TwoWayAnovaMapOptions::ParseFromValue(EvaluateConstantOptions(context, *args[3], kTwoWay), kTwoWay);
		if (opts.terms.has_value()) {
			bind->terms = opts.terms.value();
		}
	}
	SetReturnTypeOf(function, LogicalType::LIST(TwoWayAnovaRowType()));
	PostHogTelemetry::Instance().RecordFunctionCall(kTwoWay);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// repeated_measures_anova_agg
//===--------------------------------------------------------------------===//

struct RmAnovaState {
	vector<double> values;
	vector<GroupLabel> subjects;
	vector<GroupLabel> conditions;
	bool initialized = false;
};

struct RmAnovaBindData : public FunctionData {
	bool sphericity = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RmAnovaBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return sphericity == other_p.Cast<RmAnovaBindData>().sphericity;
	}
};

const char *const RM_OWN_FIELDS[] = {"ss_condition", "ss_subjects",
                                     "ss_error",     "ms_condition",
                                     "ms_error",     "n_subjects",
                                     "n_conditions", "n_excluded_subjects",
                                     "mauchly_w",    "mauchly_statistic",
                                     "mauchly_df",   "mauchly_p_value",
                                     "gg_epsilon",   "gg_df1",
                                     "gg_df2",       "gg_p_value"};

LogicalType RmAnovaResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("df2", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	for (auto name : RM_OWN_FIELDS) {
		string field(name);
		bool is_count = field == "n_subjects" || field == "n_conditions" || field == "n_excluded_subjects";
		c.push_back(make_pair(field, LogicalType(is_count ? LogicalTypeId::BIGINT : LogicalTypeId::DOUBLE)));
	}
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

//! Index of the first own field (ss_condition).
constexpr idx_t kRmOwnField = 5;

void RmInitialize(data_ptr_t state_p) {
	new (state_p) RmAnovaState();
}

void RmDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (RmAnovaState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~RmAnovaState();
	}
}

void RmUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, s_data, c_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, s_data);
	inputs[2].ToUnifiedFormat(count, c_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y = UnifiedVectorFormat::GetData<double>(y_data);
	const bool s_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;
	const bool c_is_string = inputs[2].GetType().id() == LogicalTypeId::VARCHAR;
	auto states = (RmAnovaState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;
		auto y_idx = y_data.sel->get_index(i);
		auto s_idx = s_data.sel->get_index(i);
		auto c_idx = c_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !s_data.validity.RowIsValid(s_idx) ||
		    !c_data.validity.RowIsValid(c_idx) || std::isnan(y[y_idx])) {
			continue;
		}
		state.values.push_back(y[y_idx]);
		state.subjects.push_back(ReadGroupLabel(s_data, s_idx, s_is_string));
		state.conditions.push_back(ReadGroupLabel(c_data, c_idx, c_is_string));
	}
}

void RmCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vector.ToUnifiedFormat(count, sdata);
	target_vector.ToUnifiedFormat(count, tdata);
	auto sources = (RmAnovaState **)sdata.data;
	auto targets = (RmAnovaState **)tdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.values = CombineTake(source.values, aggr_input_data);
			target.subjects = CombineTake(source.subjects, aggr_input_data);
			target.conditions = CombineTake(source.conditions, aggr_input_data);
			target.initialized = true;
			continue;
		}
		target.values.insert(target.values.end(), source.values.begin(), source.values.end());
		target.subjects.insert(target.subjects.end(), source.subjects.begin(), source.subjects.end());
		target.conditions.insert(target.conditions.end(), source.conditions.begin(), source.conditions.end());
	}
}

void RmFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count, idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<RmAnovaBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (RmAnovaState **)sdata.data;
	auto &fields = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (!state.initialized || state.values.empty()) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		// Subjects (rows) and conditions (columns) in sorted label order.
		std::map<GroupLabel, idx_t> subject_index, condition_index;
		for (auto &s : state.subjects) {
			subject_index[s] = 0;
		}
		for (auto &c : state.conditions) {
			condition_index[c] = 0;
		}
		idx_t next = 0;
		for (auto &entry : subject_index) {
			entry.second = next++;
		}
		next = 0;
		for (auto &entry : condition_index) {
			entry.second = next++;
		}
		const idx_t n_conditions = condition_index.size();
		vector<double> wide(subject_index.size() * n_conditions, std::numeric_limits<double>::quiet_NaN());
		for (idx_t j = 0; j < state.values.size(); j++) {
			auto &cell = wide[subject_index[state.subjects[j]] * n_conditions + condition_index[state.conditions[j]]];
			if (!std::isnan(cell)) {
				throw InvalidInputException("%s: subject %s has more than one value for condition %s; aggregate the "
				                            "replicates first (one row per subject and condition)",
				                            kRm, state.subjects[j].ToString(), state.conditions[j].ToString());
			}
			cell = state.values[j];
		}
		// A subject without a value for every condition is excluded (listwise, as afex / ez do).
		vector<double> complete;
		complete.reserve(wide.size());
		idx_t n_excluded = 0;
		for (idx_t s = 0; s < subject_index.size(); s++) {
			auto row = wide.begin() + static_cast<std::ptrdiff_t>(s * n_conditions);
			bool full = std::none_of(row, row + static_cast<std::ptrdiff_t>(n_conditions),
			                         [](double v) { return std::isnan(v); });
			if (full) {
				complete.insert(complete.end(), row, row + static_cast<std::ptrdiff_t>(n_conditions));
			} else {
				n_excluded++;
			}
		}
		const idx_t n_subjects = subject_index.size() - n_excluded;
		if (n_subjects < 2 || n_conditions < 2) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		AnofoxRmAnovaResult res;
		AnofoxError error;
		if (!anofox_repeated_measures_anova(complete.data(), n_subjects, n_conditions, bind.sphericity, &res, &error)) {
			ThrowUnlessDegenerate(kRm, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		// method, statistic, df1, df2, p_value and the appended unified fields.
		WriteTestResultFields(result, result_idx, 0, res.unified);
		idx_t k = kRmOwnField;
		for (double v : {res.ss_condition, res.ss_subjects, res.ss_error, res.ms_condition, res.ms_error}) {
			SetDouble(StructEntry(fields[k++]), result_idx, v);
		}
		SetBigint(StructEntry(fields[k++]), result_idx, static_cast<int64_t>(res.n_subjects));
		SetBigint(StructEntry(fields[k++]), result_idx, static_cast<int64_t>(res.n_conditions));
		SetBigint(StructEntry(fields[k++]), result_idx, static_cast<int64_t>(n_excluded));
		for (double v : {res.mauchly_w, res.mauchly_chi_square, res.mauchly_df, res.mauchly_p_value, res.gg_epsilon,
		                 res.gg_df1, res.gg_df2, res.gg_p_value}) {
			SetDouble(StructEntry(fields[k++]), result_idx, v);
		}
	}
}

unique_ptr<FunctionData> RmBind(ClientContext &context, AggregateBindFunction &function,
                                vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<RmAnovaBindData>();
	if (args.size() >= 4) {
		auto opts = RmAnovaMapOptions::ParseFromValue(EvaluateConstantOptions(context, *args[3], kRm), kRm);
		if (opts.sphericity.has_value()) {
			bind->sphericity = opts.sphericity.value();
		}
	}
	SetReturnTypeOf(function, RmAnovaResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kRm);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

//! Every argument list: y, two BIGINT / VARCHAR labels, optionally options.
vector<vector<LogicalType>> LabelOverloads() {
	const vector<LogicalType> label_types = {LogicalType::BIGINT, LogicalType::VARCHAR};
	vector<vector<LogicalType>> out;
	for (bool with_options : {false, true}) {
		for (auto &first : label_types) {
			for (auto &second : label_types) {
				vector<LogicalType> args = {LogicalType::DOUBLE, first, second};
				if (with_options) {
					args.push_back(LogicalType::ANY);
				}
				out.push_back(std::move(args));
			}
		}
	}
	return out;
}

void RegisterSet(ExtensionLoader &loader, AggregateFunctionSet set, const vector<string> &parameter_names,
                 const string &description, const vector<string> &examples) {
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	for (auto &args : LabelOverloads()) {
		FunctionDescription d;
		d.description = description;
		d.examples = examples;
		d.categories = {"hypothesis-testing", "anova"};
		d.parameter_names =
		    vector<string>(parameter_names.begin(), parameter_names.begin() + static_cast<std::ptrdiff_t>(args.size()));
		d.parameter_types = args;
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterFactorialAnovaAggregateFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet two_way(kTwoWay);
	for (auto &args : LabelOverloads()) {
		two_way.AddFunction(AggregateFunction(
		    kTwoWay, args, LogicalType::ANY, AggregateFunction::StateSize<TwoWayAnovaState>,
		    ANOFOX_AGG_INIT(TwoWayInitialize), ANOFOX_GUARDED_UPDATE(TwoWayUpdate, TwoWayDestroy, TwoWayInitialize),
		    TwoWayCombine, ANOFOX_GUARDED_FINALIZE(TwoWayFinalize, TwoWayDestroy, TwoWayInitialize), nullptr,
		    ANOFOX_AGG_BIND(TwoWayBind), TwoWayDestroy));
	}
	RegisterSet(loader, std::move(two_way), {"y", "factor_a", "factor_b", "options"},
	            "Two-way ANOVA with interaction: one list element each for factor A, factor B and the A x B "
	            "interaction, with Type III sums of squares (R's car::Anova(lm(y ~ a * b), type = 3) with sum "
	            "contrasts; equal to summary(aov()) for balanced designs). Fields: term, method, statistic (F), df1, "
	            "df2 (residual df), p_value, sum_sq, mean_sq, residual_sum_sq, residual_mean_sq, n_observations and "
	            "effect_size (partial eta squared). Every factor combination must be observed. Options: terms (labels "
	            "of the two factors, default ['factor_a', 'factor_b']).",
	            {"two_way_anova_agg(yield, fertilizer, variety)",
	             "two_way_anova_agg(yield, fertilizer, variety, {'terms': ['fertilizer', 'variety']})"});

	AggregateFunctionSet rm(kRm);
	for (auto &args : LabelOverloads()) {
		rm.AddFunction(AggregateFunction(
		    kRm, args, LogicalType::ANY, AggregateFunction::StateSize<RmAnovaState>, ANOFOX_AGG_INIT(RmInitialize),
		    ANOFOX_GUARDED_UPDATE(RmUpdate, RmDestroy, RmInitialize), RmCombine,
		    ANOFOX_GUARDED_FINALIZE(RmFinalize, RmDestroy, RmInitialize), nullptr, ANOFOX_AGG_BIND(RmBind), RmDestroy));
	}
	RegisterSet(loader, std::move(rm), {"y", "subject", "condition", "options"},
	            "One-way repeated-measures ANOVA of y over the conditions measured on every subject (long format, one "
	            "row per subject and condition): the within-subjects F test (R's aov(y ~ condition + Error(subject / "
	            "condition))), Mauchly's sphericity test and the Greenhouse-Geisser corrected p-value (three or more "
	            "conditions). Subjects missing a condition are excluded (n_excluded_subjects). Options: sphericity "
	            "(default true).",
	            {"repeated_measures_anova_agg(score, patient_id, visit)",
	             "repeated_measures_anova_agg(score, patient_id, visit, {'sphericity': false})"});
}

} // namespace duckdb
