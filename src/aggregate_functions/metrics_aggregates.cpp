// Metric pack (#368) and ranking metrics (#369): upstream anofox-statistics 0.10
// `metrics`, exposed as
//   classification_metrics_agg(y_true, y_pred [, options])
//     y_true / y_pred BIGINT, VARCHAR or BOOLEAN labels (multiclass), or y_pred a
//     DOUBLE score turned into the label 1 / true when score >= options.threshold;
//   regression_metrics_agg(y_true DOUBLE, y_pred DOUBLE);
//   ranking_metrics_agg(query_id, relevance DOUBLE, score DOUBLE [, options]);
// and the table macros classification_metrics_by, regression_metrics_by,
// ranking_metrics_by (one row per group) and data_quality_by (#369 part 2, SQL
// only: missing share, distinct count, new / dropped categories and out-of-range
// share against a reference table, basic stats).
//
// Rows with a NULL argument are skipped; no rows left gives NULL. Labels are coded
// densely in label order before the FFI call, so the results (and the label order
// of the confusion matrix) do not depend on the row order.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/anofox_statistics_extension.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "canonical_order.hpp"
#include "two_group.hpp"

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Shared helpers
//===--------------------------------------------------------------------===//

//! Walk a MAP / STRUCT options value as (lower-case key, value) pairs.
template <class FN>
void VisitOptions(const Value &options, FN &&fn) {
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

template <class T>
T OptionAs(const char *fn, const string &key, const Value &v, const LogicalType &type) {
	if (v.IsNull()) {
		throw InvalidInputException("%s: option '%s' must not be NULL", fn, key);
	}
	try {
		return v.DefaultCastAs(type).GetValue<T>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be %s, got '%s'", fn, key, type.ToString(), v.ToString());
	}
}

//! Write `v` to a DOUBLE child, NULL when NaN (an undefined metric).
inline void SetDoubleOrNull(Vector &vec, idx_t idx, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(vec, idx, true);
	} else {
		FlatDataMut<double>(vec)[idx] = v;
	}
}

AnofoxDataArray AsArray(const vector<double> &v) {
	AnofoxDataArray a;
	a.data = v.data();
	a.validity = nullptr;
	a.len = static_cast<size_t>(v.size());
	return a;
}

//! Write one label to row k of a BIGINT / VARCHAR / BOOLEAN vector.
void SetLabel(Vector &vec, idx_t k, const GroupLabel &label) {
	switch (vec.GetType().id()) {
	case LogicalTypeId::VARCHAR:
		FlatDataMut<string_t>(vec)[k] = StringVector::AddString(vec, *label.str_value);
		break;
	case LogicalTypeId::BOOLEAN:
		FlatDataMut<bool>(vec)[k] = label.int_value != 0;
		break;
	default:
		FlatDataMut<int64_t>(vec)[k] = label.int_value;
		break;
	}
}

//! Data errors found in Update; recorded in the state (DuckDB relocates states with
//! a raw copy, so no message string), propagated by Combine, raised in Finalize.
enum class MetricsDataError : uint8_t { NONE, NON_BINARY_Y_TRUE };

//===--------------------------------------------------------------------===//
// classification_metrics_agg
//===--------------------------------------------------------------------===//
enum class LabelKind : uint8_t { INTEGER, VARCHAR, BOOLEAN };

LogicalType LabelType(LabelKind kind) {
	switch (kind) {
	case LabelKind::VARCHAR:
		return LogicalType::VARCHAR;
	case LabelKind::BOOLEAN:
		return LogicalType::BOOLEAN;
	default:
		return LogicalType::BIGINT;
	}
}

const char *AverageName(int32_t average) {
	switch (average) {
	case ANOFOX_AVERAGE_BINARY:
		return "binary";
	case ANOFOX_AVERAGE_MICRO:
		return "micro";
	case ANOFOX_AVERAGE_WEIGHTED:
		return "weighted";
	default:
		return "macro";
	}
}

LogicalType ClassificationResultType(LabelKind kind) {
	child_list_t<LogicalType> c;
	auto add = [&](const char *name, const LogicalType &type) {
		c.push_back(MakeChild(name, type));
	};
	add("n", LogicalType::BIGINT);
	for (auto name :
	     {"accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc", "cohen_kappa"}) {
		add(name, LogicalType::DOUBLE);
	}
	add("average", LogicalType::VARCHAR);
	add("labels", LogicalType::LIST(LabelType(kind)));
	add("confusion", LogicalType::LIST(LogicalType::LIST(LogicalType::BIGINT)));
	return LogicalType::STRUCT(std::move(c));
}

//! The labels of y_true and y_pred as dense codes, interleaved (true, pred) per row.
//! Relocatable: vectors and shared pointers only.
struct ClassificationState {
	GroupCoder labels;
	vector<double> codes;
	bool nan_score;
	MetricsDataError error;
	int64_t error_value;

	ClassificationState() : nan_score(false), error(MetricsDataError::NONE), error_value(0) {
	}
};

struct ClassificationBindData : public FunctionData {
	LabelKind kind = LabelKind::INTEGER;
	bool score = false; // y_pred is a DOUBLE score
	int32_t average = ANOFOX_AVERAGE_MACRO;
	bool has_pos_label = false;
	GroupLabel pos_label;
	double zero_division = 0.0;
	double threshold = 0.5;
	string options_text; // for Equals

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ClassificationBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ClassificationBindData>();
		return kind == o.kind && score == o.score && options_text == o.options_text;
	}
};

constexpr const char *CLASSIFICATION_NAME = "classification_metrics_agg";

void ClassificationInitialize(data_ptr_t state_p) {
	new (state_p) ClassificationState();
}

void ClassificationDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClassificationState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~ClassificationState();
	}
}

GroupLabel ReadLabel(const UnifiedVectorFormat &data, idx_t idx, LogicalTypeId type) {
	switch (type) {
	case LogicalTypeId::VARCHAR:
		return GroupLabel::FromString(UnifiedVectorFormat::GetData<string_t>(data)[idx].GetString());
	case LogicalTypeId::BOOLEAN:
		return GroupLabel::FromInt(UnifiedVectorFormat::GetData<bool>(data)[idx] ? 1 : 0);
	default:
		return GroupLabel::FromInt(UnifiedVectorFormat::GetData<int64_t>(data)[idx]);
	}
}

void ClassificationUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                          idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<ClassificationBindData>();
	UnifiedVectorFormat t_data, p_data, sdata;
	inputs[0].ToUnifiedFormat(count, t_data);
	inputs[1].ToUnifiedFormat(count, p_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClassificationState **)sdata.data;
	const auto t_type = inputs[0].GetType().id();
	const auto p_type = inputs[1].GetType().id();

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto t_idx = t_data.sel->get_index(i);
		auto p_idx = p_data.sel->get_index(i);
		if (!t_data.validity.RowIsValid(t_idx) || !p_data.validity.RowIsValid(p_idx)) {
			continue;
		}
		auto truth = ReadLabel(t_data, t_idx, t_type);
		GroupLabel pred;
		if (bind.score) {
			double s = UnifiedVectorFormat::GetData<double>(p_data)[p_idx];
			if (std::isnan(s)) {
				state.nan_score = true;
				continue;
			}
			if (truth.int_value != 0 && truth.int_value != 1) {
				if (state.error == MetricsDataError::NONE) {
					state.error = MetricsDataError::NON_BINARY_Y_TRUE;
					state.error_value = truth.int_value;
				}
				continue;
			}
			pred = GroupLabel::FromInt(s >= bind.threshold ? 1 : 0);
		} else {
			pred = ReadLabel(p_data, p_idx, p_type);
		}
		state.codes.push_back(state.labels.Code(truth));
		state.codes.push_back(state.labels.Code(pred));
	}
}

void ClassificationCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                           idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (ClassificationState **)source_data.data;
	auto targets = (ClassificationState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		target.nan_score = target.nan_score || source.nan_score;
		if (target.error == MetricsDataError::NONE) {
			target.error = source.error;
			target.error_value = source.error_value;
		}
		if (target.labels.labels.empty() && target.codes.empty()) {
			target.labels.Assign(source.labels.labels);
			target.codes = CombineTake(source.codes, aggr_input_data);
			continue;
		}
		auto map = target.labels.Remap(source.labels);
		for (auto c : source.codes) {
			target.codes.push_back(map[static_cast<idx_t>(c)]);
		}
	}
}

//! Write the k x k confusion counts (row-major) as LIST(LIST(BIGINT)) to row.
void SetConfusion(Vector &outer, idx_t row, const vector<uint64_t> &counts, idx_t k) {
	auto outer_offset = ListVector::GetListSize(outer);
	ListVector::Reserve(outer, outer_offset + k);
	auto &inner = ListVector::GetEntry(outer);
	auto inner_offset = ListVector::GetListSize(inner);
	ListVector::Reserve(inner, inner_offset + k * k);
	auto values = FlatDataMut<int64_t>(ListVector::GetEntry(inner));
	auto inner_entries = FlatDataMut<list_entry_t>(inner);
	for (idx_t r = 0; r < k; r++) {
		for (idx_t c = 0; c < k; c++) {
			values[inner_offset + r * k + c] = static_cast<int64_t>(counts[r * k + c]);
		}
		inner_entries[outer_offset + r].offset = inner_offset + r * k;
		inner_entries[outer_offset + r].length = k;
	}
	ListVector::SetListSize(inner, inner_offset + k * k);
	ListVector::SetListSize(outer, outer_offset + k);
	auto &entry = FlatDataMut<list_entry_t>(outer)[row];
	entry.offset = outer_offset;
	entry.length = k;
}

void ClassificationFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ClassificationBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClassificationState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.error == MetricsDataError::NON_BINARY_Y_TRUE) {
			throw InvalidInputException("%s: with a score as y_pred, y_true must be BOOLEAN or 0/1, got %lld",
			                            CLASSIFICATION_NAME, static_cast<long long>(state.error_value));
		}
		if (state.nan_score || state.codes.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Code k is the k-th smallest label: the FFI input and the label order of
		// the result do not depend on the row order.
		state.labels.Canonicalize(state.codes);
		const idx_t k = state.labels.labels.size();
		const idx_t n = state.codes.size() / 2;
		vector<int64_t> y_true(n), y_pred(n);
		for (idx_t r = 0; r < n; r++) {
			y_true[r] = static_cast<int64_t>(state.codes[2 * r]);
			y_pred[r] = static_cast<int64_t>(state.codes[2 * r + 1]);
		}
		int64_t pos_code = -1;
		if (bind.average == ANOFOX_AVERAGE_BINARY) {
			for (idx_t c = 0; c < k; c++) {
				if (state.labels.labels[c] == bind.pos_label) {
					pos_code = static_cast<int64_t>(c);
				}
			}
		}
		vector<uint64_t> counts(k * k);
		AnofoxClassificationMetricsResult r;
		AnofoxError error;
		if (!anofox_classification_metrics(y_true.data(), y_pred.data(), static_cast<size_t>(n), bind.average, pos_code,
		                                   bind.zero_division, static_cast<size_t>(k), counts.data(), &r, &error)) {
			ThrowUnlessDegenerate(CLASSIFICATION_NAME, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(entries[0]))[row] = static_cast<int64_t>(r.n);
		const double values[] = {r.accuracy,    r.precision,         r.recall, r.f1,
		                         r.specificity, r.balanced_accuracy, r.mcc,    r.cohen_kappa};
		for (idx_t v = 0; v < 8; v++) {
			SetDoubleOrNull(StructEntry(entries[1 + v]), row, values[v]);
		}
		auto &average = StructEntry(entries[9]);
		FlatDataMut<string_t>(average)[row] = StringVector::AddString(average, AverageName(bind.average));
		auto &labels = StructEntry(entries[10]);
		auto label_offset = ListVector::GetListSize(labels);
		ListVector::Reserve(labels, label_offset + k);
		auto &label_child = ListVector::GetEntry(labels);
		for (idx_t c = 0; c < k; c++) {
			SetLabel(label_child, label_offset + c, state.labels.labels[c]);
		}
		ListVector::SetListSize(labels, label_offset + k);
		auto &label_entry = FlatDataMut<list_entry_t>(labels)[row];
		label_entry.offset = label_offset;
		label_entry.length = k;
		SetConfusion(StructEntry(entries[11]), row, counts, k);
	}
}

void ParseClassificationOptions(const Value &options, ClassificationBindData &bind) {
	const char *fn = CLASSIFICATION_NAME;
	std::optional<Value> pos_label;
	bool has_threshold = false;
	VisitOptions(options, [&](const string &key, const Value &v) {
		if (key == "average") {
			auto name = StringUtil::Lower(OptionAs<string>(fn, key, v, LogicalType::VARCHAR));
			if (name == "binary") {
				bind.average = ANOFOX_AVERAGE_BINARY;
			} else if (name == "macro") {
				bind.average = ANOFOX_AVERAGE_MACRO;
			} else if (name == "micro") {
				bind.average = ANOFOX_AVERAGE_MICRO;
			} else if (name == "weighted") {
				bind.average = ANOFOX_AVERAGE_WEIGHTED;
			} else {
				throw InvalidInputException(
				    "%s: invalid average '%s'. Valid values are 'binary', 'macro', 'micro', 'weighted'", fn,
				    v.ToString());
			}
		} else if (key == "pos_label") {
			if (v.IsNull()) {
				throw InvalidInputException("%s: option 'pos_label' must not be NULL", fn);
			}
			pos_label = v;
		} else if (key == "zero_division") {
			bind.zero_division = OptionAs<double>(fn, key, v, LogicalType::DOUBLE);
		} else if (key == "threshold") {
			bind.threshold = OptionAs<double>(fn, key, v, LogicalType::DOUBLE);
			if (!std::isfinite(bind.threshold)) {
				throw InvalidInputException("%s: threshold must be finite, got %s", fn, v.ToString());
			}
			has_threshold = true;
		} else {
			throw InvalidInputException(
			    "%s: unknown option '%s'. Supported options: average, pos_label, zero_division, threshold", fn, key);
		}
	});
	if (bind.score && !has_threshold) {
		throw InvalidInputException("%s: y_pred is a DOUBLE score; set options.threshold to turn it into labels "
		                            "(predicted 1 / true when score >= threshold), or pass predicted labels",
		                            fn);
	}
	if (!bind.score && has_threshold) {
		throw InvalidInputException("%s: threshold applies only to a DOUBLE score as y_pred", fn);
	}
	if (pos_label && bind.average != ANOFOX_AVERAGE_BINARY) {
		throw InvalidInputException("%s: pos_label applies only to average = 'binary'", fn);
	}
	if (bind.average != ANOFOX_AVERAGE_BINARY) {
		return;
	}
	bind.has_pos_label = true;
	if (bind.kind == LabelKind::VARCHAR) {
		if (!pos_label) {
			throw InvalidInputException("%s: average = 'binary' with VARCHAR labels needs options.pos_label", fn);
		}
		bind.pos_label = GroupLabel::FromString(pos_label->ToString());
	} else if (bind.kind == LabelKind::BOOLEAN) {
		bind.pos_label = GroupLabel::FromInt(
		    pos_label ? (OptionAs<bool>(fn, "pos_label", *pos_label, LogicalType::BOOLEAN) ? 1 : 0) : 1);
	} else {
		bind.pos_label =
		    GroupLabel::FromInt(pos_label ? OptionAs<int64_t>(fn, "pos_label", *pos_label, LogicalType::BIGINT) : 1);
	}
}

template <LabelKind KIND, bool SCORE>
unique_ptr<FunctionData> ClassificationBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, ClassificationResultType(KIND));
	auto bind = make_uniq<ClassificationBindData>();
	bind->kind = KIND;
	bind->score = SCORE;
	Value options;
	if (arguments.size() >= 3) {
		options = EvaluateConstantOptions(context, *arguments[2], CLASSIFICATION_NAME);
		bind->options_text = options.ToString();
	}
	ParseClassificationOptions(options, *bind);
	PostHogTelemetry::Instance().RecordFunctionCall(CLASSIFICATION_NAME);
	return std::move(bind);
}

template <LabelKind KIND, bool SCORE>
void AddClassificationOverloads(AggregateFunctionSet &set) {
	auto label = LabelType(KIND);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {label, SCORE ? LogicalType::DOUBLE : label};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction f(
		    CLASSIFICATION_NAME, args, LogicalType::ANY, AggregateFunction::StateSize<ClassificationState>,
		    ANOFOX_AGG_INIT(ClassificationInitialize),
		    ANOFOX_GUARDED_UPDATE(ClassificationUpdate, ClassificationDestroy, ClassificationInitialize),
		    ClassificationCombine,
		    ANOFOX_GUARDED_FINALIZE(ClassificationFinalize, ClassificationDestroy, ClassificationInitialize), nullptr,
		    ANOFOX_AGG_BIND(ClassificationBind<KIND, SCORE>), ClassificationDestroy);
		set.AddFunction(f);
	}
}

//===--------------------------------------------------------------------===//
// regression_metrics_agg
//===--------------------------------------------------------------------===//
struct RegressionState {
	vector<double> y_true;
	vector<double> y_pred;
};

constexpr const char *REGRESSION_NAME = "regression_metrics_agg";

LogicalType RegressionResultType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("n", LogicalType::BIGINT));
	for (auto name : {"mae", "mse", "rmse", "mape", "smape", "median_ae", "r2", "explained_variance", "max_error"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	return LogicalType::STRUCT(std::move(c));
}

void RegressionInitialize(data_ptr_t state_p) {
	new (state_p) RegressionState();
}

void RegressionDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (RegressionState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~RegressionState();
	}
}

void RegressionUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat t_data, p_data, sdata;
	inputs[0].ToUnifiedFormat(count, t_data);
	inputs[1].ToUnifiedFormat(count, p_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto t = UnifiedVectorFormat::GetData<double>(t_data);
	auto p = UnifiedVectorFormat::GetData<double>(p_data);
	auto states = (RegressionState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto t_idx = t_data.sel->get_index(i);
		auto p_idx = p_data.sel->get_index(i);
		if (!t_data.validity.RowIsValid(t_idx) || !p_data.validity.RowIsValid(p_idx)) {
			continue;
		}
		auto &state = *states[sdata.sel->get_index(i)];
		state.y_true.push_back(t[t_idx]);
		state.y_pred.push_back(p[p_idx]);
	}
}

void RegressionCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (RegressionState **)source_data.data;
	auto targets = (RegressionState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (target.y_true.empty()) {
			target.y_true = CombineTake(source.y_true, aggr_input_data);
			target.y_pred = CombineTake(source.y_pred, aggr_input_data);
			continue;
		}
		target.y_true.insert(target.y_true.end(), source.y_true.begin(), source.y_true.end());
		target.y_pred.insert(target.y_pred.end(), source.y_pred.begin(), source.y_pred.end());
	}
}

void RegressionFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (RegressionState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.y_true.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical sums whatever the thread count (#226).
		CanonicalRows().Key(state.y_true).Key(state.y_pred).Sort();
		AnofoxRegressionMetricsResult r;
		AnofoxError error;
		if (!anofox_regression_metrics(AsArray(state.y_true), AsArray(state.y_pred), &r, &error)) {
			ThrowUnlessDegenerate(REGRESSION_NAME, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(entries[0]))[row] = static_cast<int64_t>(r.n);
		const double values[] = {r.mae,      r.mse, r.rmse, r.mape, r.smape, r.median_ae, r.r2, r.explained_variance,
		                         r.max_error};
		for (idx_t v = 0; v < 9; v++) {
			SetDoubleOrNull(StructEntry(entries[1 + v]), row, values[v]);
		}
	}
}

unique_ptr<FunctionData> RegressionBind(ClientContext &, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &) {
	SetReturnTypeOf(function, RegressionResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(REGRESSION_NAME);
	return nullptr;
}

//===--------------------------------------------------------------------===//
// ranking_metrics_agg
//===--------------------------------------------------------------------===//
struct RankingState {
	GroupCoder queries;
	vector<double> query_codes;
	vector<double> relevance;
	vector<double> score;
};

struct RankingBindData : public FunctionData {
	AnofoxRankingOptions options {std::nan(""), 10, false};
	string options_text;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RankingBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return options_text == other_p.Cast<RankingBindData>().options_text;
	}
};

constexpr const char *RANKING_NAME = "ranking_metrics_agg";

LogicalType RankingResultType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("n_queries", LogicalType::BIGINT));
	c.push_back(MakeChild("k", LogicalType::BIGINT));
	for (auto name : {"ndcg", "map", "mrr", "precision", "recall", "hit_rate"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	return LogicalType::STRUCT(std::move(c));
}

void RankingInitialize(data_ptr_t state_p) {
	new (state_p) RankingState();
}

void RankingDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (RankingState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~RankingState();
	}
}

void RankingUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat q_data, r_data, s_data, sdata;
	inputs[0].ToUnifiedFormat(count, q_data);
	inputs[1].ToUnifiedFormat(count, r_data);
	inputs[2].ToUnifiedFormat(count, s_data);
	state_vector.ToUnifiedFormat(count, sdata);
	const bool q_string = inputs[0].GetType().id() == LogicalTypeId::VARCHAR;
	auto rel = UnifiedVectorFormat::GetData<double>(r_data);
	auto sc = UnifiedVectorFormat::GetData<double>(s_data);
	auto states = (RankingState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto q_idx = q_data.sel->get_index(i);
		auto r_idx = r_data.sel->get_index(i);
		auto s_idx = s_data.sel->get_index(i);
		if (!q_data.validity.RowIsValid(q_idx) || !r_data.validity.RowIsValid(r_idx) ||
		    !s_data.validity.RowIsValid(s_idx)) {
			continue;
		}
		auto &state = *states[sdata.sel->get_index(i)];
		state.query_codes.push_back(state.queries.Code(ReadGroupLabel(q_data, q_idx, q_string)));
		state.relevance.push_back(rel[r_idx]);
		state.score.push_back(sc[s_idx]);
	}
}

void RankingCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (RankingState **)source_data.data;
	auto targets = (RankingState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (target.query_codes.empty() && target.queries.labels.empty()) {
			target.queries.Assign(source.queries.labels);
			target.query_codes = CombineTake(source.query_codes, aggr_input_data);
			target.relevance = CombineTake(source.relevance, aggr_input_data);
			target.score = CombineTake(source.score, aggr_input_data);
			continue;
		}
		auto map = target.queries.Remap(source.queries);
		for (auto c : source.query_codes) {
			target.query_codes.push_back(map[static_cast<idx_t>(c)]);
		}
		target.relevance.insert(target.relevance.end(), source.relevance.begin(), source.relevance.end());
		target.score.insert(target.score.end(), source.score.begin(), source.score.end());
	}
}

void RankingFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                     idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<RankingBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (RankingState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.query_codes.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order (queries by label, items by increasing score, then
		// relevance): upstream ranks tied scores in input order, so a tie ranks the
		// less relevant item first, whatever order the threads delivered the rows.
		state.queries.Canonicalize(state.query_codes);
		CanonicalRows().Key(state.query_codes).Key(state.score).Key(state.relevance).Sort();
		vector<int64_t> query(state.query_codes.size());
		for (idx_t r = 0; r < query.size(); r++) {
			query[r] = static_cast<int64_t>(state.query_codes[r]);
		}
		AnofoxRankingMetricsResult r;
		AnofoxError error;
		if (!anofox_ranking_metrics(query.data(), AsArray(state.relevance), AsArray(state.score), bind.options, &r,
		                            &error)) {
			ThrowUnlessDegenerate(RANKING_NAME, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(entries[0]))[row] = static_cast<int64_t>(r.n_queries);
		FlatDataMut<int64_t>(StructEntry(entries[1]))[row] = static_cast<int64_t>(bind.options.k);
		const double values[] = {r.ndcg, r.map, r.mrr, r.precision, r.recall, r.hit_rate};
		for (idx_t v = 0; v < 6; v++) {
			SetDoubleOrNull(StructEntry(entries[2 + v]), row, values[v]);
		}
	}
}

unique_ptr<FunctionData> RankingBind(ClientContext &context, AggregateBindFunction &function,
                                     vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, RankingResultType());
	auto bind = make_uniq<RankingBindData>();
	if (arguments.size() >= 4) {
		auto options = EvaluateConstantOptions(context, *arguments[3], RANKING_NAME);
		bind->options_text = options.ToString();
		VisitOptions(options, [&](const string &key, const Value &v) {
			if (key == "k") {
				auto k = OptionAs<int64_t>(RANKING_NAME, key, v, LogicalType::BIGINT);
				if (k < 1) {
					throw InvalidInputException("%s: k must be at least 1, got %lld", RANKING_NAME,
					                            static_cast<long long>(k));
				}
				bind->options.k = static_cast<size_t>(k);
			} else if (key == "min_relevance") {
				bind->options.min_relevance = OptionAs<double>(RANKING_NAME, key, v, LogicalType::DOUBLE);
				if (!std::isfinite(bind->options.min_relevance)) {
					throw InvalidInputException("%s: min_relevance must be finite, got %s", RANKING_NAME, v.ToString());
				}
			} else if (key == "include_no_feedback") {
				bind->options.include_no_feedback = OptionAs<bool>(RANKING_NAME, key, v, LogicalType::BOOLEAN);
			} else {
				throw InvalidInputException(
				    "%s: unknown option '%s'. Supported options: k, min_relevance, include_no_feedback", RANKING_NAME,
				    key);
			}
		});
	}
	PostHogTelemetry::Instance().RecordFunctionCall(RANKING_NAME);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// Table macros
//===--------------------------------------------------------------------===//
const char *const CLASSIFICATION_BY_SQL = R"(
SELECT group_col,
    (_anofox_m).n AS n,
    (_anofox_m).accuracy AS accuracy,
    (_anofox_m).precision AS precision,
    (_anofox_m).recall AS recall,
    (_anofox_m).f1 AS f1,
    (_anofox_m).specificity AS specificity,
    (_anofox_m).balanced_accuracy AS balanced_accuracy,
    (_anofox_m).mcc AS mcc,
    (_anofox_m).cohen_kappa AS cohen_kappa,
    (_anofox_m).average AS average,
    (_anofox_m).labels AS labels,
    (_anofox_m).confusion AS confusion
FROM (
    SELECT group_col, classification_metrics_agg(y_true_col, y_pred_col, options) AS _anofox_m
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
)
ORDER BY group_col
)";

const char *const REGRESSION_BY_SQL = R"(
SELECT group_col,
    (_anofox_m).n AS n,
    (_anofox_m).mae AS mae,
    (_anofox_m).mse AS mse,
    (_anofox_m).rmse AS rmse,
    (_anofox_m).mape AS mape,
    (_anofox_m).smape AS smape,
    (_anofox_m).median_ae AS median_ae,
    (_anofox_m).r2 AS r2,
    (_anofox_m).explained_variance AS explained_variance,
    (_anofox_m).max_error AS max_error
FROM (
    SELECT group_col, regression_metrics_agg(y_true_col::DOUBLE, y_pred_col::DOUBLE) AS _anofox_m
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
)
ORDER BY group_col
)";

const char *const RANKING_BY_SQL = R"(
SELECT group_col,
    (_anofox_m).n_queries AS n_queries,
    (_anofox_m).k AS k,
    (_anofox_m).ndcg AS ndcg,
    (_anofox_m).map AS map,
    (_anofox_m).mrr AS mrr,
    (_anofox_m).precision AS precision,
    (_anofox_m).recall AS recall,
    (_anofox_m).hit_rate AS hit_rate
FROM (
    SELECT group_col,
        ranking_metrics_agg(query_col, relevance_col::DOUBLE, score_col::DOUBLE, options) AS _anofox_m
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
)
ORDER BY group_col
)";

// data_quality_by (#369): per column of `columns`, the profile of source and, with
// a reference table, the comparison against it. Numeric columns (integer, float,
// decimal) are profiled as numbers, every other type as categories (its VARCHAR
// text). Without a reference the comparison columns are NULL.
const char *const DATA_QUALITY_BY_SQL = R"(
WITH _anofox_cur AS (
    SELECT COLUMNS(lambda _anofox_c: list_contains(columns, _anofox_c)) FROM query_table(source::VARCHAR)
),
_anofox_ref AS (
    SELECT COLUMNS(lambda _anofox_c: list_contains(columns, _anofox_c))
    FROM query_table(coalesce(reference, source)::VARCHAR)
),
_anofox_cur_t AS (
    SELECT {
        'type': typeof(COLUMNS(*)),
        'is_num': typeof(COLUMNS(*)) IN ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
                         'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'UHUGEINT', 'FLOAT', 'DOUBLE')
                  OR typeof(COLUMNS(*)) LIKE 'DECIMAL%',
        'num': CASE WHEN typeof(COLUMNS(*)) IN ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
                         'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'UHUGEINT', 'FLOAT', 'DOUBLE')
                    OR typeof(COLUMNS(*)) LIKE 'DECIMAL%'
               THEN TRY_CAST(COLUMNS(*) AS DOUBLE) END,
        'str': CASE WHEN typeof(COLUMNS(*)) NOT IN ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
                         'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'UHUGEINT', 'FLOAT', 'DOUBLE')
                    AND typeof(COLUMNS(*)) NOT LIKE 'DECIMAL%'
               THEN COLUMNS(*)::VARCHAR END
    }
    FROM _anofox_cur
),
_anofox_ref_t AS (
    SELECT {
        'num': CASE WHEN typeof(COLUMNS(*)) IN ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
                         'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'UHUGEINT', 'FLOAT', 'DOUBLE')
                    OR typeof(COLUMNS(*)) LIKE 'DECIMAL%'
               THEN TRY_CAST(COLUMNS(*) AS DOUBLE) END,
        'str': CASE WHEN typeof(COLUMNS(*)) NOT IN ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
                         'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'UHUGEINT', 'FLOAT', 'DOUBLE')
                    AND typeof(COLUMNS(*)) NOT LIKE 'DECIMAL%'
               THEN COLUMNS(*)::VARCHAR END
    }
    FROM _anofox_ref
),
_anofox_cur_l AS (
    UNPIVOT _anofox_cur_t ON COLUMNS(*) INTO NAME column_name VALUE _anofox_v
),
_anofox_ref_l AS (
    UNPIVOT _anofox_ref_t ON COLUMNS(*) INTO NAME column_name VALUE _anofox_v
),
_anofox_ref_s AS (
    SELECT column_name,
        min((_anofox_v).num) AS _anofox_min,
        max((_anofox_v).num) AS _anofox_max,
        list(DISTINCT (_anofox_v).str) FILTER (WHERE (_anofox_v).str IS NOT NULL) AS _anofox_cats
    FROM _anofox_ref_l
    GROUP BY column_name
),
_anofox_cur_s AS (
    SELECT l.column_name,
        any_value((_anofox_v).type) AS column_type,
        bool_or((_anofox_v).is_num) AS _anofox_numeric,
        count(*) AS n,
        count(*) FILTER (WHERE (_anofox_v).num IS NULL AND (_anofox_v).str IS NULL) AS n_missing,
        count(DISTINCT coalesce((_anofox_v).str, (_anofox_v).num::VARCHAR)) AS n_distinct,
        min((_anofox_v).num) AS min,
        max((_anofox_v).num) AS max,
        avg((_anofox_v).num) AS mean,
        stddev_samp((_anofox_v).num) AS sd,
        list(DISTINCT (_anofox_v).str) FILTER (WHERE (_anofox_v).str IS NOT NULL) AS _anofox_cats,
        avg(CASE WHEN (_anofox_v).num IS NULL THEN NULL
                 WHEN (_anofox_v).num < r._anofox_min OR (_anofox_v).num > r._anofox_max THEN 1.0
                 ELSE 0.0 END) AS _anofox_oor,
        any_value(r._anofox_min) AS reference_min,
        any_value(r._anofox_max) AS reference_max,
        any_value(r._anofox_cats) AS _anofox_ref_cats
    FROM _anofox_cur_l l LEFT JOIN _anofox_ref_s r ON l.column_name = r.column_name
    GROUP BY l.column_name
)
SELECT column_name,
    column_type,
    n,
    n_missing,
    n_missing / n AS missing_share,
    n_distinct,
    min,
    max,
    mean,
    sd,
    CASE WHEN reference IS NOT NULL THEN reference_min END AS reference_min,
    CASE WHEN reference IS NOT NULL THEN reference_max END AS reference_max,
    CASE WHEN reference IS NOT NULL AND _anofox_numeric THEN _anofox_oor END AS out_of_range_share,
    CASE WHEN reference IS NOT NULL AND NOT _anofox_numeric
         THEN list_sort(list_filter(coalesce(_anofox_cats, []),
                                    lambda _anofox_x: NOT list_contains(coalesce(_anofox_ref_cats, []), _anofox_x)))
    END AS new_categories,
    CASE WHEN reference IS NOT NULL AND NOT _anofox_numeric
         THEN list_sort(list_filter(coalesce(_anofox_ref_cats, []),
                                    lambda _anofox_x: NOT list_contains(coalesce(_anofox_cats, []), _anofox_x)))
    END AS dropped_categories
FROM _anofox_cur_s
ORDER BY list_position(columns, column_name)
)";

} // namespace

void RegisterMetricsFunctions(ExtensionLoader &loader) {
	{
		AggregateFunctionSet set(CLASSIFICATION_NAME);
		AddClassificationOverloads<LabelKind::INTEGER, false>(set);
		AddClassificationOverloads<LabelKind::VARCHAR, false>(set);
		AddClassificationOverloads<LabelKind::BOOLEAN, false>(set);
		AddClassificationOverloads<LabelKind::INTEGER, true>(set);
		AddClassificationOverloads<LabelKind::BOOLEAN, true>(set);
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description =
		    "Classification metrics of predicted against true labels (scikit-learn sklearn.metrics, R yardstick): "
		    "n, accuracy, precision, recall, f1, specificity (per class, combined by options.average: 'macro' "
		    "(default), 'micro', 'weighted' or 'binary' with pos_label, default 1 / true), balanced_accuracy, mcc, "
		    "cohen_kappa (NULL when undefined), the average used, the sorted labels and the confusion matrix "
		    "(rows = true, columns = predicted label). Labels are BIGINT, VARCHAR or BOOLEAN; a DOUBLE score as "
		    "y_pred needs options.threshold (predicted 1 / true when score >= threshold). options.zero_division "
		    "(default 0) is the value of a ratio with a zero denominator. NULL rows are skipped.";
		d.examples = {"classification_metrics_agg(label, predicted, {'average': 'weighted'})",
		              "classification_metrics_agg(label, score, {'threshold': 0.5, 'average': 'binary'})"};
		d.categories = {"classification", "metrics"};
		d.parameter_names = {"y_true", "y_pred", "options"};
		d.parameter_types = {LogicalType::ANY, LogicalType::ANY, LogicalType::ANY};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set(REGRESSION_NAME);
		AggregateFunction f(REGRESSION_NAME, {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::ANY,
		                    AggregateFunction::StateSize<RegressionState>, ANOFOX_AGG_INIT(RegressionInitialize),
		                    ANOFOX_GUARDED_UPDATE(RegressionUpdate, RegressionDestroy, RegressionInitialize),
		                    RegressionCombine,
		                    ANOFOX_GUARDED_FINALIZE(RegressionFinalize, RegressionDestroy, RegressionInitialize),
		                    nullptr, ANOFOX_AGG_BIND(RegressionBind), RegressionDestroy);
		set.AddFunction(f);
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description =
		    "Regression metrics of y_pred against y_true (scikit-learn sklearn.metrics): n, mae, mse, rmse, mape "
		    "and smape (fractions, not percent; mape divides by max(|y|, machine epsilon)), median_ae, r2, "
		    "explained_variance (NULL for a single row; 1 or 0 for a constant y_true), max_error. NULL rows are "
		    "skipped; a NaN or infinite value gives NULL.";
		d.examples = {"regression_metrics_agg(actual, forecast)"};
		d.categories = {"regression", "metrics"};
		d.parameter_names = {"y_true", "y_pred"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set(RANKING_NAME);
		for (auto &query_type : {LogicalType::BIGINT, LogicalType::VARCHAR}) {
			for (bool with_options : {false, true}) {
				vector<LogicalType> args = {query_type, LogicalType::DOUBLE, LogicalType::DOUBLE};
				if (with_options) {
					args.push_back(LogicalType::ANY);
				}
				AggregateFunction f(RANKING_NAME, args, LogicalType::ANY, AggregateFunction::StateSize<RankingState>,
				                    ANOFOX_AGG_INIT(RankingInitialize),
				                    ANOFOX_GUARDED_UPDATE(RankingUpdate, RankingDestroy, RankingInitialize),
				                    RankingCombine,
				                    ANOFOX_GUARDED_FINALIZE(RankingFinalize, RankingDestroy, RankingInitialize),
				                    nullptr, ANOFOX_AGG_BIND(RankingBind), RankingDestroy);
				set.AddFunction(f);
			}
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description =
		    "Ranking (recommender) metrics at k (options.k, default 10), the items of each query ranked by "
		    "decreasing score and averaged over the queries: ndcg (scikit-learn ndcg_score, graded relevance as "
		    "gain, tied scores averaged), map, mrr, precision, recall and hit_rate (evidently). An item is relevant "
		    "when relevance > 0, or >= options.min_relevance (then NDCG uses binary gains too). Queries without a "
		    "relevant item are left out unless options.include_no_feedback is true. Tied scores rank the less "
		    "relevant item first. NULL rows are skipped.";
		d.examples = {"ranking_metrics_agg(user_id, relevance, score, {'k': 5})"};
		d.categories = {"ranking", "metrics"};
		d.parameter_names = {"query_id", "relevance", "score", "options"};
		d.parameter_types = {LogicalType::ANY, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}

	auto classification_by = CreateContractTableMacro(
	    "classification_metrics_by", {"source", "group_col", "y_true_col", "y_pred_col"}, {{"options", "NULL"}},
	    CLASSIFICATION_BY_SQL,
	    "Table macro: classification_metrics_agg(y_true_col, y_pred_col, options) per group of source, one row "
	    "per group: the group column, n, accuracy, precision, recall, f1, specificity, balanced_accuracy, mcc, "
	    "cohen_kappa, average, labels, confusion.",
	    "classification_metrics_by('scored', model, label, predicted, options := {'average': 'macro'})",
	    {"classification", "metrics", "table-macro"});
	loader.RegisterFunction(*classification_by);
	auto regression_by = CreateContractTableMacro(
	    "regression_metrics_by", {"source", "group_col", "y_true_col", "y_pred_col"}, {}, REGRESSION_BY_SQL,
	    "Table macro: regression_metrics_agg(y_true_col, y_pred_col) per group of source, one row per group: the "
	    "group column, n, mae, mse, rmse, mape, smape, median_ae, r2, explained_variance, max_error.",
	    "regression_metrics_by('forecasts', series_id, actual, forecast)", {"regression", "metrics", "table-macro"});
	loader.RegisterFunction(*regression_by);
	auto ranking_by = CreateContractTableMacro(
	    "ranking_metrics_by", {"source", "group_col", "query_col", "relevance_col", "score_col"}, {{"options", "NULL"}},
	    RANKING_BY_SQL,
	    "Table macro: ranking_metrics_agg(query_col, relevance_col, score_col, options) per group of source, one "
	    "row per group: the group column, n_queries, k, ndcg, map, mrr, precision, recall, hit_rate.",
	    "ranking_metrics_by('recommendations', model, user_id, relevance, score, options := {'k': 5})",
	    {"ranking", "metrics", "table-macro"});
	loader.RegisterFunction(*ranking_by);
	auto data_quality_by = CreateContractTableMacro(
	    "data_quality_by", {"source", "columns"}, {{"reference", "NULL"}}, DATA_QUALITY_BY_SQL,
	    "Table macro: a data-quality summary of the columns of source named in columns, one row per column in "
	    "that order: column_name, column_type, n, n_missing, missing_share, n_distinct, and for numeric columns "
	    "min, max, mean, sd. With reference := '<table>' also reference_min / reference_max, out_of_range_share "
	    "(numeric: share of the non-missing values outside the reference range) and new_categories / "
	    "dropped_categories (other types: sorted categories seen only in source / only in the reference).",
	    "data_quality_by('events_today', ['age', 'city'], reference := 'events_last_month')",
	    {"drift", "data-quality", "table-macro"});
	loader.RegisterFunction(*data_quality_by);
}

} // namespace duckdb
