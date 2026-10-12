#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "telemetry.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Classifier evaluation (#153).
//
//   roc_agg(y_true, score [, drop_intermediate])    -> LIST(STRUCT(threshold, fpr, tpr))
//   pr_agg(y_true, score)                           -> LIST(STRUCT(threshold, recall, precision))
//   calibration_agg(y_true, prob, n_bins [, conf_level])
//                     -> LIST(STRUCT(bin, mean_pred, frac_pos, n, conf_low, conf_high))
//   auc_agg(y_true, score), brier_agg(y_true, prob), log_loss_agg(y_true, prob) -> DOUBLE
//
// y_true is BOOLEAN or a numeric 0/1 (any other value is an error). Rows with a
// NULL y_true or score are skipped; no rows left gives NULL. The curves follow
// scikit-learn (tied scores form one point; constant scores give the two points
// (0, 0) and (1, 1)). A rate that is undefined because one class is absent is
// NULL, and auc_agg is NULL then. A NaN score gives NULL, a probability outside
// [0, 1] is an error.
//===--------------------------------------------------------------------===//

enum class ClassifierKind : uint8_t { ROC, PR, CALIBRATION, AUC, BRIER, LOG_LOSS };

struct ClassifierState {
	vector<double> scores;
	vector<double> labels;

	void Reset() {
		scores.clear();
		labels.clear();
	}
};

struct ClassifierBindData : public FunctionData {
	ClassifierKind kind;
	string name;
	bool drop_intermediate = true;
	idx_t n_bins = 10;
	double conf_level = 0.95;

	ClassifierBindData(ClassifierKind kind_p, string name_p) : kind(kind_p), name(std::move(name_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ClassifierBindData>(kind, name);
		result->drop_intermediate = drop_intermediate;
		result->n_bins = n_bins;
		result->conf_level = conf_level;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ClassifierBindData>();
		return kind == o.kind && drop_intermediate == o.drop_intermediate && n_bins == o.n_bins &&
		       conf_level == o.conf_level;
	}
};

static LogicalType CurveType(ClassifierKind kind) {
	child_list_t<LogicalType> c;
	switch (kind) {
	case ClassifierKind::ROC:
		c.push_back(make_pair("threshold", LogicalType::DOUBLE));
		c.push_back(make_pair("fpr", LogicalType::DOUBLE));
		c.push_back(make_pair("tpr", LogicalType::DOUBLE));
		break;
	case ClassifierKind::PR:
		c.push_back(make_pair("threshold", LogicalType::DOUBLE));
		c.push_back(make_pair("recall", LogicalType::DOUBLE));
		c.push_back(make_pair("precision", LogicalType::DOUBLE));
		break;
	case ClassifierKind::CALIBRATION:
		c.push_back(make_pair("bin", LogicalType::BIGINT));
		c.push_back(make_pair("mean_pred", LogicalType::DOUBLE));
		c.push_back(make_pair("frac_pos", LogicalType::DOUBLE));
		c.push_back(make_pair("n", LogicalType::BIGINT));
		c.push_back(make_pair("conf_low", LogicalType::DOUBLE));
		c.push_back(make_pair("conf_high", LogicalType::DOUBLE));
		break;
	default:
		return LogicalType::DOUBLE;
	}
	return LogicalType::LIST(LogicalType::STRUCT(std::move(c)));
}

static void ClassifierInitialize(const AggregateFunction &, data_ptr_t state_p) {
	new (state_p) ClassifierState();
}

static void ClassifierDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClassifierState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~ClassifierState();
	}
}

static void ClassifierUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                             idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<ClassifierBindData>();
	const bool bool_labels = inputs[0].GetType().id() == LogicalTypeId::BOOLEAN;

	UnifiedVectorFormat y_data, s_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, s_data);
	auto s_values = UnifiedVectorFormat::GetData<double>(s_data);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClassifierState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto y_idx = y_data.sel->get_index(i);
		auto s_idx = s_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !s_data.validity.RowIsValid(s_idx)) {
			continue;
		}
		double label;
		if (bool_labels) {
			label = UnifiedVectorFormat::GetData<bool>(y_data)[y_idx] ? 1.0 : 0.0;
		} else {
			label = UnifiedVectorFormat::GetData<double>(y_data)[y_idx];
			if (label != 0.0 && label != 1.0) {
				throw InvalidInputException("%s: y_true must be BOOLEAN or 0/1, got %s", bind_data.name,
				                            std::to_string(label));
			}
		}
		auto &state = *states[sdata.sel->get_index(i)];
		state.labels.push_back(label);
		state.scores.push_back(s_values[s_idx]);
	}
}

static void ClassifierCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (ClassifierState **)source_data.data;
	auto targets = (ClassifierState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		target.scores.insert(target.scores.end(), source.scores.begin(), source.scores.end());
		target.labels.insert(target.labels.end(), source.labels.begin(), source.labels.end());
	}
}

//! Write `v` to a DOUBLE child, NULL when NaN (undefined rate, missing threshold).
static inline void SetDoubleOrNull(Vector &vec, idx_t idx, double v) {
	if (std::isnan(v)) {
		FlatVector::SetNull(vec, idx, true);
	} else {
		FlatVector::GetData<double>(vec)[idx] = v;
	}
}

static void WriteCurve(ClassifierKind kind, const AnofoxClassifierCurve &curve, Vector &result, idx_t row) {
	auto list_data = FlatVector::GetData<list_entry_t>(result);
	auto child_offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, child_offset + curve.len);
	auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));

	for (idx_t j = 0; j < curve.len; j++) {
		const idx_t k = child_offset + j;
		if (kind == ClassifierKind::CALIBRATION) {
			FlatVector::GetData<int64_t>(*entries[0])[k] = curve.bin[j];
			SetDoubleOrNull(*entries[1], k, curve.x[j]);
			SetDoubleOrNull(*entries[2], k, curve.y[j]);
			FlatVector::GetData<int64_t>(*entries[3])[k] = curve.n[j];
			SetDoubleOrNull(*entries[4], k, curve.y_low[j]);
			SetDoubleOrNull(*entries[5], k, curve.y_high[j]);
		} else {
			SetDoubleOrNull(*entries[0], k, curve.threshold[j]);
			SetDoubleOrNull(*entries[1], k, curve.x[j]);
			SetDoubleOrNull(*entries[2], k, curve.y[j]);
		}
	}
	ListVector::SetListSize(result, child_offset + curve.len);
	list_data[row].offset = child_offset;
	list_data[row].length = curve.len;
}

static void ClassifierFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                               idx_t count, idx_t offset) {
	auto &bind_data = aggr_input_data.bind_data->Cast<ClassifierBindData>();
	const string &name = bind_data.name;
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClassifierState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (state.scores.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		AnofoxDataArray scores {state.scores.data(), nullptr, static_cast<size_t>(state.scores.size())};
		AnofoxDataArray labels {state.labels.data(), nullptr, static_cast<size_t>(state.labels.size())};
		AnofoxError error;
		bool ok;

		if (bind_data.kind == ClassifierKind::AUC || bind_data.kind == ClassifierKind::BRIER ||
		    bind_data.kind == ClassifierKind::LOG_LOSS) {
			auto metric = bind_data.kind == ClassifierKind::AUC     ? ANOFOX_CLASSIFIER_METRIC_AUC
			              : bind_data.kind == ClassifierKind::BRIER ? ANOFOX_CLASSIFIER_METRIC_BRIER
			                                                        : ANOFOX_CLASSIFIER_METRIC_LOG_LOSS;
			double value = 0.0;
			ok = anofox_classifier_metric(metric, scores, labels, &value, &error);
			if (ok) {
				FlatVector::GetData<double>(result)[row] = value;
			}
		} else {
			AnofoxClassifierCurve curve {};
			switch (bind_data.kind) {
			case ClassifierKind::ROC:
				ok = anofox_roc_curve(scores, labels, bind_data.drop_intermediate, &curve, &error);
				break;
			case ClassifierKind::PR:
				ok = anofox_pr_curve(scores, labels, &curve, &error);
				break;
			default:
				ok = anofox_calibration_curve(scores, labels, static_cast<size_t>(bind_data.n_bins),
				                              bind_data.conf_level, &curve, &error);
				break;
			}
			if (ok) {
				WriteCurve(bind_data.kind, curve, result, row);
				anofox_free_classifier_curve(&curve);
			}
		}
		state.Reset();
		if (!ok) {
			ThrowUnlessDegenerate(name.c_str(), error);
			FlatVector::SetNull(result, row, true);
		}
	}
}

//! Evaluate a constant scalar argument; it must not vary per row.
static Value ConstantArgument(ClientContext &context, Expression &expr, const string &fn, const char *param) {
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: %s must be a constant", fn, string(param));
	}
	auto value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (value.IsNull()) {
		throw InvalidInputException("%s: %s must not be NULL", fn, string(param));
	}
	return value;
}

template <ClassifierKind KIND>
static unique_ptr<FunctionData> ClassifierBind(ClientContext &context, AggregateFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<ClassifierBindData>(KIND, function.name);
	const string &fn = result->name;
	if (KIND == ClassifierKind::ROC && arguments.size() >= 3) {
		result->drop_intermediate =
		    ConstantArgument(context, *arguments[2], fn, "drop_intermediate").GetValue<bool>();
	}
	if (KIND == ClassifierKind::CALIBRATION) {
		auto n_bins = ConstantArgument(context, *arguments[2], fn, "n_bins").GetValue<int64_t>();
		if (n_bins < 1) {
			throw InvalidInputException("%s: n_bins must be >= 1, got %lld", fn, (long long)n_bins);
		}
		result->n_bins = static_cast<idx_t>(n_bins);
		if (arguments.size() >= 4) {
			auto conf_level = ConstantArgument(context, *arguments[3], fn, "conf_level").GetValue<double>();
			if (!(conf_level > 0.0 && conf_level < 1.0)) {
				throw InvalidInputException("%s: conf_level must be in (0, 1), got %s", fn,
				                            std::to_string(conf_level));
			}
			result->conf_level = conf_level;
		}
	}
	function.return_type = CurveType(KIND);
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(result);
}

struct ClassifierSpec {
	const char *name;
	ClassifierKind kind;
	bind_aggregate_function_t bind;
	//! Name of the second argument (score or prob).
	const char *value_name;
	//! Extra constant parameters, each optional overload adding one more.
	vector<pair<string, LogicalType>> required_extra;
	vector<pair<string, LogicalType>> optional_extra;
	const char *description;
	const char *example;
};

static void RegisterClassifierAggregate(ExtensionLoader &loader, const ClassifierSpec &spec) {
	AggregateFunctionSet func_set(spec.name);
	vector<FunctionDescription> descriptions;
	const vector<LogicalType> y_types = {LogicalType::BOOLEAN, LogicalType::DOUBLE};

	// Overloads: y_true BOOLEAN or DOUBLE (numerics cast), with and without each optional parameter.
	for (auto &y_type : y_types) {
		for (idx_t n_optional = 0; n_optional <= spec.optional_extra.size(); n_optional++) {
			vector<LogicalType> args = {y_type, LogicalType::DOUBLE};
			vector<string> names = {"y_true", spec.value_name};
			for (auto &p : spec.required_extra) {
				args.push_back(p.second);
				names.push_back(p.first);
			}
			for (idx_t k = 0; k < n_optional; k++) {
				args.push_back(spec.optional_extra[k].second);
				names.push_back(spec.optional_extra[k].first);
			}
			AggregateFunction fn(spec.name, args, LogicalType::ANY, AggregateFunction::StateSize<ClassifierState>,
			                     ClassifierInitialize,
			                     ANOFOX_GUARDED_UPDATE(ClassifierUpdate, ClassifierDestroy, ClassifierInitialize),
			                     ClassifierCombine,
			                     ANOFOX_GUARDED_FINALIZE(ClassifierFinalize, ClassifierDestroy, ClassifierInitialize),
			                     nullptr, spec.bind, ClassifierDestroy);
			func_set.AddFunction(fn);

			FunctionDescription d;
			d.description = spec.description;
			d.examples = {spec.example};
			d.categories = {"classification"};
			d.parameter_names = names;
			d.parameter_types = args;
			descriptions.push_back(std::move(d));
		}
	}
	CreateAggregateFunctionInfo info(func_set);
	info.descriptions = std::move(descriptions);
	loader.RegisterFunction(info);
}

void RegisterClassifierAggregateFunctions(ExtensionLoader &loader) {
	const ClassifierSpec specs[] = {
	    {"roc_agg", ClassifierKind::ROC, ClassifierBind<ClassifierKind::ROC>, "score", {},
	     {{"drop_intermediate", LogicalType::BOOLEAN}},
	     "ROC curve of a binary classifier as a LIST of {threshold, fpr, tpr}, as scikit-learn roc_curve: the "
	     "first point is (0, 0) at threshold +inf, tied scores form one point, and drop_intermediate (default "
	     "true) drops collinear points. y_true is BOOLEAN or 0/1.",
	     "roc_agg(y_true, score)"},
	    {"pr_agg", ClassifierKind::PR, ClassifierBind<ClassifierKind::PR>, "score", {}, {},
	     "Precision-recall curve as a LIST of {threshold, recall, precision} in increasing threshold order, as "
	     "scikit-learn precision_recall_curve; the final point (recall 0, precision 1) has a NULL threshold.",
	     "pr_agg(y_true, score)"},
	    {"calibration_agg", ClassifierKind::CALIBRATION, ClassifierBind<ClassifierKind::CALIBRATION>, "prob",
	     {{"n_bins", LogicalType::BIGINT}}, {{"conf_level", LogicalType::DOUBLE}},
	     "Calibration (reliability) curve over n_bins equal-width bins of [0, 1] as a LIST of {bin, mean_pred, "
	     "frac_pos, n, conf_low, conf_high} (scikit-learn calibration_curve, strategy 'uniform'); empty bins are "
	     "omitted, bin is zero-based, and conf_low/conf_high are Wilson intervals (default conf_level 0.95).",
	     "calibration_agg(y_true, prob, 10)"},
	    {"auc_agg", ClassifierKind::AUC, ClassifierBind<ClassifierKind::AUC>, "score", {}, {},
	     "Area under the ROC curve (scikit-learn roc_auc_score, pROC auc); ties count one half. NULL when only "
	     "one class is present.",
	     "auc_agg(y_true, score)"},
	    {"brier_agg", ClassifierKind::BRIER, ClassifierBind<ClassifierKind::BRIER>, "prob", {}, {},
	     "Brier score: mean squared difference between the predicted probability and the 0/1 outcome "
	     "(scikit-learn brier_score_loss).",
	     "brier_agg(y_true, prob)"},
	    {"log_loss_agg", ClassifierKind::LOG_LOSS, ClassifierBind<ClassifierKind::LOG_LOSS>, "prob", {}, {},
	     "Log loss (mean binary cross-entropy, scikit-learn log_loss); probabilities are clipped to "
	     "[eps, 1 - eps].",
	     "log_loss_agg(y_true, prob)"},
	};
	for (auto &spec : specs) {
		RegisterClassifierAggregate(loader, spec);
	}
}

} // namespace duckdb
