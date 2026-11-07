// Confidence-based performance estimation of a binary classifier (#365), as
// NannyML's CBPE(problem_type = "classification_binary"):
//
//   cbpe_fit_agg(y_true, score [, options])                 -> model STRUCT
//   cbpe_fit_agg(y_true, score, prediction, options)        -> model STRUCT
//   cbpe_estimate_agg(model, score [, prediction])          -> STRUCT of estimated metrics
//   cbpe_realized_agg(model, y_true, score [, prediction])  -> STRUCT of realized metrics
//   cbpe_by(source, chunk_col, y_col, score_col, is_reference, options := NULL, prediction := NULL)
//
// The reference period (labelled) calibrates the scores (isotonic, Platt or none)
// and stores the sampling-error components; the model STRUCT holds these parts
// and is rebuilt upstream with CbpeModel::from_parts to estimate the confusion
// matrix, accuracy, precision, recall, F1, specificity and ROC AUC of unlabelled
// chunks with NannyML's sampling-error bands. All statistics come from upstream
// anofox-statistics monitoring::cbpe; this file buffers rows, reads and writes the
// STRUCTs.
//
// y_true is BOOLEAN or a numeric 0/1. A row is predicted positive when score >=
// threshold, unless a prediction is given (a NULL prediction falls back to the
// threshold). Rows with a NULL score (or a NULL y_true where labels are read) are
// skipped; no rows give NULL. The model is taken from the first row of a group
// where it is not NULL; no model gives NULL. A reference with one class, or
// non-finite scores, give NULL.

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"

#include "../include/anofox_cbpe_ffi.h"
#include "../include/anofox_stats_ffi.h"
#include "../include/canonical_order.hpp"
#include "../include/duckdb_compat.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "aggregate_finalize_guard.hpp"
#include "anofox_statistics_extension.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const char *const kFit = "cbpe_fit_agg";
const char *const kEstimate = "cbpe_estimate_agg";
const char *const kRealized = "cbpe_realized_agg";

const char *const CALIBRATION_NAMES[] = {"isotonic", "platt", "none"};

//! Order of AnofoxCbpeModel::sampling.
const char *const SAMPLING_NAMES[ANOFOX_CBPE_SAMPLING_FIELDS] = {
    "accuracy_std",       "precision_std",      "precision_fraction",   "recall_std",
    "recall_fraction",    "specificity_std",    "specificity_fraction", "f1_std",
    "f1_fraction",        "roc_auc_std",        "roc_auc_fraction",     "true_positive_std",
    "false_positive_std", "false_negative_std", "true_negative_std"};

//! Order of AnofoxCbpeEstimate::metrics / AnofoxCbpeRealized::metrics; the last four are counts.
const char *const METRIC_NAMES[ANOFOX_CBPE_METRICS] = {
    "roc_auc",        "f1",           "precision",     "recall",
    "specificity",    "accuracy",     "true_positive", "false_positive",
    "false_negative", "true_negative"};
constexpr idx_t kFirstCount = 6;

//===--------------------------------------------------------------------===//
// Types
//===--------------------------------------------------------------------===//

// Model STRUCT field positions.
enum ModelField : idx_t {
	MF_CALIBRATION = 0,
	MF_THRESHOLD,
	MF_ISOTONIC_X,
	MF_ISOTONIC_Y,
	MF_PLATT_A,
	MF_PLATT_B,
	MF_SAMPLING,
	MF_N_REFERENCE,
	MF_MODEL_TYPE
};

LogicalType SamplingType() {
	child_list_t<LogicalType> c;
	for (auto name : SAMPLING_NAMES) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	return LogicalType::STRUCT(std::move(c));
}

LogicalType ModelType() {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("calibration", LogicalType::VARCHAR));
	c.push_back(MakeChild("threshold", LogicalType::DOUBLE));
	c.push_back(MakeChild("isotonic_x", dlist));
	c.push_back(MakeChild("isotonic_y", dlist));
	c.push_back(MakeChild("platt_a", LogicalType::DOUBLE));
	c.push_back(MakeChild("platt_b", LogicalType::DOUBLE));
	c.push_back(MakeChild("sampling", SamplingType()));
	c.push_back(MakeChild("n_reference", LogicalType::BIGINT));
	c.push_back(MakeChild("model_type", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType EstimateType() {
	child_list_t<LogicalType> m;
	m.push_back(MakeChild("value", LogicalType::DOUBLE));
	m.push_back(MakeChild("sampling_error", LogicalType::DOUBLE));
	m.push_back(MakeChild("lower_confidence", LogicalType::DOUBLE));
	m.push_back(MakeChild("upper_confidence", LogicalType::DOUBLE));
	auto metric = LogicalType::STRUCT(std::move(m));
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("n", LogicalType::BIGINT));
	for (auto name : METRIC_NAMES) {
		c.push_back(MakeChild(name, metric));
	}
	return LogicalType::STRUCT(std::move(c));
}

LogicalType RealizedType() {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("n", LogicalType::BIGINT));
	for (idx_t k = 0; k < ANOFOX_CBPE_METRICS; k++) {
		c.push_back(MakeChild(METRIC_NAMES[k], k < kFirstCount ? LogicalType::DOUBLE : LogicalType::BIGINT));
	}
	return LogicalType::STRUCT(std::move(c));
}

//===--------------------------------------------------------------------===//
// State
//===--------------------------------------------------------------------===//

//! Data errors found in Update, raised in Finalize (parallel states must not leak).
enum CbpeError : int32_t { ERR_NONE = 0, ERR_LABEL, ERR_MODEL };

struct CbpeState {
	vector<double> scores;
	vector<double> labels;      // 0 / 1
	vector<double> predictions; // 0 / 1, -1 missing
	// Stored model parts (cbpe_estimate_agg / cbpe_realized_agg).
	vector<double> knots_x;
	vector<double> knots_y;
	double sampling[ANOFOX_CBPE_SAMPLING_FIELDS];
	double threshold;
	double platt_a;
	double platt_b;
	double bad_label;
	int32_t calibration;
	int32_t error;
	bool has_model;

	void Reset() {
		scores.clear();
		labels.clear();
		predictions.clear();
		knots_x.clear();
		knots_y.clear();
		has_model = false;
		error = ERR_NONE;
	}
};

struct CbpeStates {
	CbpeStates(Vector &state_vector, idx_t count) {
		state_vector.ToUnifiedFormat(count, sdata);
		states = (CbpeState **)sdata.data;
	}
	CbpeState &operator[](idx_t i) {
		return *states[sdata.sel->get_index(i)];
	}
	UnifiedVectorFormat sdata;
	CbpeState **states;
};

void CbpeInitialize(data_ptr_t state_p) {
	auto state = new (state_p) CbpeState();
	state->threshold = kNaN;
	state->platt_a = kNaN;
	state->platt_b = kNaN;
	state->bad_label = kNaN;
	state->calibration = 0;
	state->error = ERR_NONE;
	state->has_model = false;
	for (auto &s : state->sampling) {
		s = kNaN;
	}
}

void CbpeDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	CbpeStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		states[i].~CbpeState();
	}
}

void CbpeCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	CbpeStates sources(source_vector, count);
	CbpeStates targets(target_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &source = sources[i];
		auto &target = targets[i];
		target.scores.insert(target.scores.end(), source.scores.begin(), source.scores.end());
		target.labels.insert(target.labels.end(), source.labels.begin(), source.labels.end());
		target.predictions.insert(target.predictions.end(), source.predictions.begin(), source.predictions.end());
		if (target.error == ERR_NONE && source.error != ERR_NONE) {
			target.error = source.error;
			target.bad_label = source.bad_label;
		}
		if (!target.has_model && source.has_model) {
			target.knots_x = source.knots_x;
			target.knots_y = source.knots_y;
			for (idx_t k = 0; k < ANOFOX_CBPE_SAMPLING_FIELDS; k++) {
				target.sampling[k] = source.sampling[k];
			}
			target.threshold = source.threshold;
			target.platt_a = source.platt_a;
			target.platt_b = source.platt_b;
			target.calibration = source.calibration;
			target.has_model = true;
		}
	}
}

//! A DOUBLE STRUCT child, NaN when NULL.
double DoubleOrNaN(const Value &v) {
	return v.IsNull() ? kNaN : v.GetValue<double>();
}

//! Reads the parts of a model STRUCT (cast to ModelType() at bind) into the state.
//! false when a required part is missing or the calibration is unknown.
bool ReadModel(const Value &model, CbpeState &state) {
	auto &f = StructValue::GetChildren(model);
	if (f[MF_CALIBRATION].IsNull() || f[MF_THRESHOLD].IsNull() || f[MF_SAMPLING].IsNull()) {
		return false;
	}
	auto cal = StringUtil::Lower(f[MF_CALIBRATION].ToString());
	int32_t code = -1;
	for (int32_t k = 0; k < 3; k++) {
		if (cal == CALIBRATION_NAMES[k]) {
			code = k;
		}
	}
	if (code < 0) {
		return false;
	}
	state.calibration = code;
	state.threshold = f[MF_THRESHOLD].GetValue<double>();
	state.platt_a = DoubleOrNaN(f[MF_PLATT_A]);
	state.platt_b = DoubleOrNaN(f[MF_PLATT_B]);
	state.knots_x.clear();
	state.knots_y.clear();
	if (code == 0) {
		if (f[MF_ISOTONIC_X].IsNull() || f[MF_ISOTONIC_Y].IsNull()) {
			return false;
		}
		for (auto &v : ListValue::GetChildren(f[MF_ISOTONIC_X])) {
			state.knots_x.push_back(DoubleOrNaN(v));
		}
		for (auto &v : ListValue::GetChildren(f[MF_ISOTONIC_Y])) {
			state.knots_y.push_back(DoubleOrNaN(v));
		}
	}
	auto &s = StructValue::GetChildren(f[MF_SAMPLING]);
	for (idx_t k = 0; k < ANOFOX_CBPE_SAMPLING_FIELDS; k++) {
		state.sampling[k] = DoubleOrNaN(s[k]);
	}
	return true;
}

AnofoxCbpeModelParts PartsOf(const CbpeState &state) {
	AnofoxCbpeModelParts p;
	p.knots_x = state.knots_x.empty() ? nullptr : state.knots_x.data();
	p.knots_y = state.knots_y.empty() ? nullptr : state.knots_y.data();
	p.n_knots = std::min<size_t>(state.knots_x.size(), state.knots_y.size());
	if (state.knots_x.size() != state.knots_y.size()) {
		// Unequal knot lists: let upstream reject them (from_parts) by passing an empty map.
		p.n_knots = 0;
	}
	p.calibration = state.calibration;
	p.threshold = state.threshold;
	p.platt_a = state.platt_a;
	p.platt_b = state.platt_b;
	for (idx_t k = 0; k < ANOFOX_CBPE_SAMPLING_FIELDS; k++) {
		p.sampling[k] = state.sampling[k];
	}
	return p;
}

//===--------------------------------------------------------------------===//
// Update
//===--------------------------------------------------------------------===//

//! Argument positions of one aggregate (-1: absent).
struct CbpeBindData : public FunctionData {
	const char *fn = kFit;
	int32_t model_arg = -1;
	int32_t label_arg = -1;
	int32_t score_arg = 0;
	int32_t prediction_arg = -1;
	// cbpe_fit_agg options
	int32_t calibration = 0;
	double threshold = 0.5;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<CbpeBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<CbpeBindData>();
		return fn == o.fn && model_arg == o.model_arg && label_arg == o.label_arg && score_arg == o.score_arg &&
		       prediction_arg == o.prediction_arg && calibration == o.calibration && threshold == o.threshold;
	}
};

void CbpeUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<CbpeBindData>();
	UnifiedVectorFormat score_data, label_data, pred_data, model_data;
	inputs[bind.score_arg].ToUnifiedFormat(count, score_data);
	auto scores = UnifiedVectorFormat::GetData<double>(score_data);
	bool bool_labels = false;
	if (bind.label_arg >= 0) {
		inputs[bind.label_arg].ToUnifiedFormat(count, label_data);
		bool_labels = inputs[bind.label_arg].GetType().id() == LogicalTypeId::BOOLEAN;
	}
	if (bind.prediction_arg >= 0) {
		inputs[bind.prediction_arg].ToUnifiedFormat(count, pred_data);
	}
	if (bind.model_arg >= 0) {
		inputs[bind.model_arg].ToUnifiedFormat(count, model_data);
	}
	CbpeStates states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		if (bind.model_arg >= 0 && !state.has_model && state.error == ERR_NONE &&
		    model_data.validity.RowIsValid(model_data.sel->get_index(i))) {
			if (ReadModel(inputs[bind.model_arg].GetValue(i), state)) {
				state.has_model = true;
			} else {
				state.error = ERR_MODEL;
			}
		}
		auto s_idx = score_data.sel->get_index(i);
		if (!score_data.validity.RowIsValid(s_idx)) {
			continue;
		}
		double label = kNaN;
		if (bind.label_arg >= 0) {
			auto l_idx = label_data.sel->get_index(i);
			if (!label_data.validity.RowIsValid(l_idx)) {
				continue;
			}
			if (bool_labels) {
				label = UnifiedVectorFormat::GetData<bool>(label_data)[l_idx] ? 1.0 : 0.0;
			} else {
				label = UnifiedVectorFormat::GetData<double>(label_data)[l_idx];
				if (label != 0.0 && label != 1.0) {
					if (state.error == ERR_NONE) {
						state.error = ERR_LABEL;
						state.bad_label = label;
					}
					continue;
				}
			}
		}
		double pred = -1.0;
		if (bind.prediction_arg >= 0) {
			auto p_idx = pred_data.sel->get_index(i);
			if (pred_data.validity.RowIsValid(p_idx)) {
				pred = UnifiedVectorFormat::GetData<bool>(pred_data)[p_idx] ? 1.0 : 0.0;
			}
		}
		state.scores.push_back(scores[s_idx]);
		state.labels.push_back(label);
		state.predictions.push_back(pred);
	}
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//

void SetDoubleOrNull(Vector &v, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(v, row, true);
	} else {
		FlatDataMut<double>(v)[row] = value;
	}
}

void AppendDoubles(Vector &list_vec, idx_t row, const double *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	auto out = FlatDataMut<double>(ListVector::GetEntry(list_vec));
	for (idx_t k = 0; k < len; k++) {
		out[offset + k] = data[k];
	}
	ListVector::SetListSize(list_vec, offset + len);
	FlatDataMut<list_entry_t>(list_vec)[row] = {offset, len};
}

//! Throws the data error recorded in Update; false when the state has no rows (NULL result).
bool PrepareState(const CbpeBindData &bind, CbpeState &state, bool needs_model) {
	if (state.error == ERR_LABEL) {
		throw InvalidInputException("%s: y_true must be BOOLEAN or 0/1, got %s", bind.fn,
		                            std::to_string(state.bad_label));
	}
	if (state.error == ERR_MODEL) {
		throw InvalidInputException("%s: model must be the STRUCT returned by cbpe_fit_agg (calibration 'isotonic', "
		                            "'platt' or 'none' with its parameters, threshold and sampling not NULL)",
		                            bind.fn);
	}
	if (state.scores.empty() || (needs_model && !state.has_model)) {
		return false;
	}
	// Canonical row order: bitwise-identical results whatever the thread count (#226).
	CanonicalRows().Key(state.scores).Key(state.labels).Key(state.predictions).Sort();
	return true;
}

//! Per-row prediction codes, or empty when no prediction argument.
vector<int8_t> PredictionCodes(const CbpeBindData &bind, const CbpeState &state) {
	vector<int8_t> codes;
	if (bind.prediction_arg >= 0) {
		codes.reserve(state.predictions.size());
		for (auto p : state.predictions) {
			codes.push_back(static_cast<int8_t>(p));
		}
	}
	return codes;
}

vector<bool> LabelBools(const CbpeState &state) {
	vector<bool> labels;
	labels.reserve(state.labels.size());
	for (auto l : state.labels) {
		labels.push_back(l == 1.0);
	}
	return labels;
}

void CbpeFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                     idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<CbpeBindData>();
	CbpeStates states(state_vector, count);
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		const idx_t row = i + offset;
		if (!PrepareState(bind, state, false)) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		auto label_bools = LabelBools(state);
		// vector<bool> is packed: copy into a contiguous bool array for the FFI.
		std::unique_ptr<bool[]> labels(new bool[label_bools.size()]);
		for (idx_t k = 0; k < label_bools.size(); k++) {
			labels[k] = label_bools[k];
		}
		auto preds = PredictionCodes(bind, state);
		AnofoxCbpeModel m {};
		AnofoxError error;
		if (!anofox_cbpe_fit(state.scores.data(), labels.get(), preds.empty() ? nullptr : preds.data(),
		                     static_cast<size_t>(state.scores.size()), bind.calibration, bind.threshold, &m, &error)) {
			state.Reset();
			ThrowUnlessDegenerate(bind.fn, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto &cal = StructEntry(e[MF_CALIBRATION]);
		FlatDataMut<string_t>(cal)[row] = StringVector::AddString(cal, CALIBRATION_NAMES[m.calibration]);
		FlatDataMut<double>(StructEntry(e[MF_THRESHOLD]))[row] = m.threshold;
		if (m.calibration == 0) {
			AppendDoubles(StructEntry(e[MF_ISOTONIC_X]), row, m.knots_x, m.n_knots);
			AppendDoubles(StructEntry(e[MF_ISOTONIC_Y]), row, m.knots_y, m.n_knots);
		} else {
			FlatVector::SetNull(StructEntry(e[MF_ISOTONIC_X]), row, true);
			FlatVector::SetNull(StructEntry(e[MF_ISOTONIC_Y]), row, true);
		}
		SetDoubleOrNull(StructEntry(e[MF_PLATT_A]), row, m.platt_a);
		SetDoubleOrNull(StructEntry(e[MF_PLATT_B]), row, m.platt_b);
		auto &sampling = StructVector::GetEntries(StructEntry(e[MF_SAMPLING]));
		for (idx_t k = 0; k < ANOFOX_CBPE_SAMPLING_FIELDS; k++) {
			SetDoubleOrNull(StructEntry(sampling[k]), row, m.sampling[k]);
		}
		FlatDataMut<int64_t>(StructEntry(e[MF_N_REFERENCE]))[row] = static_cast<int64_t>(m.n_reference);
		auto &mt = StructEntry(e[MF_MODEL_TYPE]);
		FlatDataMut<string_t>(mt)[row] = StringVector::AddString(mt, "cbpe");
		anofox_free_cbpe_model(&m);
		state.Reset();
	}
}

void CbpeEstimateFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                          idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<CbpeBindData>();
	CbpeStates states(state_vector, count);
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		const idx_t row = i + offset;
		if (!PrepareState(bind, state, true)) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		auto preds = PredictionCodes(bind, state);
		AnofoxCbpeEstimate est {};
		AnofoxError error;
		if (!anofox_cbpe_estimate(PartsOf(state), state.scores.data(), preds.empty() ? nullptr : preds.data(),
		                          static_cast<size_t>(state.scores.size()), &est, &error)) {
			state.Reset();
			ThrowUnlessDegenerate(bind.fn, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(e[0]))[row] = static_cast<int64_t>(est.n);
		for (idx_t k = 0; k < ANOFOX_CBPE_METRICS; k++) {
			auto &m = StructVector::GetEntries(StructEntry(e[k + 1]));
			SetDoubleOrNull(StructEntry(m[0]), row, est.metrics[k].value);
			SetDoubleOrNull(StructEntry(m[1]), row, est.metrics[k].sampling_error);
			SetDoubleOrNull(StructEntry(m[2]), row, est.metrics[k].lower_confidence);
			SetDoubleOrNull(StructEntry(m[3]), row, est.metrics[k].upper_confidence);
		}
		state.Reset();
	}
}

void CbpeRealizedFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                          idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<CbpeBindData>();
	CbpeStates states(state_vector, count);
	auto &e = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		const idx_t row = i + offset;
		if (!PrepareState(bind, state, true)) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		auto label_bools = LabelBools(state);
		std::unique_ptr<bool[]> labels(new bool[label_bools.size()]);
		for (idx_t k = 0; k < label_bools.size(); k++) {
			labels[k] = label_bools[k];
		}
		auto preds = PredictionCodes(bind, state);
		AnofoxCbpeRealized real {};
		AnofoxError error;
		if (!anofox_cbpe_realized(PartsOf(state), state.scores.data(), labels.get(),
		                          preds.empty() ? nullptr : preds.data(), static_cast<size_t>(state.scores.size()),
		                          &real, &error)) {
			state.Reset();
			ThrowUnlessDegenerate(bind.fn, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		FlatDataMut<int64_t>(StructEntry(e[0]))[row] = static_cast<int64_t>(real.n);
		for (idx_t k = 0; k < ANOFOX_CBPE_METRICS; k++) {
			auto &child = StructEntry(e[k + 1]);
			if (k < kFirstCount) {
				SetDoubleOrNull(child, row, real.metrics[k]);
			} else {
				FlatDataMut<int64_t>(child)[row] = static_cast<int64_t>(real.metrics[k]);
			}
		}
		state.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//

void VisitOptions(const char *fn, const Value &options,
                  const std::function<void(const string &, const Value &)> &visit) {
	if (options.IsNull()) {
		return;
	}
	if (options.type().id() == LogicalTypeId::STRUCT) {
		auto &vals = StructValue::GetChildren(options);
		for (idx_t i = 0; i < vals.size(); i++) {
			visit(StringUtil::Lower(NameString(StructType::GetChildName(options.type(), i))), UnwrapVariant(vals[i]));
		}
	} else if (options.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(options)) {
			auto &kv = StructValue::GetChildren(entry);
			visit(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else {
		throw InvalidInputException("%s: options must be a MAP or STRUCT", fn);
	}
}

unique_ptr<FunctionData> CbpeFitBind(ClientContext &context, AggregateBindFunction &function,
                                     vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<CbpeBindData>();
	const char *fn = kFit;
	bind->fn = fn;
	bind->label_arg = 0;
	bind->score_arg = 1;
	idx_t options_arg = 2;
	if (arguments.size() == 4) {
		bind->prediction_arg = 2;
		options_arg = 3;
	}
	if (arguments.size() > options_arg) {
		VisitOptions(
		    fn, EvaluateConstantOptions(context, *arguments[options_arg], fn), [&](const string &key, const Value &v) {
			    if (v.IsNull()) {
				    throw InvalidInputException("%s: option '%s' must not be NULL", fn, key);
			    }
			    if (key == "calibration") {
				    auto c = StringUtil::Lower(v.ToString());
				    if (c == "isotonic") {
					    bind->calibration = 0;
				    } else if (c == "platt" || c == "sigmoid") {
					    bind->calibration = 1;
				    } else if (c == "none") {
					    bind->calibration = 2;
				    } else {
					    throw InvalidInputException(
					        "%s: option 'calibration' must be 'isotonic', 'platt' or 'none', got '%s'", fn, c);
				    }
			    } else if (key == "threshold") {
				    auto t = v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
				    if (!std::isfinite(t)) {
					    throw InvalidInputException("%s: option 'threshold' must be finite", fn);
				    }
				    bind->threshold = t;
			    } else {
				    throw InvalidInputException("%s: unknown option '%s' (supported: calibration, threshold)", fn, key);
			    }
		    });
	}
	SetReturnTypeOf(function, ModelType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//! Casts the model argument to ModelType() (by field name).
void CastModelArgument(ClientContext &context, vector<unique_ptr<Expression>> &arguments, idx_t idx, const char *fn) {
	auto &type = ReturnTypeOf(*arguments[idx]);
	if (type.id() == LogicalTypeId::SQLNULL) {
		arguments[idx] = BoundCastExpression::AddCastToType(context, std::move(arguments[idx]), ModelType());
		return;
	}
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: model must be the STRUCT returned by cbpe_fit_agg, got %s", fn,
		                            type.ToString());
	}
	auto target = ModelType();
	auto &have = StructType::GetChildTypes(type);
	for (auto &want : StructType::GetChildTypes(target)) {
		bool found = false;
		for (auto &h : have) {
			found = found || NameString(h.first) == NameString(want.first);
		}
		if (!found) {
			throw InvalidInputException("%s: model must be the STRUCT returned by cbpe_fit_agg (no field '%s')", fn,
			                            NameString(want.first));
		}
	}
	if (type != target) {
		arguments[idx] = BoundCastExpression::AddCastToType(context, std::move(arguments[idx]), target);
	}
}

template <bool REALIZED>
unique_ptr<FunctionData> CbpeApplyBind(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<CbpeBindData>();
	const char *fn = REALIZED ? kRealized : kEstimate;
	bind->fn = fn;
	bind->model_arg = 0;
	if (REALIZED) {
		bind->label_arg = 1;
		bind->score_arg = 2;
		bind->prediction_arg = arguments.size() >= 4 ? 3 : -1;
	} else {
		bind->score_arg = 1;
		bind->prediction_arg = arguments.size() >= 3 ? 2 : -1;
	}
	CastModelArgument(context, arguments, 0, fn);
	SetReturnTypeOf(function, REALIZED ? RealizedType() : EstimateType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// cbpe_by
//===--------------------------------------------------------------------===//

// One row per chunk and metric, reference chunks first. The model is fitted on the rows where `is_reference`
// is true; every chunk is estimated, and realized where it has labels.
const char *const CBPE_BY_SQL = R"(
WITH _anofox_src AS (
    SELECT chunk_col, (is_reference)::BOOLEAN AS _anofox_ref, y_col AS _anofox_y, score_col AS _anofox_s,
           (prediction)::BOOLEAN AS _anofox_p
    FROM query_table(source::VARCHAR)
),
_anofox_model AS (
    SELECT cbpe_fit_agg(_anofox_y, _anofox_s, _anofox_p, options) AS _anofox_m
    FROM _anofox_src WHERE _anofox_ref
),
_anofox_chunks AS (
    SELECT chunk_col, bool_or(coalesce(_anofox_ref, false)) AS _anofox_isref,
           cbpe_estimate_agg(_anofox_m, _anofox_s, _anofox_p) AS _anofox_e,
           cbpe_realized_agg(_anofox_m, _anofox_y, _anofox_s, _anofox_p) AS _anofox_r
    FROM _anofox_src CROSS JOIN _anofox_model
    GROUP BY chunk_col
),
_anofox_long AS (
    SELECT chunk_col, _anofox_isref, (_anofox_e).n AS n, (_anofox_r).n AS n_labelled, unnest([
        {'k': 1, 'metric': 'roc_auc', 'e': (_anofox_e).roc_auc, 'r': (_anofox_r).roc_auc::DOUBLE},
        {'k': 2, 'metric': 'f1', 'e': (_anofox_e).f1, 'r': (_anofox_r).f1::DOUBLE},
        {'k': 3, 'metric': 'precision', 'e': (_anofox_e).precision, 'r': (_anofox_r).precision::DOUBLE},
        {'k': 4, 'metric': 'recall', 'e': (_anofox_e).recall, 'r': (_anofox_r).recall::DOUBLE},
        {'k': 5, 'metric': 'specificity', 'e': (_anofox_e).specificity, 'r': (_anofox_r).specificity::DOUBLE},
        {'k': 6, 'metric': 'accuracy', 'e': (_anofox_e).accuracy, 'r': (_anofox_r).accuracy::DOUBLE},
        {'k': 7, 'metric': 'true_positive', 'e': (_anofox_e).true_positive, 'r': (_anofox_r).true_positive::DOUBLE},
        {'k': 8, 'metric': 'false_positive', 'e': (_anofox_e).false_positive, 'r': (_anofox_r).false_positive::DOUBLE},
        {'k': 9, 'metric': 'false_negative', 'e': (_anofox_e).false_negative, 'r': (_anofox_r).false_negative::DOUBLE},
        {'k': 10, 'metric': 'true_negative', 'e': (_anofox_e).true_negative, 'r': (_anofox_r).true_negative::DOUBLE}
    ]) AS _anofox_x
    FROM _anofox_chunks
)
SELECT chunk_col, _anofox_isref AS is_reference, (_anofox_x).metric AS metric, n,
       (_anofox_x).e.value AS estimate, (_anofox_x).e.sampling_error AS sampling_error,
       (_anofox_x).e.lower_confidence AS lower_confidence, (_anofox_x).e.upper_confidence AS upper_confidence,
       n_labelled, (_anofox_x).r AS realized
FROM _anofox_long
ORDER BY _anofox_isref DESC, chunk_col, (_anofox_x).k
)";

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

AggregateFunction MakeAggregate(const char *name, vector<LogicalType> args, aggregate_finalize_t finalize,
                                bind_aggregate_function_t bind) {
	return AggregateFunction(name, std::move(args), LogicalType::ANY, AggregateFunction::StateSize<CbpeState>,
	                         ANOFOX_AGG_INIT(CbpeInitialize),
	                         ANOFOX_GUARDED_UPDATE(CbpeUpdate, CbpeDestroy, CbpeInitialize), CbpeCombine, finalize,
	                         nullptr, bind, CbpeDestroy);
}

void Describe(CreateAggregateFunctionInfo &info, const string &description, const string &example,
              const vector<string> &names, const vector<LogicalType> &types) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = {"monitoring"};
	d.parameter_names = names;
	d.parameter_types = types;
	info.descriptions.push_back(std::move(d));
}

void RegisterSet(ExtensionLoader &loader, AggregateFunctionSet &set,
                 const std::function<void(CreateAggregateFunctionInfo &)> &describe) {
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	describe(info);
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterCbpeFunctions(ExtensionLoader &loader) {
	const auto D = LogicalType::DOUBLE;
	const auto B = LogicalType::BOOLEAN;
	const auto A = LogicalType::ANY;
	auto fit_final = ANOFOX_GUARDED_FINALIZE(CbpeFitFinalize, CbpeDestroy, CbpeInitialize);
	auto est_final = ANOFOX_GUARDED_FINALIZE(CbpeEstimateFinalize, CbpeDestroy, CbpeInitialize);
	auto real_final = ANOFOX_GUARDED_FINALIZE(CbpeRealizedFinalize, CbpeDestroy, CbpeInitialize);
	{
		AggregateFunctionSet set("cbpe_fit_agg");
		vector<vector<LogicalType>> overloads;
		for (auto &y : {B, D}) {
			overloads.push_back({y, D});
			overloads.push_back({y, D, A});
			overloads.push_back({y, D, B, A});
		}
		for (auto &args : overloads) {
			set.AddFunction(MakeAggregate("cbpe_fit_agg", args, fit_final, ANOFOX_AGG_BIND(CbpeFitBind)));
		}
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			const string desc =
			    "Fits NannyML's CBPE (confidence-based performance estimation, binary classification) on a labelled "
			    "reference period: calibrates the scores against y_true (BOOLEAN or 0/1) and stores the "
			    "sampling-error components. Returns the model STRUCT(calibration, threshold, isotonic_x, "
			    "isotonic_y, platt_a, platt_b, sampling, n_reference, model_type) for cbpe_estimate_agg and "
			    "cbpe_realized_agg. Options: calibration ('isotonic' default, 'platt', 'none'), threshold "
			    "(default 0.5; a row is predicted positive when score >= threshold). The optional prediction "
			    "column gives the model's predictions instead (a NULL falls back to the threshold). NULL when "
			    "the reference has one class.";
			// One description per overload, in registration order.
			for (auto &y : {B, D}) {
				Describe(info, desc, "cbpe_fit_agg(y_true, score)", {"y_true", "score"}, {y, D});
				Describe(info, desc, "cbpe_fit_agg(y_true, score, {'calibration': 'platt'})",
				         {"y_true", "score", "options"}, {y, D, A});
				Describe(info, desc, "cbpe_fit_agg(y_true, score, y_pred, NULL)",
				         {"y_true", "score", "prediction", "options"}, {y, D, B, A});
			}
		});
	}
	{
		AggregateFunctionSet set("cbpe_estimate_agg");
		set.AddFunction(MakeAggregate("cbpe_estimate_agg", {A, D}, est_final, ANOFOX_AGG_BIND(CbpeApplyBind<false>)));
		set.AddFunction(
		    MakeAggregate("cbpe_estimate_agg", {A, D, B}, est_final, ANOFOX_AGG_BIND(CbpeApplyBind<false>)));
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			const string desc =
			    "CBPE estimate of a binary classifier's performance on a chunk of unlabelled scores with a "
			    "cbpe_fit_agg model: STRUCT(n, roc_auc, f1, precision, recall, specificity, accuracy, "
			    "true_positive, false_positive, false_negative, true_negative), each STRUCT(value, "
			    "sampling_error, lower_confidence, upper_confidence) with NannyML's band value -/+ 3 sampling "
			    "errors (clipped to [0, 1], counts to >= 0). The model is taken from the first row where it is "
			    "not NULL.";
			Describe(info, desc, "cbpe_estimate_agg(m, score)", {"model", "score"}, {A, D});
			Describe(info, desc, "cbpe_estimate_agg(m, score, y_pred)", {"model", "score", "prediction"}, {A, D, B});
		});
	}
	{
		AggregateFunctionSet set("cbpe_realized_agg");
		for (auto &y : {B, D}) {
			set.AddFunction(
			    MakeAggregate("cbpe_realized_agg", {A, y, D}, real_final, ANOFOX_AGG_BIND(CbpeApplyBind<true>)));
			set.AddFunction(
			    MakeAggregate("cbpe_realized_agg", {A, y, D, B}, real_final, ANOFOX_AGG_BIND(CbpeApplyBind<true>)));
		}
		RegisterSet(loader, set, [&](CreateAggregateFunctionInfo &info) {
			const string desc =
			    "Realized performance of a labelled chunk with a cbpe_fit_agg model's threshold (or the given "
			    "predictions), as NannyML: STRUCT(n, roc_auc, f1, precision, recall, specificity, accuracy, "
			    "true_positive, false_positive, false_negative, true_negative). ROC AUC is NULL with one label "
			    "class; F1, precision and recall are NULL with one label or one predicted class; specificity "
			    "is NULL without actual negatives. Rows with a NULL y_true are skipped.";
			for (auto &y : {B, D}) {
				Describe(info, desc, "cbpe_realized_agg(m, y_true, score)", {"model", "y_true", "score"}, {A, y, D});
				Describe(info, desc, "cbpe_realized_agg(m, y_true, score, y_pred)",
				         {"model", "y_true", "score", "prediction"}, {A, y, D, B});
			}
		});
	}
	{
		auto macro = CreateContractTableMacro(
		    "cbpe_by", {"source", "chunk_col", "y_col", "score_col", "is_reference"},
		    {{"options", "NULL"}, {"prediction", "NULL"}}, CBPE_BY_SQL,
		    "Table macro: NannyML CBPE per chunk. Fits cbpe_fit_agg(y_col, score_col, prediction, options) on the "
		    "rows of source where the BOOLEAN expression is_reference is true, then returns one row per chunk and "
		    "metric (reference chunks first) (roc_auc, f1, precision, recall, specificity, accuracy, true_positive, "
		    "false_positive, "
		    "false_negative, true_negative): chunk_col, is_reference, metric, n, estimate, sampling_error, "
		    "lower_confidence, upper_confidence (cbpe_estimate_agg) and n_labelled, realized "
		    "(cbpe_realized_agg over the chunk's rows with a label; NULL without labels).",
		    "cbpe_by('scores', week, y_true, score, period = 'reference')", {"monitoring", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

} // namespace duckdb
