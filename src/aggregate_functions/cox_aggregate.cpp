#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "telemetry.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Cox proportional hazards regression (#312), survival::coxph via
// anofox-regression's survival::cox.
//
//   cox_fit_agg(time DOUBLE, x DOUBLE[], event DOUBLE [, weight DOUBLE] [, options])
//   cox_zph_agg(...same arguments...)         -- cox.zph proportional-hazards test
//   cox_augment_agg(...same arguments...)     -- per-row residuals (window aggregate)
//   cox_survival(model, x, times [, stratum]) -- survfit(fit, newdata) point estimate
//
// `event` is 1 for an observed event and 0 for a censored row. Offsets, strata,
// counting-process entry times and robust clusters are columns of x named by
// their 1-based position in the options (as the GLMs' `offset`); they are
// removed from the design.
//===--------------------------------------------------------------------===//

namespace {

enum class CoxMode { FIT, ZPH, AUGMENT };

struct CoxBindData : public FunctionData {
	CoxMode mode = CoxMode::FIT;
	bool weighted = false;
	AnofoxCoxOptions options {1e-9, 0.0, 0.95, 20, 0, 0, true, false};
	bool compute_inference = true;
	// 1-based positions in x; 0 = none.
	idx_t offset_col = 0, strata_col = 0, entry_col = 0, cluster_col = 0;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<CoxBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<CoxBindData>();
		return mode == o.mode && weighted == o.weighted && options.tolerance == o.options.tolerance &&
		       options.ridge == o.options.ridge && options.confidence_level == o.options.confidence_level &&
		       options.max_iterations == o.options.max_iterations && options.ties == o.options.ties &&
		       options.zph_transform == o.options.zph_transform && options.timefix == o.options.timefix &&
		       options.robust == o.options.robust && compute_inference == o.compute_inference &&
		       offset_col == o.offset_col && strata_col == o.strata_col && entry_col == o.entry_col &&
		       cluster_col == o.cluster_col;
	}
};

const char *ModeName(CoxMode mode) {
	switch (mode) {
	case CoxMode::ZPH:
		return "cox_zph_agg";
	case CoxMode::AUGMENT:
		return "cox_augment_agg";
	default:
		return "cox_fit_agg";
	}
}

//! Walk a MAP / STRUCT options value as (lower-case key, value) pairs.
template <class FN>
void VisitCoxOptions(const Value &options, FN &&fn) {
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

void ParseCoxOptions(const Value &value, CoxBindData &bind) {
	const string fn = ModeName(bind.mode);
	auto as_double = [&](const string &key, const Value &v) {
		try {
			return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
		}
	};
	auto as_bool = [&](const string &key, const Value &v) {
		try {
			return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a boolean, got '%s'", fn, key, v.ToString());
		}
	};
	auto as_column = [&](const string &key, const Value &v) -> idx_t {
		double d = as_double(key, v);
		if (!(d >= 1.0) || d != std::floor(d) || d > 1e9) {
			throw InvalidInputException("%s: option '%s' must be a 1-based position in x, got '%s'", fn, key,
			                            v.ToString());
		}
		return (idx_t)d;
	};
	VisitCoxOptions(value, [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "ties" || key == "method") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "efron") {
				bind.options.ties = 0;
			} else if (s == "breslow") {
				bind.options.ties = 1;
			} else {
				throw InvalidInputException("%s: unknown ties '%s'. Expected 'efron' or 'breslow'.", fn, v.ToString());
			}
		} else if (key == "ridge" || key == "lambda") {
			double d = as_double(key, v);
			if (!(d >= 0.0) || !std::isfinite(d)) {
				throw InvalidInputException("%s: ridge must be finite and >= 0, got %g", fn, d);
			}
			bind.options.ridge = d;
		} else if (key == "max_iterations" || key == "max_iter") {
			double d = as_double(key, v);
			if (!(d >= 1.0) || d > 1e6) {
				throw InvalidInputException("%s: max_iterations must be between 1 and 1000000, got %g", fn, d);
			}
			bind.options.max_iterations = (uint32_t)d;
		} else if (key == "tolerance" || key == "eps") {
			double d = as_double(key, v);
			if (!(d > 0.0) || !std::isfinite(d)) {
				throw InvalidInputException("%s: tolerance must be positive, got %g", fn, d);
			}
			bind.options.tolerance = d;
		} else if (key == "confidence_level") {
			double d = as_double(key, v);
			if (!(d > 0.0 && d < 1.0)) {
				throw InvalidInputException("%s: confidence_level must be in (0, 1), got %g", fn, d);
			}
			bind.options.confidence_level = d;
		} else if (key == "compute_inference") {
			bind.compute_inference = as_bool(key, v);
		} else if (key == "timefix") {
			bind.options.timefix = as_bool(key, v);
		} else if (key == "robust") {
			bind.options.robust = as_bool(key, v);
		} else if (key == "offset") {
			bind.offset_col = as_column(key, v);
		} else if (key == "strata") {
			bind.strata_col = as_column(key, v);
		} else if (key == "entry" || key == "start") {
			bind.entry_col = as_column(key, v);
		} else if (key == "cluster") {
			bind.cluster_col = as_column(key, v);
		} else if (key == "transform" && bind.mode == CoxMode::ZPH) {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "km") {
				bind.options.zph_transform = 0;
			} else if (s == "rank") {
				bind.options.zph_transform = 1;
			} else if (s == "identity") {
				bind.options.zph_transform = 2;
			} else if (s == "log") {
				bind.options.zph_transform = 3;
			} else {
				throw InvalidInputException("%s: unknown transform '%s'. Expected 'km', 'rank', 'identity' or 'log'.",
				                            fn, v.ToString());
			}
		} else {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: ties, ridge, max_iterations, "
			                            "tolerance, confidence_level, compute_inference, timefix, robust, offset, "
			                            "strata, entry, cluster%s",
			                            fn, key, bind.mode == CoxMode::ZPH ? ", transform" : "");
		}
	});
	idx_t cols[] = {bind.offset_col, bind.strata_col, bind.entry_col, bind.cluster_col};
	for (idx_t a = 0; a < 4; a++) {
		for (idx_t b = a + 1; b < 4; b++) {
			if (cols[a] != 0 && cols[a] == cols[b]) {
				throw InvalidInputException("%s: offset, strata, entry and cluster must name different columns of x",
				                            fn);
			}
		}
	}
}

struct CoxState {
	vector<double> time;    // NaN for a NULL
	vector<double> event;   // NaN for a NULL
	vector<double> weights; // only when weighted; NaN for a NULL
	vector<vector<double>> x_rows; // empty for a NULL x list
	idx_t n_features = 0;
	bool initialized = false;
};

void CoxInitialize(data_ptr_t state_p) {
	new (state_p) CoxState();
}

void CoxDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (CoxState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~CoxState();
	}
}

//! A DOUBLE view of a numeric argument (cast when it is another numeric type).
struct DoubleInput {
	unique_ptr<Vector> cast;
	UnifiedVectorFormat data;
	const double *values = nullptr;

	void Load(Vector &input, idx_t count) {
		Vector *source = &input;
		if (input.GetType().id() != LogicalTypeId::DOUBLE) {
			cast = make_uniq<Vector>(LogicalType::DOUBLE, count);
			VectorOperations::DefaultCast(input, *cast, count);
			source = cast.get();
		}
		source->ToUnifiedFormat(count, data);
		values = UnifiedVectorFormat::GetData<double>(data);
	}
	double Get(idx_t i) const {
		auto idx = data.sel->get_index(i);
		return data.validity.RowIsValid(idx) ? values[idx] : std::numeric_limits<double>::quiet_NaN();
	}
};

void CoxUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<CoxBindData>();
	DoubleInput time_in, event_in, weight_in;
	time_in.Load(inputs[0], count);
	event_in.Load(inputs[2], count);
	if (bind.weighted) {
		weight_in.Load(inputs[3], count);
	}
	UnifiedVectorFormat x_data;
	inputs[1].ToUnifiedFormat(count, x_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (CoxState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		vector<double> x_row;
		auto x_idx = x_data.sel->get_index(i);
		if (x_data.validity.RowIsValid(x_idx)) {
			auto entry = x_list_data[x_idx];
			if (!state.initialized) {
				state.n_features = entry.length;
				state.initialized = true;
			} else if (entry.length != state.n_features) {
				throw InvalidInputException("%s: inconsistent feature count: expected %llu, got %llu",
				                            ModeName(bind.mode), (unsigned long long)state.n_features,
				                            (unsigned long long)entry.length);
			}
			x_row.resize(entry.length);
			for (idx_t j = 0; j < entry.length; j++) {
				auto c = entry.offset + j;
				// A NULL element makes the row non-finite: it is left out of the fit.
				x_row[j] = x_child_validity.RowIsValid(c) ? x_child_data[c] : std::numeric_limits<double>::quiet_NaN();
			}
		} else if (bind.mode != CoxMode::AUGMENT) {
			continue;
		}
		state.time.push_back(time_in.Get(i));
		state.event.push_back(event_in.Get(i));
		if (bind.weighted) {
			state.weights.push_back(weight_in.Get(i));
		}
		state.x_rows.push_back(std::move(x_row));
	}
}

void CoxCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (CoxState **)source_data.data;
	auto targets = (CoxState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.initialized && target.initialized && source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count during combine");
		}
		if (source.initialized && !target.initialized) {
			target.n_features = source.n_features;
			target.initialized = true;
		}
		// Appended in order (the window frame order of cox_augment_agg).
		target.time.insert(target.time.end(), source.time.begin(), source.time.end());
		target.event.insert(target.event.end(), source.event.begin(), source.event.end());
		target.weights.insert(target.weights.end(), source.weights.begin(), source.weights.end());
		for (auto &row : source.x_rows) {
			target.x_rows.push_back(row);
		}
	}
}

//! The columns of one fit, in canonical row order, and the FFI view of them.
struct CoxCall {
	vector<double> time, event, weights, x, offset, strata, entry, cluster;
	vector<idx_t> design; // 0-based positions in x of the design columns
	vector<idx_t> order;  // sorted row k was input row order[k] (empty: identity)
	AnofoxCoxData data {};
};

void PrepareCall(const CoxBindData &bind, CoxState &state, CoxCall &call) {
	const string fn = ModeName(bind.mode);
	const idx_t n = state.time.size();
	const idx_t p = state.n_features;
	for (idx_t c : {bind.offset_col, bind.strata_col, bind.entry_col, bind.cluster_col}) {
		if (c > p) {
			throw InvalidInputException("%s: option column %llu is beyond the %llu columns of x", fn,
			                            (unsigned long long)c, (unsigned long long)p);
		}
	}
	vector<vector<double>> columns(p, vector<double>(n));
	for (idx_t r = 0; r < n; r++) {
		for (idx_t j = 0; j < p; j++) {
			columns[j][r] = state.x_rows[r].empty() ? std::numeric_limits<double>::quiet_NaN() : state.x_rows[r][j];
		}
	}
	call.time = std::move(state.time);
	call.event = std::move(state.event);
	call.weights = std::move(state.weights);
	// Canonical row order: bitwise-identical results whatever the thread count (#226).
	CanonicalRows rows;
	rows.Keys(columns).Key(call.time).Key(call.event);
	if (bind.weighted) {
		rows.Key(call.weights);
	}
	call.order = rows.Sort();

	for (idx_t j = 0; j < p; j++) {
		const idx_t pos = j + 1;
		if (pos == bind.offset_col) {
			call.offset = std::move(columns[j]);
		} else if (pos == bind.strata_col) {
			call.strata = std::move(columns[j]);
		} else if (pos == bind.entry_col) {
			call.entry = std::move(columns[j]);
		} else if (pos == bind.cluster_col) {
			call.cluster = std::move(columns[j]);
		} else {
			call.design.push_back(j);
			call.x.insert(call.x.end(), columns[j].begin(), columns[j].end());
		}
	}
	auto ptr = [](const vector<double> &v, bool present) { return present ? v.data() : nullptr; };
	call.data.time = call.time.data();
	call.data.event = call.event.data();
	call.data.x = call.x.empty() ? nullptr : call.x.data();
	call.data.weights = ptr(call.weights, bind.weighted);
	call.data.offset = ptr(call.offset, bind.offset_col != 0);
	call.data.strata = ptr(call.strata, bind.strata_col != 0);
	call.data.entry = ptr(call.entry, bind.entry_col != 0);
	call.data.cluster = ptr(call.cluster, bind.cluster_col != 0);
	call.data.n_rows = n;
	call.data.n_features = call.design.size();
}

//===--------------------------------------------------------------------===//
// cox_fit_agg
//===--------------------------------------------------------------------===//
LogicalType BaselineType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("stratum", LogicalType::DOUBLE));
	c.push_back(make_pair("time", LogicalType::DOUBLE));
	c.push_back(make_pair("cumhaz", LogicalType::DOUBLE));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(c)));
}

LogicalType CoxFitResultType() {
	child_list_t<LogicalType> c;
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	c.push_back(make_pair("coefficients", dlist));
	c.push_back(make_pair("hazard_ratios", dlist));
	c.push_back(make_pair("log_likelihood", LogicalType::DOUBLE));
	c.push_back(make_pair("null_log_likelihood", LogicalType::DOUBLE));
	c.push_back(make_pair("lr_statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("lr_p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("wald_statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("wald_p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("score_statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("score_p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("test_df", LogicalType::BIGINT));
	c.push_back(make_pair("concordance", LogicalType::DOUBLE));
	c.push_back(make_pair("pseudo_r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_events", LogicalType::BIGINT));
	c.push_back(make_pair("n_censored", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("n_strata", LogicalType::BIGINT));
	c.push_back(make_pair("iterations", LogicalType::INTEGER));
	c.push_back(make_pair("converged", LogicalType::BOOLEAN));
	c.push_back(make_pair("ties", LogicalType::VARCHAR));
	c.push_back(make_pair("robust", LogicalType::BOOLEAN));
	// Stable shape (#152): inference fields always present, NULL without compute_inference.
	AppendCoefficientInferenceFields(c, "z_values", false);
	c.push_back(make_pair("hr_conf_low", dlist));
	c.push_back(make_pair("hr_conf_high", dlist));
	c.push_back(make_pair("naive_std_errors", dlist));
	c.push_back(make_pair("vcov", dlist));
	c.push_back(make_pair("baseline", BaselineType()));
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void SetList(Vector &target, idx_t row, const double *values, idx_t len) {
	auto list_data = FlatDataMut<list_entry_t>(target);
	auto offset = ListVector::GetListSize(target);
	ListVector::Reserve(target, offset + len);
	auto &child = ListVector::GetEntry(target);
	auto child_values = FlatDataMut<double>(child);
	for (idx_t j = 0; j < len; j++) {
		if (std::isfinite(values[j])) {
			child_values[offset + j] = values[j];
		} else {
			FlatVector::SetNull(child, offset + j, true);
		}
	}
	ListVector::SetListSize(target, offset + len);
	list_data[row] = list_entry_t(offset, len);
}

void SetDouble(Vector &v, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(v)[row] = value;
	} else {
		FlatVector::SetNull(v, row, true);
	}
}

void CoxFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                    idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<CoxBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (CoxState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	auto &types = StructType::GetChildTypes(result.GetType());
	auto field = [&](const char *name) -> Vector & {
		for (idx_t k = 0; k < types.size(); k++) {
			if (types[k].first == name) {
				return StructEntry(entries[k]);
			}
		}
		throw InternalException("cox_fit_agg: no field %s", name);
	};
	ModelStructWriter writer(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (!state.initialized || state.time.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		CoxCall call;
		PrepareCall(bind, state, call);
		AnofoxCoxFitResult fit {};
		AnofoxError error;
		if (!anofox_cox_fit(&call.data, bind.options, &fit, &error)) {
			ThrowUnlessDegenerate("cox_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		const idx_t p = fit.n_features;
		SetList(field("coefficients"), row, fit.coefficients, p);
		SetList(field("hazard_ratios"), row, fit.hazard_ratios, p);
		SetDouble(field("log_likelihood"), row, fit.log_likelihood);
		SetDouble(field("null_log_likelihood"), row, fit.null_log_likelihood);
		SetDouble(field("lr_statistic"), row, fit.lr_statistic);
		SetDouble(field("lr_p_value"), row, fit.lr_p_value);
		SetDouble(field("wald_statistic"), row, fit.wald_statistic);
		SetDouble(field("wald_p_value"), row, fit.wald_p_value);
		SetDouble(field("score_statistic"), row, fit.score_statistic);
		SetDouble(field("score_p_value"), row, fit.score_p_value);
		FlatDataMut<int64_t>(field("test_df"))[row] = (int64_t)fit.df;
		SetDouble(field("concordance"), row, fit.concordance);
		SetDouble(field("pseudo_r_squared"), row, fit.r_squared);
		FlatDataMut<int64_t>(field("n_observations"))[row] = (int64_t)fit.n_observations;
		FlatDataMut<int64_t>(field("n_events"))[row] = (int64_t)fit.n_events;
		FlatDataMut<int64_t>(field("n_censored"))[row] = (int64_t)(fit.n_observations - fit.n_events);
		FlatDataMut<int64_t>(field("n_features"))[row] = (int64_t)p;
		FlatDataMut<int64_t>(field("n_strata"))[row] = (int64_t)fit.n_strata;
		FlatDataMut<int32_t>(field("iterations"))[row] = (int32_t)fit.iterations;
		FlatDataMut<bool>(field("converged"))[row] = fit.converged;
		field("ties").SetValue(row, Value(bind.options.ties == 1 ? "breslow" : "efron"));
		FlatDataMut<bool>(field("robust"))[row] = fit.robust;

		if (bind.compute_inference) {
			AnofoxFitResultInference view = {fit.std_errors, fit.z_values, fit.p_values, fit.conf_low,
			                                 fit.conf_high,  static_cast<size_t>(p), fit.confidence_level, NAN, NAN};
			writer.WriteInference(row, &view);
			SetList(field("hr_conf_low"), row, fit.hr_conf_low, p);
			SetList(field("hr_conf_high"), row, fit.hr_conf_high, p);
			SetList(field("naive_std_errors"), row, fit.naive_std_errors, p);
			SetList(field("vcov"), row, fit.vcov, p * p);
		} else {
			writer.WriteInference(row, nullptr);
			for (auto name : {"hr_conf_low", "hr_conf_high", "naive_std_errors", "vcov"}) {
				FlatVector::SetNull(field(name), row, true);
			}
		}

		auto &base = field("baseline");
		auto base_offset = ListVector::GetListSize(base);
		ListVector::Reserve(base, base_offset + fit.n_baseline);
		auto &base_entries = StructVector::GetEntries(ListVector::GetEntry(base));
		for (idx_t k = 0; k < fit.n_baseline; k++) {
			SetDouble(StructEntry(base_entries[0]), base_offset + k, fit.baseline_stratum[k]);
			SetDouble(StructEntry(base_entries[1]), base_offset + k, fit.baseline_time[k]);
			SetDouble(StructEntry(base_entries[2]), base_offset + k, fit.baseline_cumhaz[k]);
		}
		ListVector::SetListSize(base, base_offset + fit.n_baseline);
		FlatDataMut<list_entry_t>(base)[row] = list_entry_t(base_offset, fit.n_baseline);

		writer.WriteSummary(row, fit.summary);
		anofox_free_cox_fit_result(&fit);
	}
}

//===--------------------------------------------------------------------===//
// cox_zph_agg
//===--------------------------------------------------------------------===//
LogicalType CoxZphResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(c)));
}

void CoxZphFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                    idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<CoxBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (CoxState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (!state.initialized || state.time.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		CoxCall call;
		PrepareCall(bind, state, call);
		AnofoxCoxZphResult zph {};
		AnofoxError error;
		if (!anofox_cox_zph(&call.data, bind.options, &zph, &error)) {
			ThrowUnlessDegenerate("cox_zph_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + zph.len);
		auto &e = StructVector::GetEntries(ListVector::GetEntry(result));
		for (idx_t k = 0; k < zph.len; k++) {
			auto out = list_offset + k;
			// Terms are named by their position in the design (x without the offset / strata / entry /
			// cluster columns), as tidy() names the coefficients.
			StructEntry(e[0]).SetValue(out, Value(k + 1 == zph.len ? string("GLOBAL") : "x" + std::to_string(k + 1)));
			SetDouble(StructEntry(e[1]), out, zph.chisq[k]);
			SetDouble(StructEntry(e[2]), out, zph.df[k]);
			SetDouble(StructEntry(e[3]), out, zph.p_value[k]);
		}
		ListVector::SetListSize(result, list_offset + zph.len);
		FlatDataMut<list_entry_t>(result)[row] = list_entry_t(list_offset, zph.len);
		anofox_free_cox_zph_result(&zph);
	}
}

//===--------------------------------------------------------------------===//
// cox_augment_agg: one STRUCT per input row, in frame order
//===--------------------------------------------------------------------===//
const char *const AUGMENT_SCALARS[] = {"linear_predictor", "risk", "martingale_residual", "deviance_residual",
                                       "stratum"};
const char *const AUGMENT_LISTS[] = {"score_residuals", "dfbeta", "dfbetas", "schoenfeld_residuals",
                                     "scaled_schoenfeld_residuals"};

LogicalType CoxAugmentResultType() {
	child_list_t<LogicalType> c;
	for (auto name : AUGMENT_SCALARS) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	for (auto name : AUGMENT_LISTS) {
		c.push_back(MakeChild(name, LogicalType::LIST(LogicalType::DOUBLE)));
	}
	c.push_back(make_pair("is_training", LogicalType::BOOLEAN));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(c)));
}

void CoxAugmentFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<CoxBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (CoxState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (!state.initialized || state.time.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		CoxCall call;
		PrepareCall(bind, state, call);
		const idx_t n = call.data.n_rows;
		AnofoxCoxAugmentResult aug {};
		AnofoxError error;
		if (!anofox_cox_augment(&call.data, bind.options, &aug, &error)) {
			ThrowUnlessDegenerate("cox_augment_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (aug.n_rows != n) {
			anofox_free_cox_augment_result(&aug);
			throw InternalException("cox_augment_agg: returned %llu rows for %llu inputs", (uint64_t)aug.n_rows,
			                        (uint64_t)n);
		}
		const idx_t p = aug.n_features;
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n);
		ListVector::SetListSize(result, list_offset + n);
		FlatDataMut<list_entry_t>(result)[row] = list_entry_t(list_offset, n);
		auto &e = StructVector::GetEntries(ListVector::GetEntry(result));
		const double *scalars[] = {aug.linear_predictor, aug.risk, aug.martingale, aug.deviance, aug.stratum};
		const double *lists[] = {aug.score_mat, aug.dfbeta_mat, aug.dfbetas_mat, aug.schoenfeld_mat,
		                         aug.scaled_schoenfeld_mat};
		for (idx_t k = 0; k < n; k++) {
			// Sorted row k belongs to input (frame) row order[k].
			const idx_t out = list_offset + (call.order.empty() ? k : call.order[k]);
			for (idx_t c = 0; c < 5; c++) {
				SetDouble(StructEntry(e[c]), out, scalars[c][k]);
			}
			for (idx_t c = 0; c < 5; c++) {
				auto &vec = StructEntry(e[5 + c]);
				const double *mat = lists[c];
				// A row without values (not fitted; Schoenfeld: not an event) is NULL, not a list of NULLs.
				bool any = false;
				for (idx_t j = 0; j < p; j++) {
					any = any || std::isfinite(mat[k * p + j]);
				}
				if (!any) {
					FlatVector::SetNull(vec, out, true);
					FlatDataMut<list_entry_t>(vec)[out] = list_entry_t(ListVector::GetListSize(vec), 0);
				} else {
					SetList(vec, out, mat + k * p, p);
				}
			}
			FlatDataMut<bool>(StructEntry(e[10]))[out] = aug.is_training[k];
		}
		anofox_free_cox_augment_result(&aug);
	}
}

unique_ptr<FunctionData> CoxBindCommon(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &arguments, CoxMode mode) {
	auto bind = make_uniq<CoxBindData>();
	bind->mode = mode;
	// (time, x, event, w, options) or (time, x, event, w) with a numeric 4th argument.
	idx_t options_index = DConstants::INVALID_INDEX;
	if (arguments.size() == 5) {
		bind->weighted = true;
		options_index = 4;
	} else if (arguments.size() == 4) {
		if (ReturnTypeOf(*arguments[3]).IsNumeric()) {
			bind->weighted = true;
		} else {
			options_index = 3;
		}
	}
	if (options_index != DConstants::INVALID_INDEX) {
		ParseCoxOptions(EvaluateConstantOptions(context, *arguments[options_index], ModeName(mode)), *bind);
	}
	switch (mode) {
	case CoxMode::ZPH:
		SetReturnTypeOf(function, CoxZphResultType());
		break;
	case CoxMode::AUGMENT:
		SetReturnTypeOf(function, CoxAugmentResultType());
		break;
	default:
		SetReturnTypeOf(function, CoxFitResultType());
	}
	PostHogTelemetry::Instance().RecordFunctionCall(ModeName(mode));
	return std::move(bind);
}

unique_ptr<FunctionData> CoxFitBind(ClientContext &context, AggregateBindFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	return CoxBindCommon(context, function, arguments, CoxMode::FIT);
}
unique_ptr<FunctionData> CoxZphBind(ClientContext &context, AggregateBindFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	return CoxBindCommon(context, function, arguments, CoxMode::ZPH);
}
unique_ptr<FunctionData> CoxAugmentBind(ClientContext &context, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	return CoxBindCommon(context, function, arguments, CoxMode::AUGMENT);
}

void AddCoxOverloads(AggregateFunctionSet &set, const char *name, aggregate_finalize_t finalize,
                     bind_aggregate_function_t bind) {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	vector<vector<LogicalType>> signatures = {
	    {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE},
	    {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY},
	    {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
	};
	for (auto &args : signatures) {
		set.AddFunction(AggregateFunction(name, args, LogicalType::ANY, AggregateFunction::StateSize<CoxState>,
		                                  ANOFOX_AGG_INIT(CoxInitialize),
		                                  ANOFOX_GUARDED_UPDATE(CoxUpdate, CoxDestroy, CoxInitialize), CoxCombine,
		                                  finalize, nullptr, bind, CoxDestroy));
	}
}

void DescribeCox(CreateAggregateFunctionInfo &info, const string &what, const string &example) {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	FunctionDescription d1;
	d1.description = what;
	d1.examples = {example + "(time, [x1, x2], event)"};
	d1.categories = {"survival", "regression"};
	d1.parameter_names = {"time", "x", "event"};
	d1.parameter_types = {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE};
	info.descriptions.push_back(d1);
	FunctionDescription d2 = d1;
	d2.description = what + " With a MAP / STRUCT of options (ties, ridge, robust, offset, strata, entry, cluster, "
	                        "...), or with a numeric case weight as 4th argument.";
	d2.examples = {example + "(time, [x1, x2], event, {'ties': 'breslow'})"};
	d2.parameter_names = {"time", "x", "event", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY};
	info.descriptions.push_back(d2);
	FunctionDescription d3 = d1;
	d3.description = what + " With case weights w (coxph(weights = w)) and options.";
	d3.examples = {example + "(time, [x1, x2], event, w, {'robust': true})"};
	d3.parameter_names = {"time", "x", "event", "weight", "options"};
	d3.parameter_types = {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY};
	info.descriptions.push_back(d3);
}

//===--------------------------------------------------------------------===//
// cox_survival(model, x, times [, stratum]) -> [{time, surv}]
//===--------------------------------------------------------------------===//
LogicalType CoxSurvivalResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("time", LogicalType::DOUBLE));
	c.push_back(make_pair("surv", LogicalType::DOUBLE));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(c)));
}

unique_ptr<FunctionData> CoxSurvivalBind(ClientContext &, ScalarBindFunction &bound_function,
                                         vector<unique_ptr<Expression>> &arguments) {
	auto &type = ReturnTypeOf(*arguments[0]);
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("cox_survival: model must be the STRUCT returned by cox_fit_agg");
	}
	for (auto name : {"coefficients", "baseline"}) {
		bool found = false;
		for (auto &child : StructType::GetChildTypes(type)) {
			found = found || child.first == name;
		}
		if (!found) {
			throw InvalidInputException("cox_survival: model has no '%s' field; pass the STRUCT returned by "
			                            "cox_fit_agg",
			                            string(name));
		}
	}
	SetArgumentTypeOf(bound_function, 0, type);
	return nullptr;
}

vector<double> DoubleList(const Value &v) {
	vector<double> out;
	for (auto &child : ListValue::GetChildren(v)) {
		out.push_back(child.IsNull() ? NAN : child.GetValue<double>());
	}
	return out;
}

void CoxSurvivalFunctionImpl(DataChunk &args, ExpressionState &, Vector &result) {
	const idx_t count = args.size();
	const bool with_stratum = args.ColumnCount() >= 4;
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	auto list_data = FlatDataMut<list_entry_t>(result);
	auto &mask = FlatValidityMut(result);

	for (idx_t i = 0; i < count; i++) {
		auto model = args.data[0].GetValue(i);
		auto x_value = args.data[1].GetValue(i);
		auto times_value = args.data[2].GetValue(i);
		if (model.IsNull() || x_value.IsNull() || times_value.IsNull()) {
			list_data[i] = list_entry_t(0, 0);
			mask.SetInvalid(i);
			continue;
		}
		const Value *coef_v = nullptr, *base_v = nullptr;
		auto &fields = StructValue::GetChildren(model);
		for (idx_t f = 0; f < fields.size(); f++) {
			const string name = NameString(StructType::GetChildName(model.type(), f));
			if (name == "coefficients") {
				coef_v = &fields[f];
			} else if (name == "baseline") {
				base_v = &fields[f];
			}
		}
		if (coef_v->IsNull() || base_v->IsNull()) {
			mask.SetInvalid(i);
			continue;
		}
		auto coefficients = DoubleList(*coef_v);
		auto x = DoubleList(x_value);
		auto times = DoubleList(times_value);
		if (x.size() != coefficients.size()) {
			throw InvalidInputException("cox_survival: x has %llu values but the model has %llu coefficients",
			                            (unsigned long long)x.size(), (unsigned long long)coefficients.size());
		}
		// The baseline points of the requested stratum (all of them for an unstratified model).
		vector<double> base_time, base_cumhaz;
		bool stratified = false, stratum_found = false;
		Value stratum_value = with_stratum ? args.data[3].GetValue(i) : Value();
		double stratum = stratum_value.IsNull() ? NAN : stratum_value.GetValue<double>();
		for (auto &point : ListValue::GetChildren(*base_v)) {
			auto &p = StructValue::GetChildren(point);
			bool point_stratified = !p[0].IsNull();
			stratified = stratified || point_stratified;
			if (point_stratified && !(p[0].GetValue<double>() == stratum)) {
				continue;
			}
			stratum_found = true;
			base_time.push_back(p[1].IsNull() ? NAN : p[1].GetValue<double>());
			base_cumhaz.push_back(p[2].IsNull() ? NAN : p[2].GetValue<double>());
		}
		if (stratified && std::isnan(stratum)) {
			throw InvalidInputException("cox_survival: the model is stratified; pass the stratum as 4th argument");
		}
		if (stratified && !stratum_found) {
			throw InvalidInputException("cox_survival: the model has no stratum %g", stratum);
		}
		vector<double> surv(times.size());
		anofox_cox_survival(coefficients.data(), coefficients.size(), base_time.data(), base_cumhaz.data(),
		                    base_time.size(), x.data(), times.data(), times.size(), surv.data());

		auto n = static_cast<idx_t>(times.size());
		auto offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, offset + n);
		auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
		for (idx_t k = 0; k < n; k++) {
			SetDouble(StructEntry(entries[0]), offset + k, times[k]);
			SetDouble(StructEntry(entries[1]), offset + k, surv[k]);
		}
		ListVector::SetListSize(result, offset + n);
		list_data[i] = list_entry_t(offset, n);
	}
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding asserts this in debug builds).
void CoxSurvivalFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	CoxSurvivalFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterCoxFunctions(ExtensionLoader &loader) {
	{
		AggregateFunctionSet set("cox_fit_agg");
		AddCoxOverloads(set, "cox_fit_agg", ANOFOX_GUARDED_FINALIZE(CoxFitFinalize, CoxDestroy, CoxInitialize),
		                ANOFOX_AGG_BIND(CoxFitBind));
		CreateAggregateFunctionInfo info(set);
		DescribeCox(info,
		            "Fits a Cox proportional hazards model (survival::coxph; Efron or Breslow ties) to right-censored "
		            "or counting-process survival data (event = 1 observed, 0 censored): log hazard ratios, hazard "
		            "ratios with Wald intervals, likelihood-ratio / Wald / score tests, concordance and the baseline "
		            "cumulative hazard.",
		            "cox_fit_agg");
		loader.RegisterFunction(info);
	}
	{
		AggregateFunctionSet set("cox_zph_agg");
		AddCoxOverloads(set, "cox_zph_agg", ANOFOX_GUARDED_FINALIZE(CoxZphFinalize, CoxDestroy, CoxInitialize),
		                ANOFOX_AGG_BIND(CoxZphBind));
		CreateAggregateFunctionInfo info(set);
		DescribeCox(info,
		            "Grambsch-Therneau test of proportional hazards of a Cox model (survival::cox.zph): one "
		            "{term, statistic, df, p_value} per covariate and a GLOBAL row; option transform: km (default), "
		            "rank, identity, log.",
		            "cox_zph_agg");
		loader.RegisterFunction(info);
	}
	{
		AggregateFunctionSet set("cox_augment_agg");
		AddCoxOverloads(set, "cox_augment_agg",
		                ANOFOX_GUARDED_FINALIZE(CoxAugmentFinalize, CoxDestroy, CoxInitialize),
		                ANOFOX_AGG_BIND(CoxAugmentBind));
		CreateAggregateFunctionInfo info(set);
		DescribeCox(info,
		            "Window aggregate: fits the same Cox model as cox_fit_agg over the frame and returns one STRUCT "
		            "per input row (in frame order): linear_predictor, risk, martingale / deviance residuals, score, "
		            "dfbeta, dfbetas and (event rows) Schoenfeld residuals, stratum, is_training. Use it over a "
		            "whole partition, as cox_augment_by does.",
		            "cox_augment_agg");
		loader.RegisterFunction(info);
	}

	ScalarFunctionSet survival_set("cox_survival");
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	for (bool with_stratum : {false, true}) {
		vector<LogicalType> args = {LogicalType::ANY, dlist, dlist};
		if (with_stratum) {
			args.push_back(LogicalType::DOUBLE);
		}
		ScalarFunction f("cox_survival", args, CoxSurvivalResultType(), CoxSurvivalFunction,
		                 ANOFOX_SCALAR_BIND(CoxSurvivalBind));
		MarkFallible(f);
		survival_set.AddFunction(f);
	}
	CreateScalarFunctionInfo info(survival_set);
	FunctionDescription d;
	d.description = "Predicted survival curve S(t | x) of a fitted Cox model (the STRUCT returned by cox_fit_agg) for "
	                "the covariate row x at each of times (survfit(fit, newdata) point estimate, from the stored "
	                "baseline cumulative hazard): one {time, surv} per time.";
	d.examples = {"cox_survival(m, [60.0, 1.0], [100, 365, 730])"};
	d.categories = {"survival"};
	d.parameter_names = {"model", "x", "times"};
	d.parameter_types = {LogicalType::ANY, dlist, dlist};
	info.descriptions.push_back(d);
	FunctionDescription d2 = d;
	d2.description = d.description + " For a stratified model, stratum picks the stratum's baseline.";
	d2.examples = {"cox_survival(m, [60.0], [100, 365], 2)"};
	d2.parameter_names = {"model", "x", "times", "stratum"};
	d2.parameter_types = {LogicalType::ANY, dlist, dlist, LogicalType::DOUBLE};
	info.descriptions.push_back(d2);
	loader.RegisterFunction(info);
}

} // namespace duckdb
