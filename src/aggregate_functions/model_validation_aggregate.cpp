// Resampling model validation and calibration (#318), as rms::validate,
// rms::calibrate, rms::calibrate.cph and rms::bootcov:
//
//   validate_agg(y DOUBLE, x DOUBLE[] [, options])
//     -> LIST of {index_name, index_orig, training, test, optimism, index_corrected, n, n_failed}
//   calibrate_agg(y DOUBLE, x DOUBLE[] [, options])
//     -> LIST of {predy, calibrated_orig, calibrated_corrected, optimism, n, mean_abs_error,
//                 mean_squared_error, quantile90_abs_error}
//   calibrate_cph_agg(time DOUBLE, x DOUBLE[], event DOUBLE, options)
//     -> LIST of {group_index, cut_low, cut_high, mean_predicted, km, km_corrected, std_err,
//                 index_orig, training, test, optimism, index_corrected, n}
//   bootcov_agg(y DOUBLE, x DOUBLE[] [, cluster ANY], [options])
//     -> LIST of {term, estimate, std_error, conf_low, conf_high, conf_level, boot_mean, vcov,
//                 n_boot, n_failed, model}
//
// and the table macros validate_by, calibrate_by (contract `curve`), calibrate_cph_by
// and bootcov_by (contract `terms`).
//
// The models are a linear model (rms::ols, the default) or a binary logistic model
// (rms::lrm, model 'logistic'), with an intercept, of y on the columns of x; a Cox
// model for calibrate_cph_agg. The resampling engine, the indexes, the curves, the
// covariance and the seeded draws are upstream anofox-regression's validation module,
// reached through anofox_validate / anofox_calibrate / anofox_calibrate_cph /
// anofox_bootcov; this file collects the rows of a group, parses the options and lays
// the result out.
//
// The resamples are drawn by row position, so every finalizer first sorts the rows
// canonically: a seed then gives the same result whatever the thread count or the
// order in which the rows arrive.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "aggregate_finalize_guard.hpp"
#include "anofox_statistics_extension.hpp"
#include "canonical_order.hpp"
#include "error_dispatch.hpp"
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

namespace {

enum class ValidationKind : uint8_t { VALIDATE, CALIBRATE, CALIBRATE_CPH, BOOTCOV };

const char *KindName(ValidationKind kind) {
	switch (kind) {
	case ValidationKind::CALIBRATE:
		return "calibrate_agg";
	case ValidationKind::CALIBRATE_CPH:
		return "calibrate_cph_agg";
	case ValidationKind::BOOTCOV:
		return "bootcov_agg";
	default:
		return "validate_agg";
	}
}

//! How a bootcov_agg cluster value becomes the DOUBLE that travels with its row.
enum class ClusterMode : uint8_t { NONE, VALUE, HASH };

struct ValidationBindData : public FunctionData {
	ValidationKind kind = ValidationKind::VALIDATE;
	ClusterMode cluster = ClusterMode::NONE;
	AnofoxResampleOptions resample {0, 0, 40, 42};
	bool has_predy = false;
	vector<double> predy;
	// calibrate_cph_agg
	double u = std::numeric_limits<double>::quiet_NaN();
	uint64_t m = 50;
	bool has_cuts = false;
	vector<double> cuts;
	int32_t ties = 0;
	uint32_t max_iterations = 20;
	double tolerance = 1e-9;
	// bootcov_agg
	double conf_level = 0.95;
	vector<string> names;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ValidationBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ValidationBindData>();
		auto same = [](double a, double b) {
			return a == b || (std::isnan(a) && std::isnan(b));
		};
		return kind == o.kind && cluster == o.cluster && resample.model == o.resample.model &&
		       resample.method == o.resample.method && resample.b == o.resample.b &&
		       resample.seed == o.resample.seed && has_predy == o.has_predy && predy == o.predy && same(u, o.u) &&
		       m == o.m && has_cuts == o.has_cuts && cuts == o.cuts && ties == o.ties &&
		       max_iterations == o.max_iterations && tolerance == o.tolerance && conf_level == o.conf_level &&
		       names == o.names;
	}
};

const char *ModelName(const ValidationBindData &bind) {
	if (bind.kind == ValidationKind::CALIBRATE_CPH) {
		return "cox";
	}
	return bind.resample.model == 1 ? "logistic" : "ols";
}

//===--------------------------------------------------------------------===//
// Options
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

//! A list option: a LIST value, or (from a MAP(VARCHAR, VARCHAR)) its text '[a, b, ...]'.
vector<Value> ListOption(const string &fn, const string &key, const Value &v, const LogicalType &child) {
	Value list = v;
	if (v.type().id() != LogicalTypeId::LIST) {
		try {
			list = v.DefaultCastAs(LogicalType::LIST(child));
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a list, got '%s'", fn, key, v.ToString());
		}
	}
	return ListValue::GetChildren(list);
}

void ParseOptions(const Value &value, ValidationBindData &bind) {
	const string fn = KindName(bind.kind);
	const bool cph = bind.kind == ValidationKind::CALIBRATE_CPH;
	const bool boot = bind.kind == ValidationKind::BOOTCOV;
	auto as_double = [&](const string &key, const Value &v) {
		try {
			return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
		}
	};
	auto as_count = [&](const string &key, const Value &v, double lo, double hi) {
		double d = as_double(key, v);
		if (!(d >= lo && d <= hi) || d != std::floor(d)) {
			throw InvalidInputException("%s: option '%s' must be an integer between %g and %g, got '%s'", fn, key, lo,
			                            hi, v.ToString());
		}
		return d;
	};
	auto as_doubles = [&](const string &key, const Value &v) {
		vector<double> out;
		for (auto &child : ListOption(fn, key, v, LogicalType::DOUBLE)) {
			if (child.IsNull()) {
				throw InvalidInputException("%s: option '%s' must not hold NULLs", fn, key);
			}
			double d = as_double(key, child);
			if (!std::isfinite(d)) {
				throw InvalidInputException("%s: option '%s' must hold finite numbers", fn, key);
			}
			out.push_back(d);
		}
		if (out.empty()) {
			throw InvalidInputException("%s: option '%s' must not be empty", fn, key);
		}
		return out;
	};
	VisitOptions(value, [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "model" && !cph) {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "ols" || s == "linear") {
				bind.resample.model = 0;
			} else if (s == "logistic" || s == "lrm") {
				bind.resample.model = 1;
			} else {
				throw InvalidInputException("%s: unknown model '%s'. Expected 'ols' or 'logistic'.", fn, v.ToString());
			}
		} else if (key == "method" && !boot) {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "boot" || s == "bootstrap") {
				bind.resample.method = 0;
			} else if (s == "crossvalidation" || s == "cv") {
				bind.resample.method = 1;
			} else if (s == ".632" || s == "632" || s == "point632") {
				bind.resample.method = 2;
			} else {
				throw InvalidInputException(
				    "%s: unknown method '%s'. Expected 'boot', 'crossvalidation' or '.632'.", fn, v.ToString());
			}
		} else if (key == "b" || key == "n_boot") {
			bind.resample.b = (uint32_t)as_count(key, v, 1, 1e6);
		} else if (key == "seed" || key == "random_state") {
			bind.resample.seed = (uint64_t)as_count(key, v, 0, 9007199254740992.0);
		} else if (key == "predy" && bind.kind == ValidationKind::CALIBRATE) {
			bind.predy = as_doubles(key, v);
			bind.has_predy = true;
		} else if (key == "u" && cph) {
			bind.u = as_double(key, v);
			if (!(bind.u > 0.0) || !std::isfinite(bind.u)) {
				throw InvalidInputException("%s: u must be a positive time, got '%s'", fn, v.ToString());
			}
		} else if (key == "m" && cph) {
			bind.m = (uint64_t)as_count(key, v, 1, 1e9);
		} else if (key == "cuts" && cph) {
			bind.cuts = as_doubles(key, v);
			if (bind.cuts.size() < 2) {
				throw InvalidInputException("%s: cuts needs at least two boundaries", fn);
			}
			std::sort(bind.cuts.begin(), bind.cuts.end());
			bind.has_cuts = true;
		} else if ((key == "ties") && cph) {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "efron") {
				bind.ties = 0;
			} else if (s == "breslow") {
				bind.ties = 1;
			} else {
				throw InvalidInputException("%s: unknown ties '%s'. Expected 'efron' or 'breslow'.", fn, v.ToString());
			}
		} else if ((key == "max_iterations" || key == "max_iter") && cph) {
			bind.max_iterations = (uint32_t)as_count(key, v, 1, 1e6);
		} else if ((key == "tolerance" || key == "eps") && cph) {
			bind.tolerance = as_double(key, v);
			if (!(bind.tolerance > 0.0) || !std::isfinite(bind.tolerance)) {
				throw InvalidInputException("%s: tolerance must be positive, got '%s'", fn, v.ToString());
			}
		} else if (key == "conf_level" || key == "confidence_level") {
			if (!boot) {
				throw InvalidInputException("%s: unknown option '%s'", fn, key);
			}
			bind.conf_level = as_double(key, v);
			if (!(bind.conf_level > 0.0 && bind.conf_level < 1.0)) {
				throw InvalidInputException("%s: conf_level must be in (0, 1), got '%s'", fn, v.ToString());
			}
		} else if (key == "feature_names" && boot) {
			bind.names.clear();
			for (auto &child : ListOption(fn, key, v, LogicalType::VARCHAR)) {
				bind.names.push_back(child.IsNull() ? string() : child.ToString());
			}
		} else {
			const char *supported = "";
			switch (bind.kind) {
			case ValidationKind::VALIDATE:
				supported = "model, method, B, seed";
				break;
			case ValidationKind::CALIBRATE:
				supported = "model, method, B, seed, predy";
				break;
			case ValidationKind::CALIBRATE_CPH:
				supported = "u, m, cuts, method, B, seed, ties, max_iterations, tolerance";
				break;
			case ValidationKind::BOOTCOV:
				supported = "model, B, seed, conf_level, feature_names";
				break;
			}
			throw InvalidInputException("%s: unknown option '%s'. Supported options: %s", fn, key, supported);
		}
	});
	if (cph && std::isnan(bind.u)) {
		throw InvalidInputException("%s: option 'u' (the time at which survival is calibrated) is required", fn);
	}
	if (bind.resample.method == 1 && bind.resample.b < 2) {
		throw InvalidInputException("%s: cross-validation needs B >= 2 groups", fn);
	}
	if (boot && bind.resample.b < 2) {
		throw InvalidInputException("%s: B must be at least 2", fn);
	}
}

//===--------------------------------------------------------------------===//
// Rows
//===--------------------------------------------------------------------===//

//! The cluster column as DOUBLEs (NULL stays NULL): numbers by value, anything else
//! by the top 53 bits of its hash (exact in a DOUBLE).
void ClusterToDouble(Vector &input, Vector &keys, idx_t count, ClusterMode mode) {
	auto out = FlatDataMut<double>(keys);
	auto &validity = FlatValidityMut(keys);
	UnifiedVectorFormat in;
	input.ToUnifiedFormat(count, in);
	if (mode == ClusterMode::HASH) {
		Vector hashes(LogicalType::HASH, count);
		VectorOperations::Hash(input, hashes, count);
		UnifiedVectorFormat hf;
		hashes.ToUnifiedFormat(count, hf);
		auto h = UnifiedVectorFormat::GetData<hash_t>(hf);
		for (idx_t i = 0; i < count; i++) {
			if (!in.validity.RowIsValid(in.sel->get_index(i))) {
				validity.SetInvalid(i);
				continue;
			}
			out[i] = static_cast<double>(h[hf.sel->get_index(i)] >> 11);
		}
		return;
	}
	Vector cast(LogicalType::DOUBLE, count);
	VectorOperations::DefaultCast(input, cast, count);
	UnifiedVectorFormat cf;
	cast.ToUnifiedFormat(count, cf);
	auto values = UnifiedVectorFormat::GetData<double>(cf);
	for (idx_t i = 0; i < count; i++) {
		auto idx = cf.sel->get_index(i);
		if (!cf.validity.RowIsValid(idx)) {
			validity.SetInvalid(i);
			continue;
		}
		out[i] = values[idx];
	}
}

void ValidationUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                      idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<ValidationBindData>();
	if (bind.kind == ValidationKind::CALIBRATE_CPH) {
		// event travels with its row; a NULL event skips the row.
		XyRowsUpdateImpl(inputs, count, state_vector, &inputs[2]);
		return;
	}
	if (bind.cluster != ClusterMode::NONE) {
		Vector keys(LogicalType::DOUBLE, count);
		ClusterToDouble(inputs[2], keys, count, bind.cluster);
		XyRowsUpdateImpl(inputs, count, state_vector, &keys);
		return;
	}
	XyRowsUpdateImpl(inputs, count, state_vector);
}

//! The rows of one call: finite rows only, in canonical order, x column-major.
struct ValidationRows {
	vector<double> y, extra, x;
	idx_t n = 0, p = 0;
};

ValidationRows PrepareRows(XyRowsState &state, bool with_extra) {
	ValidationRows rows;
	rows.p = state.n_features;
	const idx_t n_in = state.y_values.size();
	// Rows with a non-finite value cannot be fitted: they are left out, like NULLs.
	vector<double> y;
	vector<double> extra;
	vector<vector<double>> x(rows.p);
	for (idx_t r = 0; r < n_in; r++) {
		bool finite = std::isfinite(state.y_values[r]) && (!with_extra || std::isfinite(state.extra[r]));
		for (idx_t j = 0; finite && j < rows.p; j++) {
			finite = std::isfinite(state.x_columns[j][r]);
		}
		if (!finite) {
			continue;
		}
		y.push_back(state.y_values[r]);
		if (with_extra) {
			extra.push_back(state.extra[r]);
		}
		for (idx_t j = 0; j < rows.p; j++) {
			x[j].push_back(state.x_columns[j][r]);
		}
	}
	CanonicalRows canonical;
	canonical.Keys(x).Key(y);
	if (with_extra) {
		canonical.Key(extra);
	}
	canonical.Sort();
	rows.n = y.size();
	rows.y = std::move(y);
	rows.extra = std::move(extra);
	rows.x.reserve(rows.n * rows.p);
	for (auto &col : x) {
		rows.x.insert(rows.x.end(), col.begin(), col.end());
	}
	return rows;
}

//! Reserve `n` rows of a LIST(STRUCT) result for `row` and return the first child index.
idx_t ReserveRows(Vector &result, idx_t row, idx_t n) {
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + n);
	ListVector::SetListSize(result, offset + n);
	FlatDataMut<list_entry_t>(result)[row] = {offset, n};
	return offset;
}

void SetBigint(Vector &vec, idx_t row, uint64_t value) {
	FlatDataMut<int64_t>(vec)[row] = static_cast<int64_t>(value);
}

//===--------------------------------------------------------------------===//
// Result types
//===--------------------------------------------------------------------===//
LogicalType ValidateRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("index_name", LogicalType::VARCHAR));
	for (auto name : {"index_orig", "training", "test", "optimism", "index_corrected"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	c.push_back(make_pair("n", LogicalType::BIGINT));
	c.push_back(make_pair("n_failed", LogicalType::BIGINT));
	c.push_back(make_pair("model", LogicalType::VARCHAR));
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType CalibrateRowType() {
	child_list_t<LogicalType> c;
	for (auto name : {"predy", "calibrated_orig", "calibrated_corrected", "optimism"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	c.push_back(make_pair("n", LogicalType::BIGINT));
	for (auto name : {"mean_abs_error", "mean_squared_error", "quantile90_abs_error"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	c.push_back(make_pair("model", LogicalType::VARCHAR));
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType CalibrateCphRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("group_index", LogicalType::BIGINT));
	for (auto name : {"cut_low", "cut_high", "mean_predicted", "km", "km_corrected", "std_err", "index_orig",
	                  "training", "test", "optimism", "index_corrected"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	c.push_back(make_pair("n", LogicalType::BIGINT));
	c.push_back(make_pair("u", LogicalType::DOUBLE));
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

LogicalType BootcovRowType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("term", LogicalType::VARCHAR));
	for (auto name : {"estimate", "std_error", "conf_low", "conf_high", "conf_level", "boot_mean"}) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	c.push_back(make_pair("vcov", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("n_boot", LogicalType::BIGINT));
	c.push_back(make_pair("n_failed", LogicalType::BIGINT));
	c.push_back(make_pair("model", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

const char *MethodName(int32_t method) {
	switch (method) {
	case 1:
		return "crossvalidation";
	case 2:
		return ".632";
	default:
		return "boot";
	}
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
bool WriteValidate(const ValidationBindData &bind, ValidationRows &rows, Vector &result, idx_t row,
                   AnofoxError &error) {
	AnofoxValidationResult t {};
	if (!anofox_validate(rows.y.data(), rows.x.data(), rows.n, rows.p, bind.resample, &t, &error)) {
		return false;
	}
	idx_t out = ReserveRows(result, row, t.n_indexes);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t k = 0; k < t.n_indexes; k++, out++) {
		StructEntry(f[0]).SetValue(out, Value(string(t.names + k * ANOFOX_VALIDATION_NAME_LEN)));
		XyRowsSetDouble(StructEntry(f[1]), out, t.index_orig[k]);
		XyRowsSetDouble(StructEntry(f[2]), out, t.training[k]);
		XyRowsSetDouble(StructEntry(f[3]), out, t.test[k]);
		XyRowsSetDouble(StructEntry(f[4]), out, t.optimism[k]);
		XyRowsSetDouble(StructEntry(f[5]), out, t.corrected[k]);
		SetBigint(StructEntry(f[6]), out, t.n[k]);
		SetBigint(StructEntry(f[7]), out, t.failed);
		StructEntry(f[8]).SetValue(out, Value(ModelName(bind)));
		StructEntry(f[9]).SetValue(out, Value(MethodName(bind.resample.method)));
	}
	anofox_free_validation_result(&t);
	return true;
}

bool WriteCalibrate(const ValidationBindData &bind, ValidationRows &rows, Vector &result, idx_t row,
                    AnofoxError &error) {
	AnofoxCalibrationResult c {};
	if (!anofox_calibrate(rows.y.data(), rows.x.data(), rows.n, rows.p, bind.resample,
	                      bind.has_predy ? bind.predy.data() : nullptr, bind.predy.size(), &c, &error)) {
		return false;
	}
	idx_t out = ReserveRows(result, row, c.len);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t k = 0; k < c.len; k++, out++) {
		XyRowsSetDouble(StructEntry(f[0]), out, c.predy[k]);
		XyRowsSetDouble(StructEntry(f[1]), out, c.calibrated_orig[k]);
		XyRowsSetDouble(StructEntry(f[2]), out, c.calibrated_corrected[k]);
		XyRowsSetDouble(StructEntry(f[3]), out, c.optimism[k]);
		SetBigint(StructEntry(f[4]), out, c.n[k]);
		XyRowsSetDouble(StructEntry(f[5]), out, c.mean_abs_error);
		XyRowsSetDouble(StructEntry(f[6]), out, c.mean_squared_error);
		XyRowsSetDouble(StructEntry(f[7]), out, c.quantile90_abs_error);
		StructEntry(f[8]).SetValue(out, Value(ModelName(bind)));
		StructEntry(f[9]).SetValue(out, Value(MethodName(bind.resample.method)));
	}
	anofox_free_calibration_result(&c);
	return true;
}

bool WriteCalibrateCph(const ValidationBindData &bind, ValidationRows &rows, Vector &result, idx_t row,
                       AnofoxError &error) {
	AnofoxCalibrateCphOptions cph {};
	cph.u = bind.u;
	cph.m = bind.m;
	cph.cuts = bind.has_cuts ? bind.cuts.data() : nullptr;
	cph.cuts_len = bind.cuts.size();
	cph.ties = bind.ties;
	cph.max_iterations = bind.max_iterations;
	cph.tolerance = bind.tolerance;
	AnofoxSurvivalCalibrationResult c {};
	if (!anofox_calibrate_cph(rows.y.data(), rows.x.data(), rows.extra.data(), rows.n, rows.p, bind.resample, cph,
	                          &c, &error)) {
		return false;
	}
	idx_t out = ReserveRows(result, row, c.n_groups);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t k = 0; k < c.n_groups; k++, out++) {
		SetBigint(StructEntry(f[0]), out, k + 1);
		const double values[] = {c.cuts[k],    c.cuts[k + 1],  c.mean_predicted[k], c.km[k],
		                         c.km_corrected[k], c.std_err[k], c.index_orig[k],     c.training[k],
		                         c.test[k],    c.optimism[k],  c.corrected[k]};
		for (idx_t v = 0; v < 11; v++) {
			XyRowsSetDouble(StructEntry(f[1 + v]), out, values[v]);
		}
		SetBigint(StructEntry(f[12]), out, c.n[k]);
		XyRowsSetDouble(StructEntry(f[13]), out, bind.u);
		StructEntry(f[14]).SetValue(out, Value(MethodName(bind.resample.method)));
	}
	anofox_free_survival_calibration_result(&c);
	return true;
}

bool WriteBootcov(const ValidationBindData &bind, ValidationRows &rows, Vector &result, idx_t row,
                  AnofoxError &error) {
	if (!bind.names.empty() && bind.names.size() != rows.p) {
		throw InvalidInputException("bootcov_agg: feature_names has %llu entries but x has %llu features",
		                            (unsigned long long)bind.names.size(), (unsigned long long)rows.p);
	}
	// Dense cluster codes in the order of the cluster values.
	vector<uint64_t> codes;
	if (bind.cluster != ClusterMode::NONE) {
		vector<double> levels = rows.extra;
		std::sort(levels.begin(), levels.end());
		levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
		codes.reserve(rows.n);
		for (double v : rows.extra) {
			codes.push_back(static_cast<uint64_t>(std::lower_bound(levels.begin(), levels.end(), v) - levels.begin()));
		}
	}
	AnofoxBootcovResult b {};
	if (!anofox_bootcov(rows.y.data(), rows.x.data(), codes.empty() ? nullptr : codes.data(), rows.n, rows.p,
	                    bind.resample, bind.conf_level, &b, &error)) {
		return false;
	}
	const idx_t k = b.n_coefficients;
	idx_t out = ReserveRows(result, row, k);
	auto &f = StructVector::GetEntries(ListVector::GetEntry(result));
	for (idx_t j = 0; j < k; j++, out++) {
		string term = j == 0 ? "(Intercept)"
		                     : (bind.names.empty() ? "x" + std::to_string(j) : bind.names[j - 1]);
		StructEntry(f[0]).SetValue(out, Value(term));
		XyRowsSetDouble(StructEntry(f[1]), out, b.estimate[j]);
		XyRowsSetDouble(StructEntry(f[2]), out, b.std_error[j]);
		XyRowsSetDouble(StructEntry(f[3]), out, b.conf_low[j]);
		XyRowsSetDouble(StructEntry(f[4]), out, b.conf_high[j]);
		XyRowsSetDouble(StructEntry(f[5]), out, bind.conf_level);
		XyRowsSetDouble(StructEntry(f[6]), out, b.boot_mean[j]);
		XyRowsSetDoubleList(StructEntry(f[7]), out, b.vcov + j * k, k);
		SetBigint(StructEntry(f[8]), out, b.n_boot);
		SetBigint(StructEntry(f[9]), out, b.failed);
		StructEntry(f[10]).SetValue(out, Value(ModelName(bind)));
	}
	anofox_free_bootcov_result(&b);
	return true;
}

void ValidationFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ValidationBindData>();
	const char *name = KindName(bind.kind);
	XyRowsStatePointers states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		const idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		const bool with_extra = bind.kind == ValidationKind::CALIBRATE_CPH || bind.cluster != ClusterMode::NONE;
		auto rows = PrepareRows(state, with_extra);
		if (rows.n == 0) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxError error;
		bool ok = false;
		switch (bind.kind) {
		case ValidationKind::VALIDATE:
			ok = WriteValidate(bind, rows, result, row, error);
			break;
		case ValidationKind::CALIBRATE:
			ok = WriteCalibrate(bind, rows, result, row, error);
			break;
		case ValidationKind::CALIBRATE_CPH:
			ok = WriteCalibrateCph(bind, rows, result, row, error);
			break;
		case ValidationKind::BOOTCOV:
			ok = WriteBootcov(bind, rows, result, row, error);
			break;
		}
		if (!ok) {
			ThrowUnlessDegenerate(name, error);
			FlatVector::SetNull(result, row, true);
		}
	}
}

//===--------------------------------------------------------------------===//
// Bind and registration
//===--------------------------------------------------------------------===//
unique_ptr<FunctionData> ValidationBindImpl(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments, ValidationKind kind) {
	auto bind = make_uniq<ValidationBindData>();
	bind->kind = kind;
	const string name = KindName(kind);
	idx_t options_idx = 2;
	if (kind == ValidationKind::CALIBRATE_CPH) {
		options_idx = 3;
	} else if (kind == ValidationKind::BOOTCOV) {
		bind->resample.b = 200; // rms::bootcov's default
		if (arguments.size() == 4) {
			options_idx = 3;
			auto &type = ReturnTypeOf(*arguments[2]);
			bind->cluster = type.id() == LogicalTypeId::SQLNULL ? ClusterMode::NONE
			                : type.IsNumeric()                    ? ClusterMode::VALUE
			                                                      : ClusterMode::HASH;
		}
	}
	if (arguments.size() > options_idx) {
		ParseOptions(EvaluateConstantOptions(context, *arguments[options_idx], name), *bind);
	} else if (kind == ValidationKind::CALIBRATE_CPH) {
		ParseOptions(Value(), *bind);
	}
	switch (kind) {
	case ValidationKind::CALIBRATE:
		SetReturnTypeOf(function, LogicalType::LIST(CalibrateRowType()));
		break;
	case ValidationKind::CALIBRATE_CPH:
		SetReturnTypeOf(function, LogicalType::LIST(CalibrateCphRowType()));
		break;
	case ValidationKind::BOOTCOV:
		SetReturnTypeOf(function, LogicalType::LIST(BootcovRowType()));
		break;
	default:
		SetReturnTypeOf(function, LogicalType::LIST(ValidateRowType()));
	}
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

unique_ptr<FunctionData> ValidateBind(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &arguments) {
	return ValidationBindImpl(context, function, arguments, ValidationKind::VALIDATE);
}
unique_ptr<FunctionData> CalibrateBind(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	return ValidationBindImpl(context, function, arguments, ValidationKind::CALIBRATE);
}
unique_ptr<FunctionData> CalibrateCphBind(ClientContext &context, AggregateBindFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	return ValidationBindImpl(context, function, arguments, ValidationKind::CALIBRATE_CPH);
}
unique_ptr<FunctionData> BootcovBind(ClientContext &context, AggregateBindFunction &function,
                                     vector<unique_ptr<Expression>> &arguments) {
	return ValidationBindImpl(context, function, arguments, ValidationKind::BOOTCOV);
}

AggregateFunction MakeAggregate(const char *name, vector<LogicalType> args, bind_aggregate_function_t bind) {
	return AggregateFunction(name, std::move(args), LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
	                         ANOFOX_AGG_INIT(XyRowsInitialize),
	                         ANOFOX_GUARDED_UPDATE(ValidationUpdate, XyRowsDestroy, XyRowsInitialize), XyRowsCombine,
	                         ANOFOX_GUARDED_FINALIZE(ValidationFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
	                         bind, XyRowsDestroy);
}

FunctionDescription Describe(const string &description, const string &example, vector<string> names,
                             vector<LogicalType> types, vector<string> categories) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = std::move(categories);
	d.parameter_names = std::move(names);
	d.parameter_types = std::move(types);
	return d;
}

// Table macros: one row per element of the aggregate's list, per group.
const char *const VALIDATE_BY_SQL =
    "SELECT group_col, (_t).index_name AS index_name, (_t).index_orig AS index_orig, (_t).training AS training, "
    "(_t).test AS test, (_t).optimism AS optimism, (_t).index_corrected AS index_corrected, (_t).n AS n, "
    "(_t).n_failed AS n_failed, (_t).model AS model, (_t).method AS method "
    "FROM (SELECT group_col, unnest(range(len(_l))) AS _i, unnest(_l) AS _t "
    "FROM (SELECT group_col, validate_agg(y_col::DOUBLE, x_cols, options) AS _l "
    "FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";

// The integration contract's `curve` schema (as lowess_by), then the calibrate columns.
const char *const CALIBRATE_BY_SQL =
    "SELECT group_col, group_col::VARCHAR AS id, 'calibrate' AS model_id, (_t).model AS model_name, "
    "'calibrate' AS curve_type, NULL::VARCHAR AS series, (_t).predy AS x, (_t).calibrated_corrected AS y, "
    "NULL::DOUBLE AS y_low, NULL::DOUBLE AS y_high, NULL::DOUBLE AS threshold, NULL::DOUBLE AS weight, "
    "NULL::BIGINT AS n_censor, (_t).predy AS predy, (_t).calibrated_orig AS calibrated_orig, "
    "(_t).calibrated_corrected AS calibrated_corrected, (_t).optimism AS optimism, (_t).n AS n, "
    "(_t).mean_abs_error AS mean_abs_error, (_t).mean_squared_error AS mean_squared_error, "
    "(_t).quantile90_abs_error AS quantile90_abs_error, (_t).method AS method "
    "FROM (SELECT group_col, unnest(range(len(_l))) AS _i, unnest(_l) AS _t "
    "FROM (SELECT group_col, calibrate_agg(y_col::DOUBLE, x_cols, options) AS _l "
    "FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";

const char *const CALIBRATE_CPH_BY_SQL =
    "SELECT group_col, (_t).group_index AS group_index, (_t).cut_low AS cut_low, (_t).cut_high AS cut_high, "
    "(_t).mean_predicted AS mean_predicted, (_t).km AS km, (_t).km_corrected AS km_corrected, "
    "(_t).std_err AS std_err, (_t).index_orig AS index_orig, (_t).training AS training, (_t).test AS test, "
    "(_t).optimism AS optimism, (_t).index_corrected AS index_corrected, (_t).n AS n, (_t).u AS u, "
    "(_t).method AS method "
    "FROM (SELECT group_col, unnest(range(len(_l))) AS _i, unnest(_l) AS _t "
    "FROM (SELECT group_col, calibrate_cph_agg(time_col::DOUBLE, x_cols, event_col::DOUBLE, options) AS _l "
    "FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";

// The integration contract's `terms` schema (as the *_tidy_by macros), then the bootstrap columns.
// names := [...] relabels the default x1, x2, ... terms.
const char *const BOOTCOV_BY_SQL =
    "SELECT group_col, group_col::VARCHAR AS id, 'bootcov' AS model_id, (_t).model AS model_name, "
    "CASE WHEN names IS NULL OR (_t).term = '(Intercept)' THEN (_t).term "
    "ELSE COALESCE((names::VARCHAR[])[TRY_CAST(substr((_t).term, 2) AS BIGINT)], (_t).term) END AS term, "
    "(_t).estimate AS estimate, (_t).std_error AS std_error, NULL::DOUBLE AS statistic, NULL::DOUBLE AS p_value, "
    "(_t).conf_low AS conf_low, (_t).conf_high AS conf_high, (_t).conf_level AS conf_level, "
    "NULL::VARCHAR AS index_name, NULL::DOUBLE AS index_value, (_t).boot_mean AS boot_mean, (_t).vcov AS vcov, "
    "(_t).n_boot AS n_boot, (_t).n_failed AS n_failed "
    "FROM (SELECT group_col, unnest(range(len(_l))) AS _i, unnest(_l) AS _t "
    "FROM (SELECT group_col, bootcov_agg(y_col::DOUBLE, x_cols, cluster, options) AS _l "
    "FROM query_table(source::VARCHAR) GROUP BY group_col)) ORDER BY group_col, _i";

} // namespace

void RegisterModelValidationFunctions(ExtensionLoader &loader) {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	const vector<string> categories = {"regression", "validation"};
	const string model_note = "The model is a linear model (rms::ols, model 'ols', the default) or a binary logistic "
	                          "model (rms::lrm, model 'logistic', y in {0, 1}) with an intercept, of y on the "
	                          "columns of x. Rows with a NULL or non-finite y or x are skipped; the rows are sorted "
	                          "canonically, so a seed gives the same resamples whatever the thread count.";
	{
		AggregateFunctionSet set("validate_agg");
		set.AddFunction(MakeAggregate("validate_agg", {LogicalType::DOUBLE, dlist}, ANOFOX_AGG_BIND(ValidateBind)));
		set.AddFunction(MakeAggregate("validate_agg", {LogicalType::DOUBLE, dlist, LogicalType::ANY},
		                              ANOFOX_AGG_BIND(ValidateBind)));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string what = "Resampling validation of a regression model (rms::validate): one row per index with "
		                    "index_orig (apparent), training, test, optimism, index_corrected "
		                    "(index_orig - optimism), n (resamples used) and n_failed. ols indexes: R-square, MSE, "
		                    "g, Intercept, Slope; logistic: Dxy, R2, Intercept, Slope, Emax, D, U, Q, B, g, gp. " +
		                    model_note;
		info.descriptions.push_back(Describe(what, "validate_agg(y, [x1, x2])", {"y", "x"},
		                                     {LogicalType::DOUBLE, dlist}, categories));
		info.descriptions.push_back(
		    Describe(what + " Options: model ('ols' | 'logistic'), method ('boot' optimism bootstrap, the default | "
		                    "'crossvalidation' | '.632'), B (resamples or cross-validation groups, default 40), "
		                    "seed (default 42).",
		             "validate_agg(y, [x1, x2], {'model': 'logistic', 'B': 200, 'seed': 1})", {"y", "x", "options"},
		             {LogicalType::DOUBLE, dlist, LogicalType::ANY}, categories));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set("calibrate_agg");
		set.AddFunction(MakeAggregate("calibrate_agg", {LogicalType::DOUBLE, dlist}, ANOFOX_AGG_BIND(CalibrateBind)));
		set.AddFunction(MakeAggregate("calibrate_agg", {LogicalType::DOUBLE, dlist, LogicalType::ANY},
		                              ANOFOX_AGG_BIND(CalibrateBind)));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string what = "Overfitting-corrected calibration curve of a regression model (rms::calibrate): one "
		                    "row per grid point predy with calibrated_orig (apparent lowess calibration), "
		                    "calibrated_corrected, optimism, n, and the mean absolute / mean squared / 0.9 quantile "
		                    "absolute error of the corrected curve. " +
		                    model_note;
		info.descriptions.push_back(Describe(what, "calibrate_agg(y, [x1, x2])", {"y", "x"},
		                                     {LogicalType::DOUBLE, dlist}, categories));
		info.descriptions.push_back(
		    Describe(what + " Options: model, method ('boot' | 'crossvalidation' | '.632'), B (default 40), seed "
		                    "(default 42), predy (the grid; default 50 points between the 5th smallest and 5th "
		                    "largest prediction).",
		             "calibrate_agg(y, [x1, x2], {'model': 'logistic', 'B': 200})", {"y", "x", "options"},
		             {LogicalType::DOUBLE, dlist, LogicalType::ANY}, categories));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set("calibrate_cph_agg");
		set.AddFunction(MakeAggregate("calibrate_cph_agg",
		                              {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY},
		                              ANOFOX_AGG_BIND(CalibrateCphBind)));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		info.descriptions.push_back(Describe(
		    "Survival calibration of a Cox model at time u (rms::calibrate.cph, cmethod 'KM'): subjects are grouped "
		    "by quantiles of predicted survival at u (about m per group, or the given cuts); one row per group "
		    "with group_index, cut_low, cut_high, mean_predicted, km (Kaplan-Meier at u), km_corrected, std_err, "
		    "and the resampling table (index_orig = km - mean_predicted, training, test, optimism, "
		    "index_corrected, n). event is 1 for an event and 0 for censoring. Options: u (required), m (default "
		    "50), cuts, method ('boot' | 'crossvalidation' | '.632'), B (default 40), seed (default 42), ties "
		    "('efron' | 'breslow'), max_iterations (20), tolerance (1e-9; rms::cph uses 1e-4 and 10).",
		    "calibrate_cph_agg(time, [x1, x2], event, {'u': 5, 'm': 60, 'B': 100})",
		    {"time", "x", "event", "options"}, {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY},
		    {"survival", "validation"}));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set("bootcov_agg");
		set.AddFunction(MakeAggregate("bootcov_agg", {LogicalType::DOUBLE, dlist}, ANOFOX_AGG_BIND(BootcovBind)));
		set.AddFunction(MakeAggregate("bootcov_agg", {LogicalType::DOUBLE, dlist, LogicalType::ANY},
		                              ANOFOX_AGG_BIND(BootcovBind)));
		set.AddFunction(MakeAggregate("bootcov_agg", {LogicalType::DOUBLE, dlist, LogicalType::ANY, LogicalType::ANY},
		                              ANOFOX_AGG_BIND(BootcovBind)));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string what = "Bootstrap covariance of a regression model's coefficients (rms::bootcov): one row per "
		                    "coefficient ((Intercept), x1, ...) with estimate (the fit on all rows), std_error "
		                    "(bootstrap), conf_low / conf_high (percentile interval), conf_level, boot_mean, vcov "
		                    "(the coefficient's row of the bootstrap covariance), n_boot and n_failed. " +
		                    model_note;
		info.descriptions.push_back(
		    Describe(what, "bootcov_agg(y, [x1, x2])", {"y", "x"}, {LogicalType::DOUBLE, dlist}, categories));
		info.descriptions.push_back(Describe(what + " Options: model, B (default 200), seed (default 42), "
		                                            "conf_level (0.95), feature_names.",
		                                     "bootcov_agg(y, [x1, x2], {'B': 500, 'seed': 1})", {"y", "x", "options"},
		                                     {LogicalType::DOUBLE, dlist, LogicalType::ANY}, categories));
		info.descriptions.push_back(
		    Describe(what + " With a cluster id (any type) per row: the cluster bootstrap resamples whole clusters "
		                    "(bootcov(cluster = id)).",
		             "bootcov_agg(y, [x1, x2], patient_id, {'model': 'logistic'})", {"y", "x", "cluster", "options"},
		             {LogicalType::DOUBLE, dlist, LogicalType::ANY, LogicalType::ANY}, categories));
		loader.RegisterFunction(std::move(info));
	}

	const vector<string> macro_categories = {"regression", "validation", "table-macro"};
	auto validate_by = CreateContractTableMacro(
	    "validate_by", {"source", "group_col", "y_col", "x_cols"}, {{"options", "NULL"}}, VALIDATE_BY_SQL,
	    "Table macro: resampling validation (validate_agg, same options) per group of source, one row per index: "
	    "the group column, index_name, index_orig, training, test, optimism, index_corrected, n, n_failed, model, "
	    "method.",
	    "validate_by('patients', site, died, [age, sbp], options := {'model': 'logistic', 'B': 200})",
	    macro_categories);
	loader.RegisterFunction(*validate_by);
	auto calibrate_by = CreateContractTableMacro(
	    "calibrate_by", {"source", "group_col", "y_col", "x_cols"}, {{"options", "NULL"}}, CALIBRATE_BY_SQL,
	    "Table macro: the overfitting-corrected calibration curve (calibrate_agg, same options) per group of source, "
	    "one row per grid point in the integration contract's `curve` schema: the group column, id (the group), "
	    "model_id ('calibrate'), model_name (the model), curve_type ('calibrate'), series, x (predy), y "
	    "(calibrated_corrected), y_low, y_high, threshold, weight, n_censor, then predy, calibrated_orig, "
	    "calibrated_corrected, optimism, n, mean_abs_error, mean_squared_error, quantile90_abs_error, method.",
	    "calibrate_by('patients', site, died, [age, sbp], options := {'model': 'logistic'})", macro_categories);
	loader.RegisterFunction(*calibrate_by);
	auto calibrate_cph_by = CreateContractTableMacro(
	    "calibrate_cph_by", {"source", "group_col", "time_col", "x_cols", "event_col"}, {{"options", "NULL"}},
	    CALIBRATE_CPH_BY_SQL,
	    "Table macro: survival calibration of a Cox model at time u (calibrate_cph_agg, same options; u is "
	    "required) per group of source, one row per group of predicted survival: the group column, group_index, "
	    "cut_low, cut_high, mean_predicted, km, km_corrected, std_err, index_orig, training, test, optimism, "
	    "index_corrected, n, u, method.",
	    "calibrate_cph_by('patients', site, days, [age, treated], died, options := {'u': 365})",
	    {"survival", "validation", "table-macro"});
	loader.RegisterFunction(*calibrate_cph_by);
	auto bootcov_by = CreateContractTableMacro(
	    "bootcov_by", {"source", "group_col", "y_col", "x_cols"},
	    {{"options", "NULL"}, {"cluster", "NULL"}, {"names", "NULL"}}, BOOTCOV_BY_SQL,
	    "Table macro: bootstrap covariance of the coefficients (bootcov_agg, same options) per group of source, one "
	    "row per term in the integration contract's `terms` schema: the group column, id (the group), model_id "
	    "('bootcov'), model_name (the model), term, estimate, std_error (bootstrap), statistic, p_value (NULL), "
	    "conf_low, conf_high (percentile interval), conf_level, index_name, index_value, then boot_mean, vcov, "
	    "n_boot, n_failed. cluster := column resamples whole clusters; names := [...] labels the terms.",
	    "bootcov_by('patients', site, died, [age, sbp], options := {'model': 'logistic'}, cluster := patient_id)",
	    macro_categories);
	loader.RegisterFunction(*bootcov_by);
}

} // namespace duckdb
