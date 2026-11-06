// Multiple imputation and Rubin pooling (#321): areg_impute_agg, areg_impute_by and
// pool_rubin_agg. The statistics are upstream anofox-regression
// imputation::areg_impute (Hmisc::aregImpute, replaying R's random draws for a seed)
// and inference::pool (Rubin's rules with the Barnard-Rubin d.f. of mice::pool, the
// D1 Wald test of mice::D1); this file only buffers rows / model STRUCTs, parses the
// options and writes the results.
//
//   -- m completed copies of the rows, then one fit per copy, then pooled
//   WITH imp AS (SELECT * FROM areg_impute_by('d', grp, id, [y, x1, x2], options := {'n_impute': 5}))
//   SELECT pool_rubin_agg(fit) FROM (
//       SELECT imputation, ols_fit_agg(x[1], [x[2], x[3]], {'compute_inference': true}) AS fit
//       FROM imp GROUP BY imputation);

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "anofox_statistics_banner.hpp"
#include "anofox_statistics_extension.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

namespace {

constexpr idx_t kNone = DConstants::INVALID_INDEX;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

//! Walk a MAP / STRUCT options value as (lower-case key, value) pairs.
template <class FN>
void VisitImputeOptions(const char *fn, const Value &options, FN &&visit) {
	const Value v = UnwrapVariant(options);
	if (v.IsNull()) {
		return;
	}
	if (v.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(v)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv.size() != 2 || kv[0].IsNull()) {
				throw InvalidInputException("%s: invalid MAP option entry", fn);
			}
			visit(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else if (v.type().id() == LogicalTypeId::STRUCT) {
		auto &children = StructValue::GetChildren(v);
		auto &types = StructType::GetChildTypes(v.type());
		for (idx_t i = 0; i < types.size(); i++) {
			visit(StringUtil::Lower(NameString(types[i].first)), UnwrapVariant(children[i]));
		}
	} else {
		throw InvalidInputException("%s: options must be a MAP or STRUCT, got %s", fn, v.type().ToString());
	}
}

double OptionNumber(const char *fn, const string &key, const Value &v) {
	try {
		return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
	}
}

bool OptionBool(const char *fn, const string &key, const Value &v) {
	try {
		return v.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
	} catch (std::exception &) {
		throw InvalidInputException("%s: option '%s' must be a boolean, got '%s'", fn, key, v.ToString());
	}
}

//! A whole number in [lo, hi].
int64_t OptionInteger(const char *fn, const string &key, const Value &v, double lo, double hi) {
	double d = OptionNumber(fn, key, v);
	if (!(d >= lo && d <= hi) || d != std::floor(d)) {
		throw InvalidInputException("%s: option '%s' must be a whole number between %.0f and %.0f, got '%s'", fn, key,
		                            lo, hi, v.ToString());
	}
	return static_cast<int64_t>(d);
}

//===--------------------------------------------------------------------===//
// areg_impute_agg(id, x DOUBLE[] [, options])
//===--------------------------------------------------------------------===//
constexpr const char *kImpute = "areg_impute_agg";

struct AregImputeBindData : public FunctionData {
	AnofoxAregImputeOptions options {0.2, -1, 5, 3, 1, 0, 1, 0};
	//! 1-based positions in x of categorical / always-linear variables.
	vector<idx_t> categorical;
	vector<idx_t> linear;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AregImputeBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<AregImputeBindData>();
		return options.fweighted == o.options.fweighted && options.burnin == o.options.burnin &&
		       options.n_impute == o.options.n_impute && options.nk == o.options.nk &&
		       options.seed == o.options.seed && options.closest == o.options.closest &&
		       options.tlinear == o.options.tlinear && categorical == o.categorical && linear == o.linear;
	}
};

vector<idx_t> OptionPositions(const string &key, const Value &v) {
	vector<idx_t> out;
	auto add = [&](const Value &e) {
		if (e.IsNull()) {
			throw InvalidInputException("%s: option '%s' must not hold NULL", kImpute, key);
		}
		out.push_back(static_cast<idx_t>(OptionInteger(kImpute, key, e, 1, 1e9)));
	};
	if (v.type().id() == LogicalTypeId::LIST) {
		for (auto &e : ListValue::GetChildren(v)) {
			add(UnwrapVariant(e));
		}
	} else {
		add(v);
	}
	return out;
}

void ParseImputeOptions(const Value &value, AregImputeBindData &bind) {
	VisitImputeOptions(kImpute, value, [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "n_impute") {
			bind.options.n_impute = static_cast<uint64_t>(OptionInteger(kImpute, key, v, 1, 1000));
		} else if (key == "nk") {
			auto nk = OptionInteger(kImpute, key, v, 0, 20);
			if (nk == 1 || nk == 2) {
				throw InvalidInputException("%s: option 'nk' must be 0 (all linear) or at least 3 knots, got %lld",
				                            kImpute, (long long)nk);
			}
			bind.options.nk = static_cast<uint64_t>(nk);
		} else if (key == "tlinear") {
			bind.options.tlinear = OptionBool(kImpute, key, v) ? 1 : 0;
		} else if (key == "burnin") {
			bind.options.burnin = OptionInteger(kImpute, key, v, 0, 1000);
		} else if (key == "match") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "weighted") {
				bind.options.closest = 0;
			} else if (s == "closest") {
				bind.options.closest = 1;
			} else {
				throw InvalidInputException("%s: unknown match '%s'. Expected 'weighted' or 'closest'.", kImpute,
				                            v.ToString());
			}
		} else if (key == "fweighted") {
			double f = OptionNumber(kImpute, key, v);
			if (!(f > 0.0) || !std::isfinite(f)) {
				throw InvalidInputException("%s: option 'fweighted' must be positive, got %g", kImpute, f);
			}
			bind.options.fweighted = f;
		} else if (key == "seed") {
			bind.options.seed = static_cast<int32_t>(OptionInteger(kImpute, key, v, -2147483647.0, 2147483647.0));
		} else if (key == "categorical") {
			bind.categorical = OptionPositions(key, v);
		} else if (key == "linear") {
			bind.linear = OptionPositions(key, v);
		} else {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: n_impute, nk, tlinear, burnin, "
			                            "match, fweighted, seed, categorical, linear",
			                            kImpute, key);
		}
	});
	for (auto c : bind.categorical) {
		if (std::find(bind.linear.begin(), bind.linear.end(), c) != bind.linear.end()) {
			throw InvalidInputException("%s: column %llu cannot be both categorical and linear", kImpute,
			                            (unsigned long long)c);
		}
	}
}

struct AregImputeState {
	vector<Value> ids;
	//! Row-major values, NaN = missing.
	vector<double> values;
	idx_t n_cols = 0;
	bool initialized = false;
};

LogicalType AregImputeRowType(const LogicalType &id_type) {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("imputation", LogicalType::INTEGER));
	children.push_back(make_pair("id", id_type));
	children.push_back(make_pair("x", LogicalType::LIST(LogicalType::DOUBLE)));
	return LogicalType::STRUCT(std::move(children));
}

void AregImputeInitialize(data_ptr_t state_p) {
	new (state_p) AregImputeState();
}

void AregImputeDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AregImputeState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~AregImputeState();
	}
}

void CheckWidth(AregImputeState &state, idx_t width) {
	if (!state.initialized) {
		state.n_cols = width;
		state.initialized = true;
	} else if (width != state.n_cols) {
		throw InvalidInputException("%s: inconsistent number of variables: expected %llu, got %llu", kImpute,
		                            (unsigned long long)state.n_cols, (unsigned long long)width);
	}
}

void AregImputeUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat id_data;
	inputs[0].ToUnifiedFormat(count, id_data);
	UnifiedVectorFormat x_data;
	inputs[1].ToUnifiedFormat(count, x_data);
	auto x_lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_values = FlatDataMut<double>(x_child);
	auto &x_validity = FlatValidityMut(x_child);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AregImputeState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto x_idx = x_data.sel->get_index(i);
		// A NULL id or a NULL x row is left out; NULL / NaN elements are the missing values.
		if (!id_data.validity.RowIsValid(id_data.sel->get_index(i)) || !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		auto entry = x_lists[x_idx];
		CheckWidth(state, entry.length);
		state.ids.push_back(inputs[0].GetValue(i));
		for (idx_t j = 0; j < entry.length; j++) {
			auto k = entry.offset + j;
			state.values.push_back(x_validity.RowIsValid(k) ? x_values[k] : kNaN);
		}
	}
}

void AregImputeCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                       idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (AregImputeState **)source_data.data;
	auto targets = (AregImputeState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.ids = CombineTake(source.ids, aggr_input_data);
			target.values = CombineTake(source.values, aggr_input_data);
			target.n_cols = source.n_cols;
			target.initialized = true;
			continue;
		}
		CheckWidth(target, source.n_cols);
		target.ids.insert(target.ids.end(), source.ids.begin(), source.ids.end());
		target.values.insert(target.values.end(), source.values.begin(), source.values.end());
	}
}

void AregImputeFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AregImputeState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<AregImputeBindData>();
	auto list_data = FlatDataMut<list_entry_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.ids.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t n = state.ids.size();
		idx_t p = state.n_cols;
		// Rows in id order: R's aregImpute sees them in that order, and the draws
		// (hence the imputations) depend on it.
		vector<idx_t> order(n);
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(),
		                 [&](idx_t a, idx_t b) { return state.ids[a] < state.ids[b]; });
		for (idx_t r = 1; r < n; r++) {
			if (state.ids[order[r]] == state.ids[order[r - 1]]) {
				throw InvalidInputException("%s: duplicate id %s", kImpute, state.ids[order[r]].ToString());
			}
		}
		vector<double> data(n * p);
		for (idx_t r = 0; r < n; r++) {
			for (idx_t j = 0; j < p; j++) {
				data[r * p + j] = state.values[order[r] * p + j];
			}
		}
		vector<int32_t> kinds(p, 0);
		for (auto &spec : {std::make_pair(&bind.categorical, 2), std::make_pair(&bind.linear, 1)}) {
			for (auto c : *spec.first) {
				if (c > p) {
					throw InvalidInputException("%s: column %llu is beyond the %llu variables of x", kImpute,
					                            (unsigned long long)c, (unsigned long long)p);
				}
				kinds[c - 1] = spec.second;
			}
		}
		idx_t m = static_cast<idx_t>(bind.options.n_impute);
		vector<double> completed(m * n * p);
		vector<double> r_squared(p);
		AnofoxError error;
		if (!anofox_areg_impute(data.data(), static_cast<size_t>(n), static_cast<size_t>(p), kinds.data(),
		                        bind.options, completed.data(), r_squared.data(), &error)) {
			ThrowUnlessDegenerate(kImpute, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t total = m * n;
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + total);
		auto &rows = ListVector::GetEntry(result);
		auto &entries = StructVector::GetEntries(rows);
		auto &imp_vec = StructEntry(entries[0]);
		auto &id_vec = StructEntry(entries[1]);
		auto &x_vec = StructEntry(entries[2]);
		for (idx_t k = 0; k < m; k++) {
			for (idx_t r = 0; r < n; r++) {
				idx_t out = list_offset + k * n + r;
				FlatDataMut<int32_t>(imp_vec)[out] = static_cast<int32_t>(k + 1);
				id_vec.SetValue(out, state.ids[order[r]]);
				auto x_offset = ListVector::GetListSize(x_vec);
				ListVector::Reserve(x_vec, x_offset + p);
				auto x_data = FlatDataMut<double>(ListVector::GetEntry(x_vec));
				for (idx_t j = 0; j < p; j++) {
					x_data[x_offset + j] = completed[(k * n + r) * p + j];
				}
				ListVector::SetListSize(x_vec, x_offset + p);
				ListVector::GetData(x_vec)[out] = list_entry_t(x_offset, p);
			}
		}
		ListVector::SetListSize(result, list_offset + total);
		list_data[row] = list_entry_t(list_offset, total);
	}
}

unique_ptr<FunctionData> AregImputeBind(ClientContext &context, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<AregImputeBindData>();
	if (arguments.size() >= 3) {
		ParseImputeOptions(EvaluateConstantOptions(context, *arguments[2], kImpute), *bind);
	}
	auto id_type = ReturnTypeOf(*arguments[0]);
	if (id_type.id() == LogicalTypeId::SQLNULL) {
		id_type = LogicalType::INTEGER;
	}
	SetReturnTypeOf(function, LogicalType::LIST(AregImputeRowType(id_type)));
	PostHogTelemetry::Instance().RecordFunctionCall(kImpute);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// pool_rubin_agg(model [, options])
//===--------------------------------------------------------------------===//
constexpr const char *kPool = "pool_rubin_agg";

//! Where a model STRUCT keeps the fields pooling reads.
struct PoolFields {
	idx_t coefficients = kNone, intercept = kNone, vcov = kNone, std_errors = kNone, intercept_se = kNone,
	      df_residual = kNone;

	bool operator==(const PoolFields &o) const {
		return coefficients == o.coefficients && intercept == o.intercept && vcov == o.vcov &&
		       std_errors == o.std_errors && intercept_se == o.intercept_se && df_residual == o.df_residual;
	}
};

struct PoolBindData : public FunctionData {
	PoolFields fields;
	double confidence_level = 0.95;
	//! Complete-data d.f. from the options (NaN: the models' df_residual).
	double df_complete = kNaN;
	//! false: Rubin's large-sample d.f. whatever the complete-data d.f.
	bool small_sample = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PoolBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PoolBindData>();
		return fields == o.fields && confidence_level == o.confidence_level &&
		       ((std::isnan(df_complete) && std::isnan(o.df_complete)) || df_complete == o.df_complete) &&
		       small_sample == o.small_sample;
	}
};

bool IsNumericList(const LogicalType &t) {
	return t.id() == LogicalTypeId::LIST && ListType::GetChildType(t).IsNumeric();
}

PoolFields FindPoolFields(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: model must be the STRUCT returned by a *_fit_agg or *_fit function, got %s",
		                            kPool, type.ToString());
	}
	PoolFields f;
	auto &children = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = NameString(children[i].first);
		auto &t = children[i].second;
		if (IsNumericList(t)) {
			if (name == "coefficients") {
				f.coefficients = i;
			} else if (name == "vcov") {
				f.vcov = i;
			} else if (name == "std_errors") {
				f.std_errors = i;
			}
		} else if (t.IsNumeric()) {
			if (name == "intercept") {
				f.intercept = i;
			} else if (name == "intercept_std_error") {
				f.intercept_se = i;
			} else if (name == "df_residual") {
				f.df_residual = i;
			}
		}
	}
	if (f.coefficients == kNone) {
		throw InvalidInputException("%s: model has no 'coefficients' field; got %s", kPool, type.ToString());
	}
	if (f.vcov == kNone && f.std_errors == kNone) {
		throw InvalidInputException("%s: model has neither a covariance 'vcov' nor 'std_errors'; got %s", kPool,
		                            type.ToString());
	}
	return f;
}

void ParsePoolOptions(const Value &value, PoolBindData &bind) {
	VisitImputeOptions(kPool, value, [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "confidence_level" || key == "conf_level") {
			double d = OptionNumber(kPool, key, v);
			if (!(d > 0.0 && d < 1.0)) {
				throw InvalidInputException("%s: confidence_level must be in (0, 1), got %g", kPool, d);
			}
			bind.confidence_level = d;
		} else if (key == "df_complete" || key == "dfcom") {
			double d = OptionNumber(kPool, key, v);
			if (!(d > 0.0) || std::isnan(d)) {
				throw InvalidInputException("%s: df_complete must be positive, got %g", kPool, d);
			}
			bind.df_complete = d;
		} else if (key == "small_sample") {
			bind.small_sample = OptionBool(kPool, key, v);
		} else {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: confidence_level, df_complete, "
			                            "small_sample",
			                            kPool, key);
		}
	});
}

struct PoolState {
	//! One entry per imputation: estimates (intercept first) and row-major covariance.
	vector<vector<double>> estimates;
	vector<vector<double>> vcovs;
	//! Smallest complete-data d.f. of the models (NaN: none).
	double df_residual = kNaN;
	bool has_intercept = false;
	bool initialized = false;
};

void PoolInitialize(data_ptr_t state_p) {
	new (state_p) PoolState();
}

void PoolDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PoolState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~PoolState();
	}
}

double ValueDouble(const Value &v) {
	return v.IsNull() ? kNaN : v.GetValue<double>();
}

vector<double> ListDoubles(const Value &v, const char *what) {
	vector<double> out;
	for (auto &c : ListValue::GetChildren(v)) {
		if (c.IsNull()) {
			throw InvalidInputException("%s: the model's %s hold a NULL (an aliased coefficient?); every imputation "
			                            "must estimate every coefficient",
			                            kPool, what);
		}
		out.push_back(c.GetValue<double>());
	}
	return out;
}

double MinDf(double a, double b) {
	if (std::isnan(a)) {
		return b;
	}
	if (std::isnan(b)) {
		return a;
	}
	return std::min(a, b);
}

void AddModel(PoolState &state, vector<double> estimates, vector<double> vcov, bool has_intercept, double df) {
	if (!state.initialized) {
		state.has_intercept = has_intercept;
		state.initialized = true;
	} else if (has_intercept != state.has_intercept || estimates.size() != state.estimates[0].size()) {
		throw InvalidInputException("%s: every imputation's model must have the same coefficients (got %llu%s and "
		                            "%llu%s)",
		                            kPool, (unsigned long long)state.estimates[0].size(),
		                            state.has_intercept ? " with intercept" : "", (unsigned long long)estimates.size(),
		                            has_intercept ? " with intercept" : "");
	}
	state.estimates.push_back(std::move(estimates));
	state.vcovs.push_back(std::move(vcov));
	state.df_residual = MinDf(state.df_residual, df);
}

void PoolUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<PoolBindData>();
	auto &f = bind.fields;
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PoolState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto model = inputs[0].GetValue(i);
		if (model.IsNull()) {
			continue;
		}
		auto &children = StructValue::GetChildren(model);
		if (children[f.coefficients].IsNull()) {
			continue; // a fit that failed (NULL coefficients) is left out
		}
		auto slopes = ListDoubles(children[f.coefficients], "coefficients");
		double intercept = f.intercept == kNone ? kNaN : ValueDouble(children[f.intercept]);
		bool has_intercept = !std::isnan(intercept);
		idx_t dim = slopes.size() + (has_intercept ? 1 : 0);
		vector<double> estimates;
		if (has_intercept) {
			estimates.push_back(intercept);
		}
		estimates.insert(estimates.end(), slopes.begin(), slopes.end());
		vector<double> vcov;
		if (f.vcov != kNone && !children[f.vcov].IsNull()) {
			vcov = ListDoubles(children[f.vcov], "vcov");
			if (vcov.size() != dim * dim) {
				throw InvalidInputException("%s: vcov has %llu entries but the model has %llu coefficients", kPool,
				                            (unsigned long long)vcov.size(), (unsigned long long)dim);
			}
		} else if (f.std_errors != kNone && !children[f.std_errors].IsNull()) {
			// Without a covariance the coefficients are pooled one by one (the D1 test
			// then assumes them uncorrelated).
			vector<double> se;
			if (has_intercept) {
				double ise = f.intercept_se == kNone ? kNaN : ValueDouble(children[f.intercept_se]);
				if (std::isnan(ise)) {
					throw InvalidInputException("%s: the model has an intercept but no intercept_std_error", kPool);
				}
				se.push_back(ise);
			}
			auto slope_se = ListDoubles(children[f.std_errors], "std_errors");
			se.insert(se.end(), slope_se.begin(), slope_se.end());
			if (se.size() != dim) {
				throw InvalidInputException("%s: std_errors has %llu entries but the model has %llu coefficients",
				                            kPool, (unsigned long long)se.size(), (unsigned long long)dim);
			}
			vcov.assign(dim * dim, 0.0);
			for (idx_t j = 0; j < dim; j++) {
				vcov[j * dim + j] = se[j] * se[j];
			}
		} else {
			throw InvalidInputException("%s: the model has no covariance or standard errors (vcov and std_errors are "
			                            "NULL); fit it with {'compute_inference': true}",
			                            kPool);
		}
		double df = f.df_residual == kNone ? kNaN : ValueDouble(children[f.df_residual]);
		AddModel(*states[sdata.sel->get_index(i)], std::move(estimates), std::move(vcov), has_intercept, df);
	}
}

void PoolCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (PoolState **)source_data.data;
	auto targets = (PoolState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.estimates = CombineTake(source.estimates, aggr_input_data);
			target.vcovs = CombineTake(source.vcovs, aggr_input_data);
			target.df_residual = source.df_residual;
			target.has_intercept = source.has_intercept;
			target.initialized = true;
			continue;
		}
		for (idx_t k = 0; k < source.estimates.size(); k++) {
			AddModel(target, source.estimates[k], source.vcovs[k], source.has_intercept, source.df_residual);
		}
	}
}

// Result layout: model-STRUCT style (tidy() reads it), slopes in the lists and the
// intercept in intercept_* scalars, then the D1 test of the slopes and the pooling
// metadata.
const char *const kPoolTermLists[] = {"coefficients", "std_errors", "t_values", "p_values", "conf_low",
                                      "conf_high",    "df",         "riv",      "lambda",   "fmi"};
const char *const kPoolInterceptFields[] = {"intercept",          "intercept_std_error", "intercept_statistic",
                                            "intercept_p_value",  "intercept_conf_low",  "intercept_conf_high",
                                            "intercept_df",       "intercept_riv",       "intercept_lambda",
                                            "intercept_fmi"};
constexpr idx_t kPoolTermCount = 10;

LogicalType PoolResultType() {
	child_list_t<LogicalType> c;
	for (auto name : kPoolTermLists) {
		c.push_back(MakeChild(name, LogicalType::LIST(LogicalType::DOUBLE)));
	}
	for (auto name : kPoolInterceptFields) {
		c.push_back(MakeChild(name, LogicalType::DOUBLE));
	}
	c.push_back(make_pair("conf_level", LogicalType::DOUBLE));
	c.push_back(make_pair("vcov", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("f_df1", LogicalType::DOUBLE));
	c.push_back(make_pair("f_df2", LogicalType::DOUBLE));
	c.push_back(make_pair("f_pvalue", LogicalType::DOUBLE));
	c.push_back(make_pair("f_riv", LogicalType::DOUBLE));
	c.push_back(make_pair("df_complete", LogicalType::DOUBLE));
	c.push_back(make_pair("n_imputations", LogicalType::BIGINT));
	c.push_back(make_pair("model_type", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

double TermField(const AnofoxPooledTerm &t, idx_t k) {
	// in the order of kPoolTermLists / kPoolInterceptFields
	const double v[kPoolTermCount] = {t.estimate,  t.std_error, t.statistic, t.p_value, t.conf_low,
	                                  t.conf_high, t.df,        t.riv,       t.lambda,  t.fmi};
	return v[k];
}

//! A DOUBLE, NULL when not finite.
void SetFinite(Vector &v, idx_t row, double d) {
	if (std::isfinite(d)) {
		FlatDataMut<double>(v)[row] = d;
	} else {
		FlatVector::SetNull(v, row, true);
	}
}

void AppendList(Vector &list_vec, idx_t row, const vector<double> &values) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + values.size());
	auto &child = ListVector::GetEntry(list_vec);
	auto data = FlatDataMut<double>(child);
	for (idx_t i = 0; i < values.size(); i++) {
		if (std::isfinite(values[i])) {
			data[offset + i] = values[i];
		} else {
			FlatVector::SetNull(child, offset + i, true);
		}
	}
	ListVector::SetListSize(list_vec, offset + values.size());
	ListVector::GetData(list_vec)[row] = list_entry_t(offset, values.size());
}

void PoolFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                  idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PoolState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<PoolBindData>();
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		idx_t m = state.estimates.size();
		if (!state.initialized || m < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical order of the imputations: bitwise-identical sums whatever the
		// thread count (#226).
		vector<idx_t> order(m);
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
			if (state.estimates[a] != state.estimates[b]) {
				return state.estimates[a] < state.estimates[b];
			}
			return state.vcovs[a] < state.vcovs[b];
		});
		idx_t dim = state.estimates[0].size();
		vector<double> est, cov;
		est.reserve(m * dim);
		cov.reserve(m * dim * dim);
		for (auto k : order) {
			est.insert(est.end(), state.estimates[k].begin(), state.estimates[k].end());
			cov.insert(cov.end(), state.vcovs[k].begin(), state.vcovs[k].end());
		}
		double df_complete = !bind.small_sample                ? kNaN
		                     : std::isnan(bind.df_complete) ? state.df_residual
		                                                       : bind.df_complete;
		if (!(df_complete > 0.0)) {
			df_complete = kNaN; // a model without residual d.f. (or <= 0): Rubin's d.f.
		}
		idx_t first_slope = state.has_intercept ? 1 : 0;
		vector<AnofoxPooledTerm> terms(dim);
		vector<double> vcov(dim * dim);
		AnofoxPooledWald wald;
		AnofoxError error;
		if (!anofox_pool_rubin(est.data(), cov.data(), static_cast<size_t>(m), static_cast<size_t>(dim), df_complete,
		                       bind.confidence_level, static_cast<size_t>(first_slope), terms.data(), vcov.data(), &wald,
		                       &error)) {
			ThrowUnlessDegenerate(kPool, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		for (idx_t k = 0; k < kPoolTermCount; k++) {
			vector<double> values;
			for (idx_t j = first_slope; j < dim; j++) {
				values.push_back(TermField(terms[j], k));
			}
			AppendList(StructEntry(entries[k]), row, values);
			auto &scalar = StructEntry(entries[kPoolTermCount + k]);
			SetFinite(scalar, row, state.has_intercept ? TermField(terms[0], k) : kNaN);
		}
		idx_t f = 2 * kPoolTermCount;
		SetFinite(StructEntry(entries[f]), row, bind.confidence_level);
		AppendList(StructEntry(entries[f + 1]), row, vcov);
		SetFinite(StructEntry(entries[f + 2]), row, wald.f);
		SetFinite(StructEntry(entries[f + 3]), row, wald.df1);
		SetFinite(StructEntry(entries[f + 4]), row, wald.df2);
		SetFinite(StructEntry(entries[f + 5]), row, wald.p_value);
		SetFinite(StructEntry(entries[f + 6]), row, wald.riv);
		SetFinite(StructEntry(entries[f + 7]), row, df_complete);
		FlatDataMut<int64_t>(StructEntry(entries[f + 8]))[row] = static_cast<int64_t>(m);
		StructEntry(entries[f + 9]).SetValue(row, Value("rubin_pooled"));
	}
}

unique_ptr<FunctionData> PoolBind(ClientContext &context, AggregateBindFunction &function,
                                  vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<PoolBindData>();
	bind->fields = FindPoolFields(ReturnTypeOf(*arguments[0]));
	if (arguments.size() >= 2) {
		ParsePoolOptions(EvaluateConstantOptions(context, *arguments[1], kPool), *bind);
	}
	SetReturnTypeOf(function, PoolResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kPool);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// areg_impute_by(source, group_col, id_col, x_cols [, options])
//===--------------------------------------------------------------------===//
const char *const AREG_IMPUTE_BY_SQL = R"(
SELECT group_col, _anofox_r.imputation AS imputation, _anofox_r.id AS id, _anofox_r.x AS x
FROM (
    SELECT group_col, UNNEST(areg_impute_agg(id_col, x_cols, options)) AS _anofox_r
    FROM query_table(source::VARCHAR)
    GROUP BY group_col
) sub
ORDER BY group_col, imputation, id
)";

} // namespace

void RegisterImputationFunctions(ExtensionLoader &loader) {
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	{
		AggregateFunctionSet func_set("areg_impute_agg");
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::ANY, dlist};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			func_set.AddFunction(AggregateFunction(
			    "areg_impute_agg", args, LogicalType::ANY, AggregateFunction::StateSize<AregImputeState>,
			    ANOFOX_AGG_INIT(AregImputeInitialize),
			    ANOFOX_GUARDED_UPDATE(AregImputeUpdate, AregImputeDestroy, AregImputeInitialize), AregImputeCombine,
			    ANOFOX_GUARDED_FINALIZE(AregImputeFinalize, AregImputeDestroy, AregImputeInitialize), nullptr,
			    ANOFOX_AGG_BIND(AregImputeBind), AregImputeDestroy));
		}
		CreateAggregateFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d1;
		d1.description =
		    "Multiple imputation by additive regression, bootstrapping and predictive mean matching (Hmisc::aregImpute "
		    "on the variables of x, rows in id order; NULL or NaN elements are missing). Returns the completed data "
		    "sets as LIST(STRUCT(imputation INTEGER, id, x DOUBLE[])), imputation 1..n_impute; observed values are "
		    "kept. With the same seed the imputations replay set.seed(seed); aregImpute(...) in R.";
		d1.examples = {"areg_impute_agg(id, [y, x1, x2])"};
		d1.categories = {"imputation"};
		d1.parameter_names = {"id", "x"};
		d1.parameter_types = {LogicalType::ANY, dlist};
		info.descriptions.push_back(std::move(d1));
		FunctionDescription d2;
		d2.description = "areg_impute_agg with options: n_impute (5), nk (3), tlinear (true), burnin (Hmisc default), "
		                 "match ('weighted' | 'closest'), fweighted (0.2), seed (1), categorical and linear (1-based "
		                 "positions in x).";
		d2.examples = {"areg_impute_agg(id, [y, x1, g], {'n_impute': 10, 'categorical': [3], 'seed': 42})"};
		d2.categories = {"imputation"};
		d2.parameter_names = {"id", "x", "options"};
		d2.parameter_types = {LogicalType::ANY, dlist, LogicalType::ANY};
		info.descriptions.push_back(std::move(d2));
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet func_set("pool_rubin_agg");
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::ANY};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			func_set.AddFunction(AggregateFunction(
			    "pool_rubin_agg", args, LogicalType::ANY, AggregateFunction::StateSize<PoolState>,
			    ANOFOX_AGG_INIT(PoolInitialize), ANOFOX_GUARDED_UPDATE(PoolUpdate, PoolDestroy, PoolInitialize),
			    PoolCombine, ANOFOX_GUARDED_FINALIZE(PoolFinalize, PoolDestroy, PoolInitialize), nullptr,
			    ANOFOX_AGG_BIND(PoolBind), PoolDestroy));
		}
		CreateAggregateFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d1;
		d1.description =
		    "Pools one fitted model STRUCT per imputed data set by Rubin's rules (mice::pool): pooled coefficients, "
		    "total standard errors, t statistics, Barnard-Rubin d.f. (complete-data d.f. from the models' df_residual), "
		    "p-values, confidence intervals, relative increase in variance (riv), lambda and the fraction of missing "
		    "information (fmi), the intercept in intercept_*, the pooled vcov and the D1 Wald test of all slopes "
		    "(mice::D1, f_statistic / f_df1 / f_df2 / f_pvalue). Reads coefficients, intercept and vcov (or "
		    "std_errors) of each model; tidy() lists the pooled terms.";
		d1.examples = {"pool_rubin_agg(fit)"};
		d1.categories = {"imputation", "regression"};
		d1.parameter_names = {"model"};
		d1.parameter_types = {LogicalType::ANY};
		info.descriptions.push_back(std::move(d1));
		FunctionDescription d2;
		d2.description = "pool_rubin_agg with options: confidence_level (0.95), df_complete (overrides the models' "
		                 "df_residual), small_sample (true; false gives Rubin's large-sample d.f.).";
		d2.examples = {"pool_rubin_agg(fit, {'confidence_level': 0.9})"};
		d2.categories = {"imputation", "regression"};
		d2.parameter_names = {"model", "options"};
		d2.parameter_types = {LogicalType::ANY, LogicalType::ANY};
		info.descriptions.push_back(std::move(d2));
		loader.RegisterFunction(std::move(info));
	}
	{
		auto macro = CreateContractTableMacro(
		    "areg_impute_by", {"source", "group_col", "id_col", "x_cols"}, {{"options", "NULL"}}, AREG_IMPUTE_BY_SQL,
		    "Table macro: multiple imputation per group of source (areg_impute_agg over the group's rows of x_cols in "
		    "id_col order, options as areg_impute_agg); returns one row per group, imputation and id with the "
		    "completed x (DOUBLE[]).",
		    "areg_impute_by('survey', region, respondent_id, [age, income, score], options := {'n_impute': 5})",
		    {"imputation", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

} // namespace duckdb
