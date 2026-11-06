// Spline knots from data (#317):
//
//   rcs_knots_agg(x [, nk [, options]])          -> DOUBLE[]  default rcs knots (rms::rcs / Hmisc::rcspline.eval)
//   bspline_knots_agg(x [, n_knots [, options]]) -> DOUBLE[]  B-spline base knots (sklearn SplineTransformer)
//
// rcs_knots_agg: nk knots (default 5, at least 3) at the quantiles of RMS
// Table 2.3, the outer knots moved to the 5th smallest / largest value when
// n < 100 and tied extremes handled as rcspline.eval (options: fractied,
// default 0.05). bspline_knots_agg: n_knots base knots (default 5, at least 2),
// options knots 'uniform' (default, min to max) | 'quantile'. NULL / NaN / Inf
// values are skipped; data that cannot place the knots (fewer than 6 values,
// too few distinct values) give NULL. The knots come from anofox-regression
// (splines::rcs_default_knots_with, BSpline::from_data).

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

static constexpr const char *RCS_KNOTS_AGG = "rcs_knots_agg";
static constexpr const char *BSPLINE_KNOTS_AGG = "bspline_knots_agg";

struct SplineKnotsState {
	vector<double> values;
};

struct SplineKnotsBindData : public FunctionData {
	bool bspline = false;
	size_t n_knots = 5;
	double fractied = 0.05;
	int32_t placement = 0; // bspline: 0 uniform, 1 quantile

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SplineKnotsBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<SplineKnotsBindData>();
		return bspline == o.bspline && n_knots == o.n_knots && fractied == o.fractied && placement == o.placement;
	}
};

static void SplineKnotsInitialize(data_ptr_t state_p) {
	new (state_p) SplineKnotsState();
}

static void SplineKnotsDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (SplineKnotsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~SplineKnotsState();
	}
}

static void SplineKnotsUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat value_data, sdata;
	inputs[0].ToUnifiedFormat(count, value_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);
	auto states = (SplineKnotsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = value_data.sel->get_index(i);
		if (!value_data.validity.RowIsValid(idx) || !std::isfinite(values[idx])) {
			continue;
		}
		states[sdata.sel->get_index(i)]->values.push_back(values[idx]);
	}
}

static void SplineKnotsCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                               idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (SplineKnotsState **)source_data.data;
	auto targets = (SplineKnotsState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (target.values.empty()) {
			target.values = CombineTake(source.values, aggr_input_data);
		} else {
			target.values.insert(target.values.end(), source.values.begin(), source.values.end());
		}
	}
}

static void SplineKnotsFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (SplineKnotsState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<SplineKnotsBindData>();
	const char *fn = bind.bspline ? BSPLINE_KNOTS_AGG : RCS_KNOTS_AGG;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxDataArray x;
		x.data = state.values.data();
		x.validity = nullptr;
		x.len = static_cast<size_t>(state.values.size());

		// rcspline.eval's alternate algorithm may place up to 2 nk knots.
		vector<double> knots(2 * bind.n_knots + 2);
		size_t len = 0;
		AnofoxError error;
		bool ok = bind.bspline ? anofox_bspline_data_knots(x, bind.n_knots, bind.placement, knots.data(),
		                                                   knots.size(), &len, &error)
		                       : anofox_rcs_default_knots(x, bind.n_knots, bind.fractied, knots.data(), knots.size(),
		                                                  &len, &error);
		if (ok && len > knots.size()) {
			knots.resize(len);
			ok = bind.bspline ? anofox_bspline_data_knots(x, bind.n_knots, bind.placement, knots.data(), knots.size(),
			                                              &len, &error)
			                  : anofox_rcs_default_knots(x, bind.n_knots, bind.fractied, knots.data(), knots.size(),
			                                             &len, &error);
		}
		if (!ok) {
			ThrowUnlessDegenerate(fn, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + len);
		auto data = FlatDataMut<double>(ListVector::GetEntry(result));
		for (idx_t j = 0; j < len; j++) {
			data[list_offset + j] = knots[j];
		}
		ListVector::SetListSize(result, list_offset + len);
		ListVector::GetData(result)[row] = {list_offset, len};
	}
}

static unique_ptr<FunctionData> SplineKnotsBind(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                                bool bspline) {
	const char *fn = bspline ? BSPLINE_KNOTS_AGG : RCS_KNOTS_AGG;
	auto bind = make_uniq<SplineKnotsBindData>();
	bind->bspline = bspline;
	if (arguments.size() >= 2) {
		if (!arguments[1]->IsFoldable()) {
			throw InvalidInputException("%s: %s must be a constant", fn, bspline ? "n_knots" : "nk");
		}
		auto v = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		if (v.IsNull()) {
			throw InvalidInputException("%s: %s must not be NULL", fn, bspline ? "n_knots" : "nk");
		}
		auto n = v.GetValue<int64_t>();
		const int64_t min_knots = bspline ? 2 : 3;
		if (n < min_knots || n > 1000) {
			throw InvalidInputException("%s: %s must be between %lld and 1000, got %lld", fn,
			                            bspline ? "n_knots" : "nk", (long long)min_knots, (long long)n);
		}
		bind->n_knots = static_cast<size_t>(n);
	}
	if (arguments.size() >= 3) {
		Value options_val = EvaluateConstantOptions(context, *arguments[2], fn);
		auto opts = SplineMapOptions::ParseFromValue(options_val, fn, bspline ? 4 : 3);
		if (opts.fractied) {
			bind->fractied = *opts.fractied;
		}
		if (opts.knots) {
			bind->placement = *opts.knots == "quantile" ? 1 : 0;
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(bind);
}

static unique_ptr<FunctionData> RcsKnotsBind(ClientContext &context, AggregateBindFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	return SplineKnotsBind(context, arguments, false);
}

static unique_ptr<FunctionData> BSplineKnotsBind(ClientContext &context, AggregateBindFunction &function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	return SplineKnotsBind(context, arguments, true);
}

static AggregateFunction SplineKnotsFunction(const char *name, vector<LogicalType> args, bool bspline) {
	return AggregateFunction(
	    name, std::move(args), LogicalType::LIST(LogicalType::DOUBLE), AggregateFunction::StateSize<SplineKnotsState>,
	    ANOFOX_AGG_INIT(SplineKnotsInitialize),
	    ANOFOX_GUARDED_UPDATE(SplineKnotsUpdate, SplineKnotsDestroy, SplineKnotsInitialize), SplineKnotsCombine,
	    ANOFOX_GUARDED_FINALIZE(SplineKnotsFinalize, SplineKnotsDestroy, SplineKnotsInitialize), nullptr,
	    bspline ? ANOFOX_AGG_BIND(BSplineKnotsBind) : ANOFOX_AGG_BIND(RcsKnotsBind), SplineKnotsDestroy);
}

static void RegisterKnotsAggregate(ExtensionLoader &loader, const char *name, bool bspline,
                                   const vector<string> &descriptions, const vector<string> &examples) {
	AggregateFunctionSet set(name);
	const vector<vector<LogicalType>> signatures = {
	    {LogicalType::DOUBLE}, {LogicalType::DOUBLE, LogicalType::BIGINT}, {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::ANY}};
	const vector<vector<string>> names = {
	    {"x"}, {"x", bspline ? "n_knots" : "nk"}, {"x", bspline ? "n_knots" : "nk", "options"}};
	for (auto &sig : signatures) {
		set.AddFunction(SplineKnotsFunction(name, sig, bspline));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	for (idx_t i = 0; i < signatures.size(); i++) {
		FunctionDescription d;
		d.description = descriptions[std::min<idx_t>(i, descriptions.size() - 1)];
		d.examples = {examples[i]};
		d.categories = {"regression", "splines"};
		d.parameter_names = names[i];
		d.parameter_types = signatures[i];
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

void RegisterSplineKnotsAggregateFunctions(ExtensionLoader &loader) {
	RegisterKnotsAggregate(
	    loader, RCS_KNOTS_AGG, false,
	    {"Default knots of a restricted cubic spline with nk knots (default 5) placed on x, as rms::rcs / "
	     "Hmisc::rcspline.eval(x, nk, knots.only = TRUE): the RMS Table 2.3 quantiles, outer knots at the 5th "
	     "smallest / largest value when n < 100. options: fractied (default 0.05)."},
	    {"rcs_knots_agg(x)", "rcs_knots_agg(x, 4)", "rcs_knots_agg(x, 4, {'fractied': 0.1})"});
	RegisterKnotsAggregate(loader, BSPLINE_KNOTS_AGG, true,
	                       {"Base knots of a B-spline placed on x (n_knots, default 5), as sklearn "
	                        "SplineTransformer: options knots 'uniform' (default) | 'quantile'."},
	                       {"bspline_knots_agg(x)", "bspline_knots_agg(x, 6)",
	                        "bspline_knots_agg(x, 6, {'knots': 'quantile'})"});
}

} // namespace duckdb
