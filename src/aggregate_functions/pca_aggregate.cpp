// Principal component analysis (#303): pca_agg, pca_transform and pca_by.
// The analysis is upstream anofox-statistics multivariate::pca / Pca::transform
// (R's prcomp(x, center = TRUE, scale. = FALSE) and predict.prcomp); this file
// only buffers rows, marshals the model STRUCT and scores rows with it.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/canonical_order.hpp"
#include "anofox_statistics_banner.hpp"
#include "anofox_statistics_extension.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// pca_agg(x DOUBLE[] [, options])
//===--------------------------------------------------------------------===//
struct PcaAggregateState {
	vector<vector<double>> x_columns;
	idx_t n_features;
	bool initialized;

	PcaAggregateState() : n_features(0), initialized(false) {
	}

	void Reset() {
		x_columns.clear();
		n_features = 0;
		initialized = false;
	}
};

struct PcaBindData : public FunctionData {
	//! Components kept; 0 = all.
	idx_t n_components = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PcaBindData>();
		copy->n_components = n_components;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		return n_components == other_p.Cast<PcaBindData>().n_components;
	}
};

static LogicalType GetPcaAggResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("sdev", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("proportion_of_variance", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("loadings", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))));
	children.push_back(make_pair("center", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("n_components", LogicalType::BIGINT));
	children.push_back(make_pair("n_features", LogicalType::BIGINT));
	children.push_back(make_pair("n_observations", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(children));
}

static void PcaAggInitialize(data_ptr_t state_p) {
	new (state_p) PcaAggregateState();
}

static void PcaAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PcaAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.~PcaAggregateState();
	}
}

static void PcaAggUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat x_data;
	inputs[0].ToUnifiedFormat(count, x_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[0]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PcaAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx)) {
			continue; // NULL row
		}
		auto entry = x_list_data[x_idx];
		// A row with a NULL, NaN or infinite element is skipped like a NULL row.
		if (ListHasNullElement(x_child_validity, entry)) {
			continue;
		}
		bool finite = true;
		for (idx_t j = 0; j < entry.length; j++) {
			finite = finite && std::isfinite(x_child_data[entry.offset + j]);
		}
		if (!finite) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(entry.length);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("pca_agg: inconsistent feature count: expected %llu, got %llu",
			                            (unsigned long long)state.n_features, (unsigned long long)entry.length);
		}
		for (idx_t j = 0; j < entry.length; j++) {
			state.x_columns[j].push_back(x_child_data[entry.offset + j]);
		}
	}
}

static void PcaAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                          idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (PcaAggregateState **)source_data.data;
	auto targets = (PcaAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("pca_agg: inconsistent feature count: expected %llu, got %llu",
			                            (unsigned long long)target.n_features,
			                            (unsigned long long)source.n_features);
		}
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

static void AppendDoubleList(Vector &list_vec, idx_t row, const double *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	auto child_data = FlatDataMut<double>(ListVector::GetEntry(list_vec));
	for (idx_t i = 0; i < len; i++) {
		child_data[offset + i] = data[i];
	}
	ListVector::SetListSize(list_vec, offset + len);
	ListVector::GetData(list_vec)[row] = {offset, len};
}

static void PcaAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PcaAggregateState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<PcaBindData>();
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.x_columns.empty() || state.x_columns[0].size() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Keys(state.x_columns).Sort();
		idx_t n = state.x_columns[0].size();
		idx_t d = state.n_features;
		vector<double> data(n * d);
		for (idx_t r = 0; r < n; r++) {
			for (idx_t j = 0; j < d; j++) {
				data[r * d + j] = state.x_columns[j][r];
			}
		}

		AnofoxPcaResult fit;
		AnofoxError error;
		if (!anofox_pca_fit(data.data(), n, d, bind.n_components, &fit, &error)) {
			ThrowUnlessDegenerate("pca_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = fit.n_components;
		AppendDoubleList(StructEntry(entries[0]), row, fit.sdev, k);
		AppendDoubleList(StructEntry(entries[1]), row, fit.proportion_of_variance, k);
		// loadings: one list of n_features loadings per component
		auto &loadings = StructEntry(entries[2]);
		auto outer_offset = ListVector::GetListSize(loadings);
		ListVector::Reserve(loadings, outer_offset + k);
		auto &inner = ListVector::GetEntry(loadings);
		for (idx_t j = 0; j < k; j++) {
			AppendDoubleList(inner, outer_offset + j, fit.loadings + j * fit.n_features, fit.n_features);
		}
		ListVector::SetListSize(loadings, outer_offset + k);
		ListVector::GetData(loadings)[row] = {outer_offset, k};
		AppendDoubleList(StructEntry(entries[3]), row, fit.center, fit.n_features);
		FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = static_cast<int64_t>(k);
		FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = static_cast<int64_t>(fit.n_features);
		FlatDataMut<int64_t>(StructEntry(entries[6]))[row] = static_cast<int64_t>(fit.n_observations);
		anofox_free_pca_result(&fit);
		state.Reset();
	}
}

static unique_ptr<FunctionData> PcaAggBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<PcaBindData>();
	if (arguments.size() >= 2) {
		auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[1], "pca_agg", {"n_components"});
		if (opts.n_components.has_value()) {
			if (opts.n_components.value() == 0) {
				throw InvalidInputException("pca_agg: option 'n_components' must be at least 1");
			}
			bind->n_components = opts.n_components.value();
		}
	}
	SetReturnTypeOf(function, GetPcaAggResultType());
	PostHogTelemetry::Instance().RecordFunctionCall("pca_agg");
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// pca_transform(model, x DOUBLE[]) -> DOUBLE[]
//===--------------------------------------------------------------------===//
struct PcaTransformBindData : public FunctionData {
	idx_t center_idx = DConstants::INVALID_INDEX;
	idx_t sdev_idx = DConstants::INVALID_INDEX;
	idx_t loadings_idx = DConstants::INVALID_INDEX;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PcaTransformBindData>();
		copy->center_idx = center_idx;
		copy->sdev_idx = sdev_idx;
		copy->loadings_idx = loadings_idx;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<PcaTransformBindData>();
		return center_idx == other.center_idx && sdev_idx == other.sdev_idx && loadings_idx == other.loadings_idx;
	}
};

static unique_ptr<FunctionData> PcaTransformBind(ClientContext &context, ScalarBindFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	auto &in_type = ReturnTypeOf(*arguments[0]);
	if (in_type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("pca_transform(model, x): model must be the STRUCT returned by pca_agg, got %s",
		                            in_type.ToString());
	}
	auto data = make_uniq<PcaTransformBindData>();
	// Cast the model's fields to the pca_agg types (a hand-written literal may hold DECIMALs).
	child_list_t<LogicalType> cast_children;
	auto &children = StructType::GetChildTypes(in_type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = children[i].first;
		auto type = children[i].second;
		if (name == "center") {
			data->center_idx = i;
			type = LogicalType::LIST(LogicalType::DOUBLE);
		} else if (name == "sdev") {
			data->sdev_idx = i;
			type = LogicalType::LIST(LogicalType::DOUBLE);
		} else if (name == "loadings") {
			data->loadings_idx = i;
			type = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
		}
		cast_children.push_back(MakeChild(name, type));
	}
	if (data->center_idx == DConstants::INVALID_INDEX || data->sdev_idx == DConstants::INVALID_INDEX ||
	    data->loadings_idx == DConstants::INVALID_INDEX) {
		throw InvalidInputException("pca_transform(model, x): model must have the fields center, sdev and loadings "
		                            "(the STRUCT returned by pca_agg), got %s",
		                            in_type.ToString());
	}
	auto target = LogicalType::STRUCT(std::move(cast_children));
	if (target != in_type) {
		arguments[0] = BoundCastExpression::AddCastToType(context, std::move(arguments[0]), target);
	}
	SetReturnTypeOf(bound_function, LogicalType::LIST(LogicalType::DOUBLE));
	PostHogTelemetry::Instance().RecordFunctionCall("pca_transform");
	return std::move(data);
}

static void PcaTransformFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = BindInfoOf(func_expr)->Cast<PcaTransformBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &model_vec = args.data[0];
	auto &x_vec = args.data[1];
	auto &entries = StructVector::GetEntries(model_vec);
	auto &center_vec = StructEntry(entries[bind.center_idx]);
	auto &sdev_vec = StructEntry(entries[bind.sdev_idx]);
	auto &loadings_vec = StructEntry(entries[bind.loadings_idx]);
	auto &center_child = ListVector::GetEntry(center_vec);
	auto &sdev_child = ListVector::GetEntry(sdev_vec);
	auto &loadings_inner = ListVector::GetEntry(loadings_vec);
	auto &loadings_child = ListVector::GetEntry(loadings_inner);
	auto x_lists = FlatDataMut<list_entry_t>(x_vec);
	auto &x_child = ListVector::GetEntry(x_vec);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);

	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(model_vec, row) || FlatVector::IsNull(x_vec, row) ||
		    FlatVector::IsNull(center_vec, row) || FlatVector::IsNull(sdev_vec, row) ||
		    FlatVector::IsNull(loadings_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto x_entry = x_lists[row];
		if (ListHasNullElement(x_child_validity, x_entry)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto c = FlatDataMut<list_entry_t>(center_vec)[row];
		auto s = FlatDataMut<list_entry_t>(sdev_vec)[row];
		auto l = FlatDataMut<list_entry_t>(loadings_vec)[row];
		idx_t d = c.length;
		idx_t k = s.length;
		if (l.length != k) {
			throw InvalidInputException("pca_transform: the model has %llu sdev values but %llu loading vectors",
			                            (unsigned long long)k, (unsigned long long)l.length);
		}
		if (x_entry.length != d) {
			throw InvalidInputException("pca_transform: the model has %llu features but x has %llu values",
			                            (unsigned long long)d, (unsigned long long)x_entry.length);
		}
		if (ListHasNullElement(FlatValidityMut(center_child), c) || ListHasNullElement(FlatValidityMut(sdev_child), s) ||
		    ListHasNullElement(FlatValidityMut(loadings_inner), l)) {
			throw InvalidInputException("pca_transform: the model's center, sdev and loadings must not hold NULLs");
		}
		vector<double> loadings(k * d);
		auto inner_lists = FlatDataMut<list_entry_t>(loadings_inner);
		auto loadings_data = FlatDataMut<double>(loadings_child);
		for (idx_t j = 0; j < k; j++) {
			auto e = inner_lists[l.offset + j];
			if (e.length != d || ListHasNullElement(FlatValidityMut(loadings_child), e)) {
				throw InvalidInputException("pca_transform: every loading vector must hold %llu non-NULL values",
				                            (unsigned long long)d);
			}
			for (idx_t i = 0; i < d; i++) {
				loadings[j * d + i] = loadings_data[e.offset + i];
			}
		}
		vector<double> scores(k);
		AnofoxError error;
		if (!anofox_pca_transform(FlatDataMut<double>(center_child) + c.offset,
		                          FlatDataMut<double>(sdev_child) + s.offset, loadings.data(), d, k,
		                          x_child_data + x_entry.offset, x_entry.length, scores.data(), &error)) {
			ThrowUnlessDegenerate("pca_transform", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AppendDoubleList(result, row, scores.data(), k);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding asserts this in debug builds).
static void PcaTransformFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	PcaTransformFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// pca_by(source, group_col, x_cols [, options]): scores per row
//===--------------------------------------------------------------------===//
static const char *const PCA_BY_SQL = R"(
SELECT
    * EXCLUDE (_anofox_pca, _anofox_rid),
    pca_transform(_anofox_pca, x_cols) AS scores
FROM (
    SELECT *,
        pca_agg(x_cols, options) OVER (PARTITION BY group_col) AS _anofox_pca
    FROM (SELECT *, row_number() OVER () AS _anofox_rid FROM query_table(source::VARCHAR))
) sub
ORDER BY group_col, _anofox_rid
)";

void RegisterPcaFunctions(ExtensionLoader &loader) {
	{
		AggregateFunctionSet func_set("pca_agg");
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::LIST(LogicalType::DOUBLE)};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			func_set.AddFunction(AggregateFunction(
			    "pca_agg", args, LogicalType::ANY, AggregateFunction::StateSize<PcaAggregateState>,
			    ANOFOX_AGG_INIT(PcaAggInitialize), ANOFOX_GUARDED_UPDATE(PcaAggUpdate, PcaAggDestroy, PcaAggInitialize),
			    PcaAggCombine, ANOFOX_GUARDED_FINALIZE(PcaAggFinalize, PcaAggDestroy, PcaAggInitialize), nullptr,
			    ANOFOX_AGG_BIND(PcaAggBind), PcaAggDestroy));
		}
		CreateAggregateFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d1;
		d1.description = "Principal component analysis of a column of feature vectors (R's prcomp(x, center = TRUE, "
		                 "scale. = FALSE)): returns STRUCT(sdev, proportion_of_variance, loadings (one list of "
		                 "n_features loadings per component), center, n_components, n_features, n_observations). "
		                 "Each component's sign is fixed so its largest absolute loading is positive. Rows with a "
		                 "NULL or non-finite element are skipped.";
		d1.examples = {"pca_agg([x1, x2, x3])"};
		d1.categories = {"multivariate"};
		d1.parameter_names = {"x"};
		d1.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE)};
		info.descriptions.push_back(std::move(d1));
		FunctionDescription d2;
		d2.description = "Principal component analysis of a column of feature vectors with options: n_components "
		                 "keeps the first components (default all, min(n - 1, n_features)).";
		d2.examples = {"pca_agg([x1, x2, x3], {'n_components': 2})"};
		d2.categories = {"multivariate"};
		d2.parameter_names = {"x", "options"};
		d2.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
		info.descriptions.push_back(std::move(d2));
		loader.RegisterFunction(std::move(info));
	}
	{
		ScalarFunction func("pca_transform", {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE)},
		                    LogicalType::LIST(LogicalType::DOUBLE),
		                    DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, PcaTransformFunction),
		                    ANOFOX_SCALAR_BIND(PcaTransformBind));
		MarkFallible(func);
		SetNullHandlingOf(func, FunctionNullHandling::SPECIAL_HANDLING);
		ScalarFunctionSet func_set("pca_transform");
		func_set.AddFunction(func);
		CreateScalarFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription desc;
		desc.description = "Scores of one feature vector on the components of a pca_agg model (R's predict.prcomp): "
		                   "the row minus the model's center, times each component's loadings. NULL when the model "
		                   "or x is NULL or x holds a NULL.";
		desc.examples = {"pca_transform(pca_agg([x1, x2]), [1.5, 2.0])"};
		desc.categories = {"multivariate"};
		desc.parameter_names = {"model", "x"};
		desc.parameter_types = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE)};
		info.descriptions.push_back(std::move(desc));
		loader.RegisterFunction(std::move(info));
	}
	{
		auto macro = CreateContractTableMacro(
		    "pca_by", {"source", "group_col", "x_cols"}, {{"options", "NULL"}}, PCA_BY_SQL,
		    "Table macro: principal component analysis per group of source (pca_agg over the group's rows of "
		    "x_cols, options as pca_agg) and returns every source row with scores (DOUBLE[]: the row's scores on "
		    "the kept components, pca_transform; NULL for rows with a NULL element) appended.",
		    "pca_by('measurements', batch, [x1, x2, x3], options := {'n_components': 2})",
		    {"multivariate", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

} // namespace duckdb
