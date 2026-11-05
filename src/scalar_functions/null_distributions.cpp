#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/duckdb_compat.hpp"
#include "../include/error_dispatch.hpp"
#include "telemetry.hpp"

namespace duckdb {

// Null distributions (#304):
//   auc_null_quantile(p, n_pos, n_neg)  -> DOUBLE
//   auc_null_p_value(auc, n_pos, n_neg) -> DOUBLE
//   ptukey(q, n_means, df [, n_ranges [, lower_tail]]) -> DOUBLE
//   qtukey(p, n_means, df [, n_ranges [, lower_tail]]) -> DOUBLE
// A NULL argument gives NULL; invalid arguments raise (as aic / bic).

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
template <void (*IMPL)(DataChunk &, ExpressionState &, Vector &)>
static void ConstantAware(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	IMPL(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

using AucNullFfi = bool (*)(double, int64_t, int64_t, double *, AnofoxError *);

template <AucNullFfi FFI>
static void AucNullImpl(DataChunk &args, ExpressionState &state, Vector &result, const char *name) {
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	idx_t count = args.size();

	UnifiedVectorFormat x_data, pos_data, neg_data;
	args.data[0].ToUnifiedFormat(count, x_data);
	args.data[1].ToUnifiedFormat(count, pos_data);
	args.data[2].ToUnifiedFormat(count, neg_data);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	auto pos_values = UnifiedVectorFormat::GetData<int64_t>(pos_data);
	auto neg_values = UnifiedVectorFormat::GetData<int64_t>(neg_data);

	auto result_data = FlatDataMut<double>(result);
	for (idx_t row = 0; row < count; row++) {
		auto x_idx = x_data.sel->get_index(row);
		auto pos_idx = pos_data.sel->get_index(row);
		auto neg_idx = neg_data.sel->get_index(row);
		if (!x_data.validity.RowIsValid(x_idx) || !pos_data.validity.RowIsValid(pos_idx) ||
		    !neg_data.validity.RowIsValid(neg_idx)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		double value;
		AnofoxError error;
		if (!FFI(x_values[x_idx], pos_values[pos_idx], neg_values[neg_idx], &value, &error)) {
			ThrowFromFfiError(name, error);
		}
		result_data[row] = value;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

static void AucNullQuantileImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	AucNullImpl<anofox_auc_null_quantile>(args, state, result, "auc_null_quantile");
}

static void AucNullPValueImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	AucNullImpl<anofox_auc_null_p_value>(args, state, result, "auc_null_p_value");
}

using TukeyFfi = bool (*)(double, double, double, double, bool, double *, AnofoxError *);

// Arguments: x, n_means, df, then optional n_ranges (default 1) and lower_tail
// (default true), as R's ptukey / qtukey.
template <TukeyFfi FFI>
static void TukeyImpl(DataChunk &args, ExpressionState &state, Vector &result, const char *name) {
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	idx_t count = args.size();
	const idx_t n_args = args.ColumnCount();

	UnifiedVectorFormat data[5];
	for (idx_t i = 0; i < n_args; i++) {
		args.data[i].ToUnifiedFormat(count, data[i]);
	}

	auto result_data = FlatDataMut<double>(result);
	for (idx_t row = 0; row < count; row++) {
		double values[4] = {0.0, 0.0, 0.0, 1.0};
		bool lower_tail = true;
		bool any_null = false;
		for (idx_t i = 0; i < n_args; i++) {
			auto idx = data[i].sel->get_index(row);
			if (!data[i].validity.RowIsValid(idx)) {
				any_null = true;
				break;
			}
			if (i < 4) {
				values[i] = UnifiedVectorFormat::GetData<double>(data[i])[idx];
			} else {
				lower_tail = UnifiedVectorFormat::GetData<bool>(data[i])[idx];
			}
		}
		if (any_null) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		double value;
		AnofoxError error;
		if (!FFI(values[0], values[1], values[2], values[3], lower_tail, &value, &error)) {
			ThrowFromFfiError(name, error);
		}
		result_data[row] = value;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

static void PtukeyImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	TukeyImpl<anofox_ptukey>(args, state, result, "ptukey");
}

static void QtukeyImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	TukeyImpl<anofox_qtukey>(args, state, result, "qtukey");
}

static void RegisterScalar(ExtensionLoader &loader, const string &name, vector<ScalarFunction> overloads,
                           const string &description, const string &example, const vector<string> &parameter_names,
                           const vector<LogicalType> &parameter_types) {
	ScalarFunctionSet set(AsDuckName(name));
	for (auto &f : overloads) {
		MarkFallible(f);
		set.AddFunction(f);
	}
	CreateScalarFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription desc;
	desc.description = description;
	desc.examples = {example};
	desc.categories = {"hypothesis-testing"};
	desc.parameter_names = parameter_names;
	desc.parameter_types = parameter_types;
	info.descriptions.push_back(std::move(desc));
	loader.RegisterFunction(std::move(info));
}

void RegisterNullDistributionFunctions(ExtensionLoader &loader) {
	const auto D = LogicalType::DOUBLE;
	const auto I = LogicalType::BIGINT;
	const auto B = LogicalType::BOOLEAN;

	RegisterScalar(loader, "auc_null_quantile",
	               {ScalarFunction({D, I, I}, D, ConstantAware<AucNullQuantileImpl>)},
	               "Quantile of the ROC AUC of a random classifier with n_pos positive and n_neg negative cases: the "
	               "smallest AUC a with P(AUC <= a) >= p, exact from the Mann-Whitney distribution "
	               "(R qwilcox(p, n_pos, n_neg) / (n_pos * n_neg)).",
	               "auc_null_quantile(0.95, 10, 10)", {"p", "n_pos", "n_neg"}, {D, I, I});

	RegisterScalar(loader, "auc_null_p_value",
	               {ScalarFunction({D, I, I}, D, ConstantAware<AucNullPValueImpl>)},
	               "One-sided p-value P(AUC_random >= auc) of an observed ROC AUC against random scores, exact from "
	               "the Mann-Whitney distribution.",
	               "auc_null_p_value(0.8, 10, 10)", {"auc", "n_pos", "n_neg"}, {D, I, I});

	RegisterScalar(loader, "ptukey",
	               {ScalarFunction({D, D, D}, D, ConstantAware<PtukeyImpl>),
	                ScalarFunction({D, D, D, D}, D, ConstantAware<PtukeyImpl>),
	                ScalarFunction({D, D, D, D, B}, D, ConstantAware<PtukeyImpl>)},
	               "Distribution function of the studentized range statistic (R ptukey(q, nmeans, df, nranges = 1, "
	               "lower.tail = TRUE)).",
	               "ptukey(3.5, 3, 12)", {"q", "n_means", "df", "n_ranges", "lower_tail"}, {D, D, D, D, B});

	RegisterScalar(loader, "qtukey",
	               {ScalarFunction({D, D, D}, D, ConstantAware<QtukeyImpl>),
	                ScalarFunction({D, D, D, D}, D, ConstantAware<QtukeyImpl>),
	                ScalarFunction({D, D, D, D, B}, D, ConstantAware<QtukeyImpl>)},
	               "Quantile function of the studentized range statistic (R qtukey(p, nmeans, df, nranges = 1, "
	               "lower.tail = TRUE)).",
	               "qtukey(0.95, 3, 12)", {"p", "n_means", "df", "n_ranges", "lower_tail"}, {D, D, D, D, B});
}

} // namespace duckdb
