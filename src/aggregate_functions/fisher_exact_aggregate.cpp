#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Fisher Exact Test Aggregate State
// Collects 2x2 contingency table counts from row_var and col_var columns
//===--------------------------------------------------------------------===//
struct FisherExactAggregateState {
    // Counts for 2x2 table: a=cell(0,0), b=cell(0,1), c=cell(1,0), d=cell(1,1)
    size_t a, b, c, d;
    bool initialized;

    FisherExactAggregateState() : a(0), b(0), c(0), d(0), initialized(false) {}

    void Reset() {
        a = b = c = d = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetFisherExactAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("odds_ratio", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct FisherExactBindData : public FunctionData {
    FisherExactMapOptions options;

    FisherExactBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<FisherExactBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<FisherExactBindData>();
        return options.alternative == other.options.alternative;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void FisherExactAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) FisherExactAggregateState();
}

static void FisherExactAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (FisherExactAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~FisherExactAggregateState();
    }
}

static void FisherExactAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat row_data, col_data;
    inputs[0].ToUnifiedFormat(count, row_data);
    inputs[1].ToUnifiedFormat(count, col_data);
    auto row_values = UnifiedVectorFormat::GetData<int32_t>(row_data);
    auto col_values = UnifiedVectorFormat::GetData<int32_t>(col_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (FisherExactAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto row_idx = row_data.sel->get_index(i);
        auto col_idx = col_data.sel->get_index(i);

        if (!row_data.validity.RowIsValid(row_idx) || !col_data.validity.RowIsValid(col_idx)) {
            continue;
        }

        int32_t row = row_values[row_idx];
        int32_t col = col_values[col_idx];

        // Build 2x2 contingency table (expecting binary 0/1 values)
        if (row == 0 && col == 0) {
            state.a++;
        } else if (row == 0 && col == 1) {
            state.b++;
        } else if (row == 1 && col == 0) {
            state.c++;
        } else if (row == 1 && col == 1) {
            state.d++;
        }
    }
}

static void FisherExactAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (FisherExactAggregateState **)source_data.data;
    auto targets = (FisherExactAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.a = source.a;
            target.b = source.b;
            target.c = source.c;
            target.d = source.d;
            target.initialized = true;
            continue;
        }

        target.a += source.a;
        target.b += source.b;
        target.c += source.c;
        target.d += source.d;
    }
}

static void FisherExactAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (FisherExactAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<FisherExactBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        size_t n = state.a + state.b + state.c + state.d;
        if (!state.initialized || n < 4) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxFisherExactOptions options;
        options.alternative = bind_data.options.alternative.value_or(Alternative::TWO_SIDED) == Alternative::TWO_SIDED
                                  ? ANOFOX_ALTERNATIVE_TWO_SIDED
                                  : (bind_data.options.alternative.value_or(Alternative::TWO_SIDED) == Alternative::LESS
                                         ? ANOFOX_ALTERNATIVE_LESS
                                         : ANOFOX_ALTERNATIVE_GREATER);
        options.confidence_level = 0.95;

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_fisher_exact(state.a, state.b, state.c, state.d, options, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic; // odds ratio
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.effect_size; // odds ratio
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Fisher's Exact Test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> FisherExactAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetFisherExactAggResultType();
    auto bind_data = make_uniq<FisherExactBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        bind_data->options = FisherExactMapOptions::ParseFromValue(options_val);
    }

    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterFisherExactAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_fisher_exact_agg");

    // With options
    auto func_with_opts = AggregateFunction(
        "anofox_stats_fisher_exact_agg", {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<FisherExactAggregateState>, FisherExactAggInitialize,
        FisherExactAggUpdate, FisherExactAggCombine, FisherExactAggFinalize,
        nullptr, FisherExactAggBind, FisherExactAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options
    auto func_no_opts = AggregateFunction(
        "anofox_stats_fisher_exact_agg", {LogicalType::INTEGER, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<FisherExactAggregateState>, FisherExactAggInitialize,
        FisherExactAggUpdate, FisherExactAggCombine, FisherExactAggFinalize,
        nullptr, FisherExactAggBind, FisherExactAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("fisher_exact_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
