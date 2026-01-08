#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

namespace duckdb {

//===--------------------------------------------------------------------===//
// Binomial Test Aggregate State
//===--------------------------------------------------------------------===//
struct BinomTestAggregateState {
    size_t successes;
    size_t trials;
    bool initialized;

    BinomTestAggregateState() : successes(0), trials(0), initialized(false) {}

    void Reset() {
        successes = 0;
        trials = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetBinomTestAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("estimate", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct BinomTestBindData : public FunctionData {
    double p0;
    AnofoxAlternative alternative;

    BinomTestBindData() : p0(0.5), alternative(ANOFOX_ALTERNATIVE_TWO_SIDED) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<BinomTestBindData>();
        copy->p0 = p0;
        copy->alternative = alternative;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<BinomTestBindData>();
        return p0 == other.p0 && alternative == other.alternative;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void BinomTestAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) BinomTestAggregateState();
}

static void BinomTestAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BinomTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~BinomTestAggregateState();
    }
}

static void BinomTestAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {
    UnifiedVectorFormat val_data;
    inputs[0].ToUnifiedFormat(count, val_data);
    auto vals = UnifiedVectorFormat::GetData<int64_t>(val_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BinomTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = val_data.sel->get_index(i);

        if (!val_data.validity.RowIsValid(val_idx)) {
            continue;
        }

        int64_t val = vals[val_idx];

        // Count successes (non-zero values treated as success)
        if (val != 0) {
            state.successes++;
        }
        state.trials++;
    }
}

static void BinomTestAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (BinomTestAggregateState **)source_data.data;
    auto targets = (BinomTestAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        target.successes += source.successes;
        target.trials += source.trials;
        target.initialized = true;
    }
}

static void BinomTestAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BinomTestAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<BinomTestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.trials < 1) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxPropTestResult binom_result;
        AnofoxError error;

        bool success = anofox_binom_test(state.successes, state.trials, bind_data.p0,
                                          bind_data.alternative, &binom_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = binom_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = binom_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = binom_result.estimate;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = binom_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = binom_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(binom_result.n);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, binom_result.method ? binom_result.method : "Exact binomial test");

        anofox_free_prop_test_result(&binom_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> BinomTestAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetBinomTestAggResultType();
    auto bind_data = make_uniq<BinomTestBindData>();

    if (arguments.size() >= 2 && arguments[1]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "p0") == 0 || strcasecmp(key, "p") == 0) {
                        bind_data->p0 = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "alternative") == 0) {
                        auto alt_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(alt_str.c_str(), "less") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_LESS;
                        } else if (strcasecmp(alt_str.c_str(), "greater") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_GREATER;
                        } else {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
                        }
                    }
                }
            }
        }
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("binom_test_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterBinomTestAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_binom_test_agg");

    // With options: (value BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_binom_test_agg", {LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<BinomTestAggregateState>, BinomTestAggInitialize,
        BinomTestAggUpdate, BinomTestAggCombine, BinomTestAggFinalize,
        nullptr, BinomTestAggBind, BinomTestAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (value BIGINT)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_binom_test_agg", {LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<BinomTestAggregateState>, BinomTestAggInitialize,
        BinomTestAggUpdate, BinomTestAggCombine, BinomTestAggFinalize,
        nullptr, BinomTestAggBind, BinomTestAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("binom_test_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
