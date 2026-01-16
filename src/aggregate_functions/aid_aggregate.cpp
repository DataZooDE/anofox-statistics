#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// AID Aggregate State - accumulates y values for each group
//===--------------------------------------------------------------------===//
struct AidAggregateState {
    vector<double> y_values;
    bool initialized;

    // Options
    double intermittent_threshold;
    AnofoxAidOutlierMethod outlier_method;

    AidAggregateState()
        : initialized(false), intermittent_threshold(0.3), outlier_method(ANOFOX_AID_OUTLIER_ZSCORE) {}

    void Reset() {
        y_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data for options
//===--------------------------------------------------------------------===//
struct AidAggregateBindData : public FunctionData {
    double intermittent_threshold = 0.3;
    AnofoxAidOutlierMethod outlier_method = ANOFOX_AID_OUTLIER_ZSCORE;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<AidAggregateBindData>();
        result->intermittent_threshold = intermittent_threshold;
        result->outlier_method = outlier_method;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<AidAggregateBindData>();
        return intermittent_threshold == other.intermittent_threshold && outlier_method == other.outlier_method;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition for AID classification
//===--------------------------------------------------------------------===//
static LogicalType GetAidAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("demand_type", LogicalType::VARCHAR));
    children.push_back(make_pair("is_intermittent", LogicalType::BOOLEAN));
    children.push_back(make_pair("distribution", LogicalType::VARCHAR));
    children.push_back(make_pair("mean", LogicalType::DOUBLE));
    children.push_back(make_pair("variance", LogicalType::DOUBLE));
    children.push_back(make_pair("zero_proportion", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("has_stockouts", LogicalType::BOOLEAN));
    children.push_back(make_pair("is_new_product", LogicalType::BOOLEAN));
    children.push_back(make_pair("is_obsolete_product", LogicalType::BOOLEAN));
    children.push_back(make_pair("stockout_count", LogicalType::BIGINT));
    children.push_back(make_pair("new_product_count", LogicalType::BIGINT));
    children.push_back(make_pair("obsolete_product_count", LogicalType::BIGINT));
    children.push_back(make_pair("high_outlier_count", LogicalType::BIGINT));
    children.push_back(make_pair("low_outlier_count", LogicalType::BIGINT));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Result type definition for AID anomaly detection
//===--------------------------------------------------------------------===//
static LogicalType GetAidAnomalyFlagsType() {
    child_list_t<LogicalType> flags_children;
    flags_children.push_back(make_pair("stockout", LogicalType::BOOLEAN));
    flags_children.push_back(make_pair("new_product", LogicalType::BOOLEAN));
    flags_children.push_back(make_pair("obsolete_product", LogicalType::BOOLEAN));
    flags_children.push_back(make_pair("high_outlier", LogicalType::BOOLEAN));
    flags_children.push_back(make_pair("low_outlier", LogicalType::BOOLEAN));
    return LogicalType::STRUCT(std::move(flags_children));
}

static LogicalType GetAidAnomalyResultType() {
    return LogicalType::LIST(GetAidAnomalyFlagsType());
}

//===--------------------------------------------------------------------===//
// AID Classification Aggregate function operations
//===--------------------------------------------------------------------===//

static void AidAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) AidAggregateState();
}

static void AidAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AidAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~AidAggregateState();
    }
}

static void AidAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<AidAggregateBindData>();

    UnifiedVectorFormat y_data;
    inputs[0].ToUnifiedFormat(count, y_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AidAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.intermittent_threshold = bind_data.intermittent_threshold;
        state.outlier_method = bind_data.outlier_method;
        state.initialized = true;

        auto y_idx = y_data.sel->get_index(i);
        if (!y_data.validity.RowIsValid(y_idx)) {
            // Add NaN for invalid rows to preserve ordering
            state.y_values.push_back(std::nan(""));
            continue;
        }
        double y_val = y_values[y_idx];
        state.y_values.push_back(y_val);
    }
}

static void AidAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (AidAggregateState **)source_data.data;
    auto targets = (AidAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_values = std::move(source.y_values);
            target.initialized = true;
            target.intermittent_threshold = source.intermittent_threshold;
            target.outlier_method = source.outlier_method;
            continue;
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
    }
}

static void AidAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AidAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.empty()) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxAidOptions options;
        options.intermittent_threshold = state.intermittent_threshold;
        options.outlier_method = state.outlier_method;

        AnofoxAidResult aid_result;
        AnofoxError error;

        bool success = anofox_aid(y_array, options, &aid_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;

        // demand_type (VARCHAR)
        FlatVector::GetData<string_t>(*struct_entries[struct_idx])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx], aid_result.demand_type);
        struct_idx++;

        // is_intermittent (BOOLEAN)
        FlatVector::GetData<bool>(*struct_entries[struct_idx++])[result_idx] = aid_result.is_intermittent;

        // distribution (VARCHAR)
        FlatVector::GetData<string_t>(*struct_entries[struct_idx])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx], aid_result.distribution);
        struct_idx++;

        // Statistics
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = aid_result.mean;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = aid_result.variance;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = aid_result.zero_proportion;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = aid_result.n_observations;

        // Pattern flags
        FlatVector::GetData<bool>(*struct_entries[struct_idx++])[result_idx] = aid_result.has_stockouts;
        FlatVector::GetData<bool>(*struct_entries[struct_idx++])[result_idx] = aid_result.is_new_product;
        FlatVector::GetData<bool>(*struct_entries[struct_idx++])[result_idx] = aid_result.is_obsolete_product;

        // Counts
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = aid_result.stockout_count;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = aid_result.new_product_count;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = aid_result.obsolete_product_count;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = aid_result.high_outlier_count;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = aid_result.low_outlier_count;

        anofox_free_aid_result(&aid_result);

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// AID Anomaly Aggregate function operations
//===--------------------------------------------------------------------===//

static void AidAnomalyAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AidAggregateState **)sdata.data;

    auto &list_child = ListVector::GetEntry(result);
    auto &struct_entries = StructVector::GetEntries(list_child);
    auto list_data = ListVector::GetData(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.empty()) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxAidOptions options;
        options.intermittent_threshold = state.intermittent_threshold;
        options.outlier_method = state.outlier_method;

        AnofoxAidAnomalyResult anomaly_result;
        AnofoxError error;

        bool success = anofox_aid_anomaly(y_array, options, &anomaly_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Set list entry - must Reserve before SetListSize to allocate child vector capacity
        auto list_offset = ListVector::GetListSize(result);
        size_t n = anomaly_result.len;
        list_data[result_idx].offset = list_offset;
        list_data[result_idx].length = (idx_t)n;
        ListVector::Reserve(result, list_offset + n);
        ListVector::SetListSize(result, list_offset + n);

        // Copy anomaly flags to struct list
        auto stockout_data = FlatVector::GetData<bool>(*struct_entries[0]);
        auto new_product_data = FlatVector::GetData<bool>(*struct_entries[1]);
        auto obsolete_product_data = FlatVector::GetData<bool>(*struct_entries[2]);
        auto high_outlier_data = FlatVector::GetData<bool>(*struct_entries[3]);
        auto low_outlier_data = FlatVector::GetData<bool>(*struct_entries[4]);

        for (size_t j = 0; j < n; j++) {
            stockout_data[list_offset + j] = anomaly_result.flags[j].stockout;
            new_product_data[list_offset + j] = anomaly_result.flags[j].new_product;
            obsolete_product_data[list_offset + j] = anomaly_result.flags[j].obsolete_product;
            high_outlier_data[list_offset + j] = anomaly_result.flags[j].high_outlier;
            low_outlier_data[list_offset + j] = anomaly_result.flags[j].low_outlier;
        }

        anofox_free_aid_anomaly_result(&anomaly_result);

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind functions
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> AidAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<AidAggregateBindData>();

    if (arguments.size() >= 2 && arguments[1]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[1]);
        if (opts.intermittent_threshold.has_value()) {
            result->intermittent_threshold = opts.intermittent_threshold.value();
        }
        if (opts.outlier_method.has_value()) {
            result->outlier_method = opts.outlier_method.value() == AidOutlierMethod::ZSCORE
                                         ? ANOFOX_AID_OUTLIER_ZSCORE
                                         : ANOFOX_AID_OUTLIER_IQR;
        }
    }

    function.return_type = GetAidAggResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("aid_agg");
    return std::move(result);
}

static unique_ptr<FunctionData> AidAnomalyAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<AidAggregateBindData>();

    if (arguments.size() >= 2 && arguments[1]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[1]);
        if (opts.intermittent_threshold.has_value()) {
            result->intermittent_threshold = opts.intermittent_threshold.value();
        }
        if (opts.outlier_method.has_value()) {
            result->outlier_method = opts.outlier_method.value() == AidOutlierMethod::ZSCORE
                                         ? ANOFOX_AID_OUTLIER_ZSCORE
                                         : ANOFOX_AID_OUTLIER_IQR;
        }
    }

    function.return_type = GetAidAnomalyResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("aid_anomaly_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterAidAggregateFunction(ExtensionLoader &loader) {
    // AID Classification
    AggregateFunctionSet aid_set("anofox_stats_aid_agg");

    auto aid_basic =
        AggregateFunction("anofox_stats_aid_agg", {LogicalType::DOUBLE}, LogicalType::ANY,
                          AggregateFunction::StateSize<AidAggregateState>, AidAggInitialize, AidAggUpdate, AidAggCombine,
                          AidAggFinalize, nullptr, AidAggBind, AidAggDestroy);
    aid_set.AddFunction(aid_basic);

    auto aid_map = AggregateFunction("anofox_stats_aid_agg", {LogicalType::DOUBLE, LogicalType::ANY}, LogicalType::ANY,
                                     AggregateFunction::StateSize<AidAggregateState>, AidAggInitialize, AidAggUpdate,
                                     AidAggCombine, AidAggFinalize, nullptr, AidAggBind, AidAggDestroy);
    aid_set.AddFunction(aid_map);

    loader.RegisterFunction(aid_set);

    // Short alias for AID
    AggregateFunctionSet aid_alias("aid_agg");
    aid_alias.AddFunction(aid_basic);
    aid_alias.AddFunction(aid_map);
    loader.RegisterFunction(aid_alias);

    // AID Anomaly Detection
    AggregateFunctionSet aid_anomaly_set("anofox_stats_aid_anomaly_agg");

    auto aid_anomaly_basic = AggregateFunction(
        "anofox_stats_aid_anomaly_agg", {LogicalType::DOUBLE}, LogicalType::ANY,
        AggregateFunction::StateSize<AidAggregateState>, AidAggInitialize, AidAggUpdate, AidAggCombine,
        AidAnomalyAggFinalize, nullptr, AidAnomalyAggBind, AidAggDestroy);
    aid_anomaly_set.AddFunction(aid_anomaly_basic);

    auto aid_anomaly_map = AggregateFunction(
        "anofox_stats_aid_anomaly_agg", {LogicalType::DOUBLE, LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<AidAggregateState>, AidAggInitialize, AidAggUpdate, AidAggCombine,
        AidAnomalyAggFinalize, nullptr, AidAnomalyAggBind, AidAggDestroy);
    aid_anomaly_set.AddFunction(aid_anomaly_map);

    loader.RegisterFunction(aid_anomaly_set);

    // Short alias for AID anomaly
    AggregateFunctionSet aid_anomaly_alias("aid_anomaly_agg");
    aid_anomaly_alias.AddFunction(aid_anomaly_basic);
    aid_anomaly_alias.AddFunction(aid_anomaly_map);
    loader.RegisterFunction(aid_anomaly_alias);
}

} // namespace duckdb
