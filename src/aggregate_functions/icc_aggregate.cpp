#include <vector>
#include <map>

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
// ICC Aggregate State
//===--------------------------------------------------------------------===//
struct IccAggregateState {
    vector<double> values;
    vector<int64_t> subject_ids;
    vector<int64_t> rater_ids;
    bool initialized;

    IccAggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        subject_ids.clear();
        rater_ids.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetIccAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("icc", LogicalType::DOUBLE));
    children.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n_subjects", LogicalType::BIGINT));
    children.push_back(make_pair("n_raters", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct IccBindData : public FunctionData {
    AnofoxIccType icc_type;

    IccBindData() : icc_type(ANOFOX_ICC_SINGLE) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<IccBindData>();
        copy->icc_type = icc_type;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<IccBindData>();
        return icc_type == other.icc_type;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void IccAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) IccAggregateState();
}

static void IccAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (IccAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~IccAggregateState();
    }
}

static void IccAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                          Vector &state_vector, idx_t count) {
    UnifiedVectorFormat val_data, subj_data, rater_data;
    inputs[0].ToUnifiedFormat(count, val_data);
    inputs[1].ToUnifiedFormat(count, subj_data);
    inputs[2].ToUnifiedFormat(count, rater_data);
    auto vals = UnifiedVectorFormat::GetData<double>(val_data);
    auto subjs = UnifiedVectorFormat::GetData<int64_t>(subj_data);
    auto raters = UnifiedVectorFormat::GetData<int64_t>(rater_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (IccAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = val_data.sel->get_index(i);
        auto subj_idx = subj_data.sel->get_index(i);
        auto rater_idx = rater_data.sel->get_index(i);

        if (!val_data.validity.RowIsValid(val_idx) ||
            !subj_data.validity.RowIsValid(subj_idx) ||
            !rater_data.validity.RowIsValid(rater_idx)) {
            continue;
        }

        double val = vals[val_idx];
        if (std::isnan(val)) {
            continue;
        }

        state.values.push_back(val);
        state.subject_ids.push_back(subjs[subj_idx]);
        state.rater_ids.push_back(raters[rater_idx]);
    }
}

static void IccAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (IccAggregateState **)source_data.data;
    auto targets = (IccAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.values = std::move(source.values);
            target.subject_ids = std::move(source.subject_ids);
            target.rater_ids = std::move(source.rater_ids);
            target.initialized = true;
            continue;
        }

        target.values.insert(target.values.end(), source.values.begin(), source.values.end());
        target.subject_ids.insert(target.subject_ids.end(), source.subject_ids.begin(), source.subject_ids.end());
        target.rater_ids.insert(target.rater_ids.end(), source.rater_ids.begin(), source.rater_ids.end());
    }
}

static void IccAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                            idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (IccAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<IccBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.values.size() < 4) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build data matrix from subject/rater pairs
        std::map<int64_t, size_t> subj_map, rater_map;
        for (auto s : state.subject_ids) {
            if (subj_map.find(s) == subj_map.end()) {
                subj_map[s] = subj_map.size();
            }
        }
        for (auto r : state.rater_ids) {
            if (rater_map.find(r) == rater_map.end()) {
                rater_map[r] = rater_map.size();
            }
        }

        size_t n_subjects = subj_map.size();
        size_t n_raters = rater_map.size();

        if (n_subjects < 2 || n_raters < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build the data matrix (row-major: subjects x raters)
        // Initialize with NaN to detect missing values
        vector<double> data(n_subjects * n_raters, std::numeric_limits<double>::quiet_NaN());

        for (size_t j = 0; j < state.values.size(); j++) {
            size_t s = subj_map[state.subject_ids[j]];
            size_t r = rater_map[state.rater_ids[j]];
            data[s * n_raters + r] = state.values[j];
        }

        // Check for missing values - ICC requires complete data
        bool has_missing = false;
        for (auto v : data) {
            if (std::isnan(v)) {
                has_missing = true;
                break;
            }
        }
        if (has_missing) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxIccResult icc_result;
        AnofoxError error;

        bool success = anofox_icc(data.data(), n_subjects, n_raters,
                                   bind_data.icc_type, &icc_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = icc_result.icc;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = icc_result.f_statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = icc_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = icc_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(icc_result.n_subjects);
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(icc_result.n_raters);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, icc_result.method ? icc_result.method : "ICC");

        anofox_free_icc_result(&icc_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> IccAggBind(ClientContext &context, AggregateFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetIccAggResultType();
    auto bind_data = make_uniq<IccBindData>();

    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[3]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "type") == 0) {
                        auto type_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(type_str.c_str(), "average") == 0) {
                            bind_data->icc_type = ANOFOX_ICC_AVERAGE;
                        } else {
                            bind_data->icc_type = ANOFOX_ICC_SINGLE;
                        }
                    }
                }
            }
        }
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("icc_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterIccAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_icc_agg");

    // With options: (value DOUBLE, subject_id BIGINT, rater_id BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_icc_agg", {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<IccAggregateState>, IccAggInitialize,
        IccAggUpdate, IccAggCombine, IccAggFinalize,
        nullptr, IccAggBind, IccAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (value DOUBLE, subject_id BIGINT, rater_id BIGINT)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_icc_agg", {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<IccAggregateState>, IccAggInitialize,
        IccAggUpdate, IccAggCombine, IccAggFinalize,
        nullptr, IccAggBind, IccAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("icc_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
