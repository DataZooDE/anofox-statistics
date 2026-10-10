#include "model_struct.hpp"

#include <cmath>
#include <cstddef>

namespace duckdb {

static_assert(sizeof(AnofoxModelSummaryFFI) == 3 * ANOFOX_MODEL_NAME_LEN + 8 * sizeof(double),
              "AnofoxModelSummaryFFI must match ModelSummaryFFI in crates/anofox-stats-ffi/src/types.rs");
// coefficients + coefficients_len, 4 doubles, n_observations + n_features, summary
// (no padding on 64-bit or wasm32).
static_assert(sizeof(AnofoxFitResultCore) ==
                  sizeof(double *) + 3 * sizeof(size_t) + 4 * sizeof(double) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxFitResultCore must match FitResultCore in crates/anofox-stats-ffi/src/types.rs");

// The summary is appended last to these result structs (after padding).
static_assert(sizeof(AnofoxGlmFitResultCore) ==
                  offsetof(AnofoxGlmFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxGlmFitResultCore must end with its summary, as GlmFitResultCore does");
static_assert(sizeof(AnofoxAftFitResultCore) ==
                  offsetof(AnofoxAftFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxAftFitResultCore must end with its summary, as AftFitResultCore does");
static_assert(sizeof(AnofoxGlmmResult) == offsetof(AnofoxGlmmResult, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxGlmmResult must end with its summary, as GlmmResultFFI does");

static_assert(sizeof(AnofoxAlmFitResultCore) ==
                  offsetof(AnofoxAlmFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxAlmFitResultCore must end with its summary, as AlmFitResultCore does");

static_assert(sizeof(AnofoxBlsFitResultCore) ==
                  offsetof(AnofoxBlsFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxBlsFitResultCore must end with its summary, as BlsFitResultCore does");
static_assert(sizeof(AnofoxPlsFitResultCore) ==
                  offsetof(AnofoxPlsFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxPlsFitResultCore must end with its summary, as PlsFitResultCore does");
static_assert(sizeof(AnofoxIsotonicFitResultCore) ==
                  offsetof(AnofoxIsotonicFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxIsotonicFitResultCore must end with its summary, as IsotonicFitResultCore does");
static_assert(sizeof(AnofoxQuantileFitResultCore) ==
                  offsetof(AnofoxQuantileFitResultCore, summary) + sizeof(AnofoxModelSummaryFFI),
              "AnofoxQuantileFitResultCore must end with its summary, as QuantileFitResultCore does");

static bool HasField(const child_list_t<LogicalType> &children, const string &name) {
	for (auto &child : children) {
		if (child.first == name) {
			return true;
		}
	}
	return false;
}

static void AppendIfMissing(child_list_t<LogicalType> &children, const string &name, const LogicalType &type) {
	if (!HasField(children, name)) {
		children.push_back(make_pair(name, type));
	}
}

void AppendCoefficientInferenceFields(child_list_t<LogicalType> &children, const char *statistic_name, bool f_test,
                                      bool ci_aliases) {
	auto list = LogicalType::LIST(LogicalType::DOUBLE);
	AppendIfMissing(children, "std_errors", list);
	AppendIfMissing(children, statistic_name, list);
	AppendIfMissing(children, "p_values", list);
	if (ci_aliases) {
		AppendIfMissing(children, "ci_lower", list);
		AppendIfMissing(children, "ci_upper", list);
	}
	if (f_test) {
		AppendIfMissing(children, "f_statistic", LogicalType::DOUBLE);
		AppendIfMissing(children, "f_pvalue", LogicalType::DOUBLE);
	}
}

void AppendModelSummaryFields(child_list_t<LogicalType> &children) {
	auto list = LogicalType::LIST(LogicalType::DOUBLE);
	AppendIfMissing(children, "conf_low", list);
	AppendIfMissing(children, "conf_high", list);
	AppendIfMissing(children, "conf_level", LogicalType::DOUBLE);
	AppendIfMissing(children, "intercept_std_error", LogicalType::DOUBLE);
	AppendIfMissing(children, "intercept_statistic", LogicalType::DOUBLE);
	AppendIfMissing(children, "intercept_p_value", LogicalType::DOUBLE);
	AppendIfMissing(children, "intercept_conf_low", LogicalType::DOUBLE);
	AppendIfMissing(children, "intercept_conf_high", LogicalType::DOUBLE);
	AppendIfMissing(children, "log_likelihood", LogicalType::DOUBLE);
	AppendIfMissing(children, "aic", LogicalType::DOUBLE);
	AppendIfMissing(children, "bic", LogicalType::DOUBLE);
	AppendIfMissing(children, "model_type", LogicalType::VARCHAR);
	AppendIfMissing(children, "family", LogicalType::VARCHAR);
	AppendIfMissing(children, "link", LogicalType::VARCHAR);
}

ModelStructWriter::ModelStructWriter(Vector &result) {
	auto &types = StructType::GetChildTypes(result.GetType());
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < types.size(); i++) {
		fields[types[i].first] = entries[i].get();
	}
}

Vector *ModelStructWriter::Field(const char *name) const {
	auto it = fields.find(name);
	return it == fields.end() ? nullptr : it->second;
}

static void WriteDouble(Vector *vec, idx_t row, double value) {
	if (!vec) {
		return;
	}
	if (std::isnan(value)) {
		FlatVector::SetNull(*vec, row, true);
	} else {
		FlatVector::GetData<double>(*vec)[row] = value;
	}
}

static void WriteList(Vector *vec, idx_t row, const double *data, size_t len) {
	if (!vec) {
		return;
	}
	if (!data && len > 0) {
		FlatVector::SetNull(*vec, row, true);
		return;
	}
	auto offset = ListVector::GetListSize(*vec);
	ListVector::Reserve(*vec, offset + len);
	auto &child = ListVector::GetEntry(*vec);
	auto child_data = FlatVector::GetData<double>(child);
	for (size_t i = 0; i < len; i++) {
		child_data[offset + i] = data[i];
	}
	ListVector::SetListSize(*vec, offset + len);
	ListVector::GetData(*vec)[row] = {offset, (idx_t)len};
}

static void WriteString(Vector *vec, idx_t row, const char *value) {
	if (!vec) {
		return;
	}
	if (!value || !value[0]) {
		FlatVector::SetNull(*vec, row, true);
		return;
	}
	FlatVector::GetData<string_t>(*vec)[row] = StringVector::AddString(*vec, value);
}

static void WriteNull(Vector *vec, idx_t row) {
	if (vec) {
		FlatVector::SetNull(*vec, row, true);
	}
}

void ModelStructWriter::WriteInference(idx_t row, const AnofoxFitResultInference *inference) {
	static const char *const LIST_FIELDS[] = {"std_errors", "t_values", "z_values", "p_values",
	                                          "ci_lower",   "ci_upper", "conf_low", "conf_high"};
	if (!inference) {
		for (auto name : LIST_FIELDS) {
			WriteNull(Field(name), row);
		}
		WriteNull(Field("conf_level"), row);
		WriteNull(Field("f_statistic"), row);
		WriteNull(Field("f_pvalue"), row);
		return;
	}
	auto len = inference->len;
	WriteList(Field("std_errors"), row, inference->std_errors, len);
	WriteList(Field("t_values"), row, inference->t_values, len);
	WriteList(Field("z_values"), row, inference->t_values, len);
	WriteList(Field("p_values"), row, inference->p_values, len);
	WriteList(Field("ci_lower"), row, inference->ci_lower, len);
	WriteList(Field("ci_upper"), row, inference->ci_upper, len);
	WriteList(Field("conf_low"), row, inference->ci_lower, len);
	WriteList(Field("conf_high"), row, inference->ci_upper, len);
	WriteDouble(Field("conf_level"), row, inference->confidence_level);
	// The F test keeps its NaN (no F test for this fit) as NaN, as before.
	if (auto f = Field("f_statistic")) {
		FlatVector::GetData<double>(*f)[row] = inference->f_statistic;
	}
	if (auto f = Field("f_pvalue")) {
		FlatVector::GetData<double>(*f)[row] = inference->f_pvalue;
	}
}

void ModelStructWriter::WriteSummary(idx_t row, const AnofoxModelSummaryFFI &summary) {
	WriteString(Field("model_type"), row, summary.model_type);
	WriteString(Field("family"), row, summary.family);
	WriteString(Field("link"), row, summary.link);
	WriteDouble(Field("log_likelihood"), row, summary.log_likelihood);
	WriteDouble(Field("aic"), row, summary.aic);
	WriteDouble(Field("bic"), row, summary.bic);
	WriteDouble(Field("intercept_std_error"), row, summary.intercept_std_error);
	WriteDouble(Field("intercept_statistic"), row, summary.intercept_statistic);
	WriteDouble(Field("intercept_p_value"), row, summary.intercept_p_value);
	WriteDouble(Field("intercept_conf_low"), row, summary.intercept_conf_low);
	WriteDouble(Field("intercept_conf_high"), row, summary.intercept_conf_high);
}

} // namespace duckdb
