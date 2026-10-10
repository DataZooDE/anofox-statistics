#pragma once

// Stable model STRUCT (#152, integration-contract `terms`/`summary`).
//
// Every fit returns the same fields whatever its options: inference fields are
// always present and NULL when not computed, and every model ends with a shared
// tail that describes it (model_type, family, link), carries its likelihood-based
// fit statistics and the intercept's inference, and names its confidence
// intervals conf_low / conf_high. ci_lower / ci_upper stay as deprecated aliases
// of conf_low / conf_high for one release.
//
// Bind:     AppendCoefficientInferenceFields(children);  // where the model had them
//           AppendModelSummaryFields(children);
// Finalize: ModelStructWriter writer(result);
//           writer.WriteInference(row, compute_inference ? &inference : nullptr);
//           writer.WriteSummary(row, core.summary);
//
// The writer finds fields by name, so a model keeps its own field order and
// fields it already had (GLMs' family, link, aic) are filled in place.

#include "duckdb.hpp"

#include "anofox_stats_ffi.h"

#include <unordered_map>

namespace duckdb {

//! Coefficient inference lists, the deprecated CI aliases and the overall F test:
//! std_errors, <statistic_name>, p_values[, ci_lower, ci_upper][, f_statistic, f_pvalue].
//! Models that never had ci_lower / ci_upper pass ci_aliases = false.
void AppendCoefficientInferenceFields(child_list_t<LogicalType> &children, const char *statistic_name = "t_values",
                                      bool f_test = true, bool ci_aliases = true);

//! The shared tail, skipping fields the model already has: conf_low, conf_high,
//! conf_level, intercept_std_error, intercept_statistic, intercept_p_value,
//! intercept_conf_low, intercept_conf_high, log_likelihood, aic, bic, model_type,
//! family, link.
void AppendModelSummaryFields(child_list_t<LogicalType> &children);

//! Fills the stable fields of a model STRUCT result vector by name.
class ModelStructWriter {
public:
	explicit ModelStructWriter(Vector &result);

	//! Coefficient inference, conf_low/conf_high (and the ci_ aliases), conf_level
	//! and the F test. A null `inference` (not requested) writes NULL to all of them.
	void WriteInference(idx_t row, const AnofoxFitResultInference *inference);

	//! model_type, family, link, log_likelihood, aic, bic and the intercept inference.
	//! NaN and an empty family are written as NULL.
	void WriteSummary(idx_t row, const AnofoxModelSummaryFFI &summary);

private:
	Vector *Field(const char *name) const;

	std::unordered_map<string, Vector *> fields;
};

} // namespace duckdb
