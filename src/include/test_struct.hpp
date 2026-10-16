#pragma once

// Unified hypothesis-test STRUCT (#159, integration-contract `test`).
//
// Every test aggregate returns the unified fields method, alternative,
// statistic, df1, df2, p_value, estimate, conf_low, conf_high, conf_level,
// effect_size, effect_name and n, always present and NULL when not applicable.
// A test keeps its own fields, names and order; the unified fields it does not
// already have are appended after them. Fields it renames (ci_lower/ci_upper ->
// conf_low/conf_high, df -> df1, ANOVA f_statistic -> statistic) stay as
// deprecated aliases for one release.
//
// Bind:     children = <the test's own fields>;
//           AppendTestResultFields(children);
// Finalize: <write the test's own fields, counting them in struct_idx>;
//           WriteTestResultFields(result, row, struct_idx, ffi_result.unified);
//
// The writer fills only the appended fields (index >= struct_idx), by name, from
// the upstream TestResult carried in AnofoxTestSummaryFFI, so the values of the
// existing fields never change.

#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#include "anofox_stats_ffi.h"

namespace duckdb {

//! The unified field names, in contract order.
extern const char *const TEST_RESULT_FIELDS[13];

//! Append the unified fields the test does not already have.
void AppendTestResultFields(child_list_t<LogicalType> &children);

//! Fill the fields from index `first_field` on (the appended unified fields)
//! from `summary`, by name. NaN, n < 0 and empty names are written as NULL.
void WriteTestResultFields(Vector &result, idx_t row, idx_t first_field, const AnofoxTestSummaryFFI &summary);

} // namespace duckdb
