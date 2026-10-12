#include "test_struct.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>

namespace duckdb {

// Nine doubles, n, then the three name buffers (no padding on 64-bit or wasm32).
static_assert(offsetof(AnofoxTestSummaryFFI, n) == 9 * sizeof(double) &&
                  offsetof(AnofoxTestSummaryFFI, method) == 10 * sizeof(double) &&
                  sizeof(AnofoxTestSummaryFFI) ==
                      10 * sizeof(double) + ANOFOX_TEST_METHOD_LEN + 2 * ANOFOX_TEST_NAME_LEN,
              "AnofoxTestSummaryFFI must match TestSummaryFFI in crates/anofox-stats-ffi/src/types.rs");

// The unified fields are appended last to these result structs (after padding).
#define ANOFOX_ASSERT_UNIFIED_LAST(T)                                                                                  \
	static_assert(sizeof(T) == offsetof(T, unified) + sizeof(AnofoxTestSummaryFFI),                                    \
	              #T " must end with its unified test fields, as its Rust counterpart does")
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxTestResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxAnovaResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxCorrelationResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxChiSquareResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxTostResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxIccResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxPropTestResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxKappaResult);
ANOFOX_ASSERT_UNIFIED_LAST(AnofoxJarqueBeraResult);
#undef ANOFOX_ASSERT_UNIFIED_LAST

// Seven doubles, then the two group indices.
static_assert(offsetof(AnofoxPairwiseRow, group1) == 7 * sizeof(double) &&
                  sizeof(AnofoxPairwiseRow) == 7 * sizeof(double) + 2 * sizeof(uint64_t),
              "AnofoxPairwiseRow must match PairwiseRowFFI in crates/anofox-stats-ffi/src/types.rs");
// rows + len, conf_level, then the three name buffers (no padding on 64-bit or wasm32).
static_assert(offsetof(AnofoxPairwiseResult, conf_level) == sizeof(AnofoxPairwiseRow *) + sizeof(size_t) &&
                  sizeof(AnofoxPairwiseResult) == sizeof(AnofoxPairwiseRow *) + sizeof(size_t) + sizeof(double) +
                                                      ANOFOX_TEST_METHOD_LEN + 2 * ANOFOX_TEST_NAME_LEN,
              "AnofoxPairwiseResult must match PairwiseResultFFI in crates/anofox-stats-ffi/src/types.rs");

const char *const TEST_RESULT_FIELDS[13] = {"method",   "alternative", "statistic",  "df1",
                                            "df2",      "p_value",     "estimate",   "conf_low",
                                            "conf_high", "conf_level", "effect_size", "effect_name",
                                            "n"};

static LogicalType TestFieldType(const string &name) {
	if (name == "method" || name == "alternative" || name == "effect_name") {
		return LogicalType::VARCHAR;
	}
	if (name == "n") {
		return LogicalType::BIGINT;
	}
	return LogicalType::DOUBLE;
}

void AppendTestResultFields(child_list_t<LogicalType> &children) {
	for (auto name : TEST_RESULT_FIELDS) {
		bool present = false;
		for (auto &child : children) {
			if (child.first == name) {
				present = true;
				break;
			}
		}
		if (!present) {
			children.push_back(make_pair(string(name), TestFieldType(name)));
		}
	}
}

static void WriteDouble(Vector &vec, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatVector::GetData<double>(vec)[row] = value;
	}
}

static void WriteName(Vector &vec, idx_t row, const char *value, size_t capacity) {
	// The buffers are NUL-terminated by the Rust side; bound the read anyway.
	auto len = strnlen(value, capacity);
	if (len == 0) {
		FlatVector::SetNull(vec, row, true);
		return;
	}
	FlatVector::GetData<string_t>(vec)[row] = StringVector::AddString(vec, value, len);
}

void WriteTestResultFields(Vector &result, idx_t row, idx_t first_field, const AnofoxTestSummaryFFI &s) {
	auto &types = StructType::GetChildTypes(result.GetType());
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = first_field; i < types.size(); i++) {
		auto &name = types[i].first;
		auto &vec = *entries[i];
		if (name == "method") {
			WriteName(vec, row, s.method, ANOFOX_TEST_METHOD_LEN);
		} else if (name == "alternative") {
			WriteName(vec, row, s.alternative, ANOFOX_TEST_NAME_LEN);
		} else if (name == "effect_name") {
			WriteName(vec, row, s.effect_name, ANOFOX_TEST_NAME_LEN);
		} else if (name == "n") {
			if (s.n < 0) {
				FlatVector::SetNull(vec, row, true);
			} else {
				FlatVector::GetData<int64_t>(vec)[row] = s.n;
			}
		} else if (name == "statistic") {
			WriteDouble(vec, row, s.statistic);
		} else if (name == "df1") {
			WriteDouble(vec, row, s.df1);
		} else if (name == "df2") {
			WriteDouble(vec, row, s.df2);
		} else if (name == "p_value") {
			WriteDouble(vec, row, s.p_value);
		} else if (name == "estimate") {
			WriteDouble(vec, row, s.estimate);
		} else if (name == "conf_low") {
			WriteDouble(vec, row, s.conf_low);
		} else if (name == "conf_high") {
			WriteDouble(vec, row, s.conf_high);
		} else if (name == "conf_level") {
			WriteDouble(vec, row, s.conf_level);
		} else if (name == "effect_size") {
			WriteDouble(vec, row, s.effect_size);
		}
	}
}

} // namespace duckdb
