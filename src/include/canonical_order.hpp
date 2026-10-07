#pragma once

#include "duckdb.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace duckdb {

//! Total order on doubles (NaN sorts last, all NaNs equal), so std::sort stays
//! well-defined even if a NaN slips into the data.
inline int CompareDoubleTotal(double a, double b) {
	const bool a_nan = std::isnan(a);
	const bool b_nan = std::isnan(b);
	if (a_nan || b_nan) {
		return a_nan == b_nan ? 0 : (a_nan ? 1 : -1);
	}
	return a < b ? -1 : (a > b ? 1 : 0);
}

//! Reorder the rows of a column-major design (y plus feature columns) into a
//! canonical order: lexicographic by (x_0, ..., x_{p-1}, y).
//!
//! Aggregates that feed a seeded random subsampler (RANSAC, Theil-Sen) call this
//! before fitting. The rows arrive in whatever order DuckDB's threads delivered
//! them, so without it the same seed would draw different subsamples depending
//! on the thread count.
template <typename YVEC, typename XCOLS>
void SortRowsCanonically(YVEC &y, XCOLS &x_columns) {
	const idx_t n = y.size();
	std::vector<idx_t> order(n);
	std::iota(order.begin(), order.end(), idx_t(0));
	std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
		for (auto &col : x_columns) {
			int c = CompareDoubleTotal(col[a], col[b]);
			if (c != 0) {
				return c < 0;
			}
		}
		return CompareDoubleTotal(y[a], y[b]) < 0;
	});
	YVEC buffer(n);
	auto apply = [&](YVEC &v) {
		for (idx_t k = 0; k < n; k++) {
			buffer[k] = v[order[k]];
		}
		v.swap(buffer);
	};
	apply(y);
	for (auto &col : x_columns) {
		apply(col);
	}
}

} // namespace duckdb
