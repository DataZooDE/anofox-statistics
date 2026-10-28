#pragma once

#include "duckdb.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace duckdb {

//! Total order on doubles (-0.0 before +0.0, NaN sorts last, all NaNs equal), so
//! std::sort stays well-defined even if a NaN slips into the data and rows that
//! compare equal are bitwise interchangeable.
inline int CompareDoubleTotal(double a, double b) {
	if (a < b) {
		return -1;
	}
	if (a > b) {
		return 1;
	}
	if (a == b) {
		if (a == 0.0) {
			const bool a_neg = std::signbit(a);
			const bool b_neg = std::signbit(b);
			return a_neg == b_neg ? 0 : (a_neg ? -1 : 1);
		}
		return 0;
	}
	const bool a_nan = std::isnan(a);
	const bool b_nan = std::isnan(b);
	return a_nan == b_nan ? 0 : (a_nan ? 1 : -1);
}

//! Reorder the rows of parallel columns into a canonical order: lexicographic by
//! the key columns, in the order they were added. Carried columns are permuted
//! along without being part of the key.
//!
//! The aggregates buffer their rows in whatever order DuckDB's threads delivered
//! them. Floating-point sums (and seeded subsampling draws) depend on that order,
//! so the finalizers sort first: the result is then bitwise identical for the same
//! multiset of rows, whatever the thread count or row order (#214, #226).
//!
//!   CanonicalRows().Keys(state.x_columns).Key(state.y_values).Carry(state.ids).Sort();
class CanonicalRows {
public:
	template <class V>
	CanonicalRows &Key(V &column) {
		keys.push_back(column.data());
		AddColumn(column);
		return *this;
	}
	template <class COLS>
	CanonicalRows &Keys(COLS &columns) {
		for (auto &col : columns) {
			Key(col);
		}
		return *this;
	}
	template <class V>
	CanonicalRows &Carry(V &column) {
		AddColumn(column);
		return *this;
	}

	//! Sort the rows. Returns the permutation: sorted row k was input row order[k]
	//! (empty when nothing was sorted, i.e. the order is the identity). Callers
	//! whose output is per input row use it to put results back in input order.
	std::vector<idx_t> Sort() {
		if (n < 2 || keys.empty()) {
			return {};
		}
		// Sort (first key, row) pairs: the first key decides nearly every comparison
		// and sits next to the index, so the sort stays cache-friendly.
		struct Entry {
			double k0;
			idx_t row;
		};
		std::vector<Entry> entries(n);
		const double *k0 = keys[0];
		for (idx_t r = 0; r < n; r++) {
			entries[r] = {k0[r], r};
		}
		const idx_t n_keys = keys.size();
		std::sort(entries.begin(), entries.end(), [&](const Entry &a, const Entry &b) {
			int c = CompareDoubleTotal(a.k0, b.k0);
			for (idx_t k = 1; c == 0 && k < n_keys; k++) {
				c = CompareDoubleTotal(keys[k][a.row], keys[k][b.row]);
			}
			return c < 0;
		});
		std::vector<idx_t> order(n);
		bool identity = true;
		for (idx_t r = 0; r < n; r++) {
			order[r] = entries[r].row;
			identity = identity && order[r] == r;
		}
		if (identity) {
			return {};
		}
		for (auto &permute : permutes) {
			permute(order);
		}
		return order;
	}

private:
	template <class V>
	void AddColumn(V &column) {
		if (permutes.empty()) {
			n = column.size();
		} else if (column.size() != n) {
			throw InternalException("CanonicalRows: columns of different lengths (%llu vs %llu)",
			                        static_cast<unsigned long long>(column.size()), static_cast<unsigned long long>(n));
		}
		V *col = &column;
		permutes.push_back([col](const std::vector<idx_t> &order) {
			V sorted;
			sorted.reserve(order.size());
			for (auto r : order) {
				sorted.push_back(std::move((*col)[r]));
			}
			col->swap(sorted);
		});
	}

	idx_t n = 0;
	std::vector<const double *> keys;
	std::vector<std::function<void(const std::vector<idx_t> &)>> permutes;
};

//! Reorder the rows of a column-major design (y plus feature columns) into a
//! canonical order: lexicographic by (x_0, ..., x_{p-1}, y).
//!
//! RANSAC and Theil-Sen feed a seeded random subsampler, so without this the same
//! seed would draw different subsamples depending on the thread count; for every
//! other fit it fixes the floating-point summation order.
template <typename YVEC, typename XCOLS>
void SortRowsCanonically(YVEC &y, XCOLS &x_columns) {
	CanonicalRows().Keys(x_columns).Key(y).Sort();
}

} // namespace duckdb
