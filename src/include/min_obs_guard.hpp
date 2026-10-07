#pragma once

#include "duckdb.hpp"

#include <cmath>
#include <vector>

namespace duckdb {

//! Minimum number of rows a linear fit needs before it is attempted (the caller
//! requires strictly more rows than this).
//!
//! Only columns that carry information are counted: a column that is constant
//! over the finite values (e.g. all zero) is aliased by the Rust core and gets a
//! NaN coefficient, so it must not make the guard reject an otherwise
//! well-determined fit. Without an intercept a constant non-zero column plays the
//! role of the intercept (the core keeps the first one), so it counts once.
//! Detailed validation (tolerance-based constant detection, rank) stays in Rust.
inline idx_t MinObsForFit(const vector<vector<double>> &x_columns, bool fit_intercept) {
	idx_t effective = 0;
	bool constant_nonzero = false;
	for (auto &col : x_columns) {
		bool have = false;
		bool constant = true;
		double first = 0.0;
		for (double v : col) {
			if (!std::isfinite(v)) {
				continue;
			}
			if (!have) {
				first = v;
				have = true;
			} else if (v != first) {
				constant = false;
				break;
			}
		}
		if (!constant) {
			effective++;
		} else if (have && first != 0.0) {
			constant_nonzero = true;
		}
	}
	if (fit_intercept) {
		return effective + 1;
	}
	return effective + (constant_nonzero ? 1 : 0);
}

} // namespace duckdb
