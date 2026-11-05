#pragma once

// Prior (case) weights of poisson_fit_agg / tweedie_fit_agg / gamma_fit_agg
// (#305): `f(y, x, w)` and `f(y, x, w, options)`, as wls_fit_agg. The fit is
// upstream's (anofox_*_fit_weighted).

#include "duckdb.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb_compat.hpp"

namespace duckdb {

//! Whether the call passes a weights column: a 4th argument (y, x, w, options),
//! or a numeric 3rd one. The 3-argument overload's 3rd parameter is ANY because
//! it also takes the options MAP / STRUCT, so a numeric 3rd argument is the weight.
inline bool GlmWeightsArgument(const vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() >= 4) {
		return true;
	}
	return arguments.size() == 3 && ReturnTypeOf(*arguments[2]).IsNumeric();
}

//! The weights column of one Update chunk as DOUBLE (cast when the 3rd argument
//! of the ANY overload is another numeric type).
struct GlmWeightsInput {
	unique_ptr<Vector> cast;
	UnifiedVectorFormat data;
	const double *values = nullptr;

	void Load(Vector &input, idx_t count) {
		Vector *source = &input;
		if (input.GetType().id() != LogicalTypeId::DOUBLE) {
			cast = make_uniq<Vector>(LogicalType::DOUBLE, count);
			VectorOperations::DefaultCast(input, *cast, count);
			source = cast.get();
		}
		source->ToUnifiedFormat(count, data);
		values = UnifiedVectorFormat::GetData<double>(data);
	}

	//! The weight of row i, or false when it is NULL.
	bool Get(idx_t i, double &w) const {
		auto idx = data.sel->get_index(i);
		if (!data.validity.RowIsValid(idx)) {
			return false;
		}
		w = values[idx];
		return true;
	}
};

} // namespace duckdb
