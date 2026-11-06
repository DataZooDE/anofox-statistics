#pragma once

// Cluster-robust (sandwich) covariance overloads of ols_fit_agg / wls_fit_agg (#309),
// implemented in src/aggregate_functions/cluster_fit_aggregate.cpp and added to the
// sibling's function set by RegisterOlsAggregateFunction / RegisterWlsAggregateFunction.

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"

namespace duckdb {

//! ols_fit_agg(y, x, cluster, options) or, `weighted`, wls_fit_agg(y, x, weight, cluster, options).
void AddClusterFitOverload(AggregateFunctionSet &set, bool weighted);

//! The FunctionDescription of that overload.
FunctionDescription ClusterFitDescription(bool weighted);

} // namespace duckdb
