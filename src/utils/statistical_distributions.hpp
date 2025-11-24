#pragma once

// Bridge to libanostat distributions - forwards DuckDB namespace to library
#include "libanostat/utils/distributions.hpp"

namespace duckdb {
namespace anofox_statistics {

// Forward all distribution functions from libanostat to DuckDB namespace
// This maintains backward compatibility while using the library implementation

using ::libanostat::utils::beta_inc_reg;
using ::libanostat::utils::ChiSquaredCDF;
using ::libanostat::utils::log_beta;
using ::libanostat::utils::log_gamma;
using ::libanostat::utils::student_t_cdf;
using ::libanostat::utils::student_t_critical;
using ::libanostat::utils::student_t_pvalue;

} // namespace anofox_statistics
} // namespace duckdb
