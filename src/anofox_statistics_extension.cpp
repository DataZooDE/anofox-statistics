#define DUCKDB_EXTENSION_MAIN

#include "anofox_statistics_extension.hpp"
#include "functions/ols_metrics.hpp"
#include "functions/ols_fit.hpp"                              // Phase 2 - Array-based OLS
#include "functions/ridge_fit.hpp"                            // Phase 2 - Ridge regression
#include "functions/wls_fit.hpp"                              // Phase 2 - Weighted LS
#include "functions/rls_fit.hpp"                              // Phase 3 - Recursive LS
#include "functions/rolling_ols.hpp"                          // Phase 3 - Rolling window
#include "functions/expanding_ols.hpp"                        // Phase 3 - Expanding window
#include "functions/aggregates/ols_aggregate.hpp"             // Phase 4 - Aggregates
#include "functions/aggregates/wls_aggregate.hpp"             // Phase 4 - WLS Aggregate
#include "functions/aggregates/ridge_aggregate.hpp"           // Phase 4 - Ridge Aggregate
#include "functions/aggregates/rls_aggregate.hpp"             // Phase 4 - RLS Aggregate
#include "functions/inference/ols_inference.hpp"              // Phase 5 - Inference
#include "functions/inference/prediction_intervals.hpp"       // Phase 5 - Prediction
#include "functions/model_selection/information_criteria.hpp" // Phase 5 - Model selection
#include "functions/diagnostics/residual_diagnostics.hpp"     // Phase 5 - Diagnostics
#include "functions/diagnostics/vif.hpp"                      // Phase 5 - Multicollinearity
#include "functions/diagnostics/normality_test.hpp"           // Phase 5 - Normality
#include "duckdb/common/exception.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

void AnofoxStatisticsExtension::Load(ExtensionLoader &loader) {
	// Phase 1: OLS regression - metrics functions (✅ completed)
	anofox_statistics::OlsMetricsFunction::Register(loader);

	// Phase 2: Regression fit functions (✅ completed)
	anofox_statistics::OlsFitFunction::Register(loader);   // Multi-variable OLS
	anofox_statistics::RidgeFitFunction::Register(loader); // Ridge regression (L2)
	anofox_statistics::WlsFitFunction::Register(loader);   // Weighted Least Squares

	// Phase 3: Online/Sequential learning (✅ completed)
	anofox_statistics::RlsFitFunction::Register(loader);       // Recursive Least Squares
	anofox_statistics::RollingOlsFunction::Register(loader);   // Rolling window OLS
	anofox_statistics::ExpandingOlsFunction::Register(loader); // Expanding window OLS

	// Phase 4: Aggregates & Window Functions (✅ completed)
	anofox_statistics::OlsAggregateFunction::Register(loader);   // OLS with GROUP BY
	anofox_statistics::WlsAggregateFunction::Register(loader);   // WLS with GROUP BY
	anofox_statistics::RidgeAggregateFunction::Register(loader); // Ridge with GROUP BY
	anofox_statistics::RlsAggregateFunction::Register(loader);   // RLS with GROUP BY

	// Phase 5: Statistical Inference & Diagnostics (✅ completed)
	anofox_statistics::OlsInferenceFunction::Register(loader);        // Coefficient inference
	anofox_statistics::OlsPredictIntervalFunction::Register(loader);  // Prediction intervals
	anofox_statistics::InformationCriteriaFunction::Register(loader); // AIC, BIC
	anofox_statistics::ResidualDiagnosticsFunction::Register(loader); // Residual diagnostics
	anofox_statistics::VifFunction::Register(loader);                 // Multicollinearity (VIF)
	anofox_statistics::NormalityTestFunction::Register(loader);       // Normality test

	// Future additions:
	// - Heteroscedasticity tests (Breusch-Pagan, White)
	// - Robust regression (M-estimators)
	// - Bootstrap inference
	// - Cross-validation
}

std::string AnofoxStatisticsExtension::Name() {
	return "anofox-statistics";
}

std::string AnofoxStatisticsExtension::Version() const {
#ifdef EXT_VERSION_ANOFOX
	return EXT_VERSION_ANOFOX;
#else
	return "0.1.0";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_statistics, loader) {
	duckdb::AnofoxStatisticsExtension extension;
	extension.Load(loader);
}

} // extern "C"
