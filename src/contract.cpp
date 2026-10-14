#include "contract.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/transaction/meta_transaction.hpp"

namespace duckdb {

static constexpr CatalogType FUNCTION_CATALOG_TYPES[] = {
    CatalogType::SCALAR_FUNCTION_ENTRY, CatalogType::AGGREGATE_FUNCTION_ENTRY, CatalogType::TABLE_FUNCTION_ENTRY,
    CatalogType::MACRO_ENTRY, CatalogType::TABLE_MACRO_ENTRY};

// Contract producers: function name -> output schema (terms, obs, prediction,
// curve, summary, test). A function is listed only once its output carries every
// required column of its schema; test/sql/contract/ checks each one.
// clang-format off
static const struct {
	const char *name;
	const char *output;
} CONTRACT_PRODUCERS[] = {
    {"ols_tidy_by", "terms"}, {"ols_glance_by", "summary"},
    {"wls_tidy_by", "terms"}, {"wls_glance_by", "summary"},
    {"ridge_tidy_by", "terms"}, {"ridge_glance_by", "summary"},
    {"elasticnet_tidy_by", "terms"}, {"elasticnet_glance_by", "summary"},
    {"huber_tidy_by", "terms"}, {"huber_glance_by", "summary"},
    {"ransac_tidy_by", "terms"}, {"ransac_glance_by", "summary"},
    {"theil_sen_tidy_by", "terms"}, {"theil_sen_glance_by", "summary"},
    {"rls_tidy_by", "terms"}, {"rls_glance_by", "summary"},
    {"lars_tidy_by", "terms"}, {"lars_glance_by", "summary"},
    {"bls_tidy_by", "terms"}, {"bls_glance_by", "summary"},
    {"nnls_tidy_by", "terms"}, {"nnls_glance_by", "summary"},
    {"pls_tidy_by", "terms"}, {"pls_glance_by", "summary"},
    {"quantile_tidy_by", "terms"}, {"quantile_glance_by", "summary"},
    {"poisson_tidy_by", "terms"}, {"poisson_glance_by", "summary"},
    {"binomial_tidy_by", "terms"}, {"binomial_glance_by", "summary"},
    {"logistic_tidy_by", "terms"}, {"logistic_glance_by", "summary"},
    {"negbinom_tidy_by", "terms"}, {"negbinom_glance_by", "summary"},
    {"gamma_tidy_by", "terms"}, {"gamma_glance_by", "summary"},
    {"tweedie_tidy_by", "terms"}, {"tweedie_glance_by", "summary"},
    {"alm_tidy_by", "terms"}, {"alm_glance_by", "summary"},
    {"aft_tidy_by", "terms"}, {"aft_glance_by", "summary"},
    {"ols_augment_by", "obs"}, {"wls_augment_by", "obs"},
    {"poisson_augment_by", "obs"}, {"binomial_augment_by", "obs"},
    {"logistic_augment_by", "obs"}, {"negbinom_augment_by", "obs"},
    {"gamma_augment_by", "obs"}, {"tweedie_augment_by", "obs"},
    {"one_way_anova_agg", "test"}, {"binom_test_agg", "test"}, {"brown_forsythe_agg", "test"},
    {"brunner_munzel_agg", "test"}, {"chisq_test_agg", "test"}, {"chisq_gof_agg", "test"},
    {"clark_west_agg", "test"}, {"cohen_kappa_agg", "test"}, {"dagostino_k2_agg", "test"},
    {"diebold_mariano_agg", "test"}, {"distance_cor_agg", "test"}, {"energy_distance_agg", "test"},
    {"fisher_exact_agg", "test"}, {"g_test_agg", "test"}, {"icc_agg", "test"},
    {"jarque_bera_agg", "test"}, {"kendall_agg", "test"}, {"kruskal_wallis_agg", "test"},
    {"mann_whitney_u_agg", "test"}, {"mcnemar_agg", "test"}, {"mmd_agg", "test"},
    {"pearson_agg", "test"}, {"permutation_t_test_agg", "test"}, {"prop_test_one_agg", "test"},
    {"prop_test_two_agg", "test"}, {"shapiro_wilk_agg", "test"}, {"spearman_agg", "test"},
    {"tost_correlation_agg", "test"}, {"tost_paired_agg", "test"}, {"tost_t_test_agg", "test"},
    {"t_test_agg", "test"}, {"wilcoxon_signed_rank_agg", "test"}, {"yuen_agg", "test"},
    {"pairwise_t_test_agg", "test"}, {"tukey_hsd_agg", "test"}, {"dunn_test_agg", "test"},
    {"pairwise_t_test_by", "test"}, {"tukey_hsd_by", "test"}, {"dunn_test_by", "test"},
    {"ridge_path_by", "terms"}, {"elasticnet_path_by", "terms"}, {"lars_path_by", "terms"},
    {"quantile_process_by", "terms"}, {"ridge_cv_by", "curve"}, {"elasticnet_cv_by", "curve"},
    {"ljung_box_agg", "test"}, {"box_pierce_agg", "test"}, {"durbin_watson_agg", "test"},
    {"breusch_pagan_agg", "test"}, {"white_test_agg", "test"}, {"log_rank_test_agg", "test"},
    {"acf_by", "curve"}, {"pacf_by", "curve"}, {"kaplan_meier_by", "curve"},
    {"roc_curve_by", "curve"}, {"pr_curve_by", "curve"}, {"calibration_curve_by", "curve"},
    {"ols_prediction_by", "prediction"}, {"wls_prediction_by", "prediction"}, {"ridge_prediction_by", "prediction"},
    {"elasticnet_prediction_by", "prediction"}, {"huber_prediction_by", "prediction"}, {"ransac_prediction_by", "prediction"},
    {"theil_sen_prediction_by", "prediction"}, {"rls_prediction_by", "prediction"}, {"bls_prediction_by", "prediction"},
    {"pls_prediction_by", "prediction"}, {"quantile_prediction_by", "prediction"}, {"alm_prediction_by", "prediction"},
    {"poisson_prediction_by", "prediction"}, {"binomial_prediction_by", "prediction"}, {"logistic_prediction_by", "prediction"},
    {"negbinom_prediction_by", "prediction"}, {"gamma_prediction_by", "prediction"}, {"tweedie_prediction_by", "prediction"}, {"isotonic_prediction_by", "prediction"},
    {"ols_effects_by", "prediction"}, {"wls_effects_by", "prediction"}, {"ridge_effects_by", "prediction"},
    {"poisson_effects_by", "prediction"}, {"binomial_effects_by", "prediction"}, {"logistic_effects_by", "prediction"},
    {"negbinom_effects_by", "prediction"}, {"gamma_effects_by", "prediction"}, {"tweedie_effects_by", "prediction"},
    {"ols_rolling_tidy_by", "terms"}, {"wls_rolling_tidy_by", "terms"}, {"ridge_rolling_tidy_by", "terms"},
    {"elasticnet_rolling_tidy_by", "terms"}, {"huber_rolling_tidy_by", "terms"}, {"poisson_rolling_tidy_by", "terms"},
    {nullptr, nullptr}
};
// clang-format on

static SchemaCatalogEntry &SystemMainSchema(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	return Catalog::GetSystemCatalog(db).GetSchema(CatalogTransaction::GetSystemTransaction(db), DEFAULT_SCHEMA);
}

case_insensitive_set_t SnapshotSystemFunctionNames(ExtensionLoader &loader) {
	case_insensitive_set_t names;
	auto &schema = SystemMainSchema(loader);
	for (auto type : FUNCTION_CATALOG_TYPES) {
		schema.Scan(type, [&](CatalogEntry &entry) { names.insert(entry.name); });
	}
	return names;
}

static void ContractVersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, ANOFOX_CONTRACT_VERSION);
}

void RegisterContractVersionFunction(ExtensionLoader &loader) {
	if (loader.TryGetFunction("anofox_contract_version")) {
		return;
	}
	ScalarFunction func("anofox_contract_version", {}, LogicalType::VARCHAR, ContractVersionFunction);
	CreateScalarFunctionInfo info(func);
	FunctionDescription desc;
	desc.description = "Returns the version of the anofox integration contract (output schemas and "
	                   "duckdb_functions() tags) this extension implements.";
	desc.examples = {"anofox_contract_version()"};
	desc.categories = {"metadata"};
	info.descriptions.push_back(std::move(desc));
	loader.RegisterFunction(std::move(info));
}

void ApplyContractTags(ExtensionLoader &loader, const case_insensitive_set_t &preexisting) {
	case_insensitive_map_t<string> outputs;
	for (idx_t i = 0; CONTRACT_PRODUCERS[i].name != nullptr; i++) {
		outputs[CONTRACT_PRODUCERS[i].name] = CONTRACT_PRODUCERS[i].output;
	}
	auto &schema = SystemMainSchema(loader);
	for (auto type : FUNCTION_CATALOG_TYPES) {
		schema.Scan(type, [&](CatalogEntry &entry) {
			// anofox_contract_version() is shared by the whole anofox family;
			// whichever extension loads first registers it, untagged.
			if (preexisting.find(entry.name) != preexisting.end() || entry.name == "anofox_contract_version") {
				return;
			}
			entry.tags["anofox.family"] = "statistics";
			auto output = outputs.find(entry.name);
			if (output != outputs.end()) {
				entry.tags["anofox.contract"] = ANOFOX_CONTRACT_VERSION;
				entry.tags["anofox.output"] = output->second;
			}
		});
	}
}

} // namespace duckdb
