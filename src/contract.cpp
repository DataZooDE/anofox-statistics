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
