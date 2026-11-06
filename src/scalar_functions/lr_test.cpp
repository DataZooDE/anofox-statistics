// lr_test(model_full, model_reduced) / lr_test(model): likelihood-ratio test of two
// nested fitted model STRUCTs, or of one fit against its null model (#310).
//
//   SELECT lr_test(poisson_fit_agg(y, [x1, x2]), poisson_fit_agg(y, [x1])).*;
//
// The STRUCTs' layouts are inspected at bind time; each row hands the likelihood
// fields (log_likelihood, deviance, dispersion, null deviance / log-likelihood,
// family, model_type, n_observations and the number of estimated coefficients) to
// anofox-regression's inference::lr_test / model_lr_test. Models of different
// kinds, families or row counts, or a "full" model with no more parameters than the
// reduced one, are input errors. A NULL model gives NULL.

#include <cmath>
#include <cstring>
#include <string>

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb_compat.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/result_fields.hpp"
#include "../include/test_struct.hpp"
#include "anofox_statistics_banner.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

constexpr idx_t kNone = DConstants::INVALID_INDEX;
constexpr const char *kName = "lr_test";

//! Where a model STRUCT keeps its likelihood fields.
struct ModelFields {
	idx_t model_type = kNone, family = kNone, log_likelihood = kNone, null_log_likelihood = kNone,
	      deviance = kNone, null_deviance = kNone, dispersion = kNone, n_observations = kNone,
	      coefficients = kNone, intercept = kNone;

	bool operator==(const ModelFields &o) const {
		return model_type == o.model_type && family == o.family && log_likelihood == o.log_likelihood &&
		       null_log_likelihood == o.null_log_likelihood && deviance == o.deviance &&
		       null_deviance == o.null_deviance && dispersion == o.dispersion &&
		       n_observations == o.n_observations && coefficients == o.coefficients && intercept == o.intercept;
	}
};

bool IsNumeric(const LogicalType &t) {
	switch (t.id()) {
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::UINTEGER:
		return true;
	default:
		return false;
	}
}

ModelFields FindFields(const unique_ptr<Expression> &arg, const char *what) {
	auto &type = ReturnTypeOf(*arg);
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: %s must be the STRUCT returned by a *_fit_agg or *_fit function, got %s",
		                            kName, what, type.ToString());
	}
	ModelFields f;
	auto &children = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = children[i].first;
		auto &t = children[i].second;
		if (t.id() == LogicalTypeId::VARCHAR) {
			if (name == "model_type") {
				f.model_type = i;
			} else if (name == "family") {
				f.family = i;
			}
		} else if (t.id() == LogicalTypeId::LIST) {
			if (name == "coefficients" && ListType::GetChildType(t).id() == LogicalTypeId::DOUBLE) {
				f.coefficients = i;
			}
		} else if (IsNumeric(t)) {
			if (name == "log_likelihood") {
				f.log_likelihood = i;
			} else if (name == "null_log_likelihood") {
				f.null_log_likelihood = i;
			} else if (name == "deviance") {
				f.deviance = i;
			} else if (name == "null_deviance") {
				f.null_deviance = i;
			} else if (name == "dispersion") {
				f.dispersion = i;
			} else if (name == "n_observations") {
				f.n_observations = i;
			} else if (name == "intercept") {
				f.intercept = i;
			}
		}
	}
	if (f.coefficients == kNone) {
		throw InvalidInputException("%s: %s has no 'coefficients' field; got %s", kName, what, type.ToString());
	}
	if (f.log_likelihood == kNone && f.deviance == kNone) {
		throw InvalidInputException("%s: %s has neither a 'log_likelihood' nor a 'deviance' field; got %s", kName,
		                            what, type.ToString());
	}
	return f;
}

struct LrTestBindData : public FunctionData {
	ModelFields full;
	ModelFields reduced;
	bool has_reduced = false;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LrTestBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<LrTestBindData>();
		return full == o.full && reduced == o.reduced && has_reduced == o.has_reduced;
	}
};

LogicalType LrTestResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("method", LogicalType::VARCHAR));
	c.push_back(make_pair("statistic", LogicalType::DOUBLE));
	c.push_back(make_pair("df1", LogicalType::DOUBLE));
	c.push_back(make_pair("p_value", LogicalType::DOUBLE));
	c.push_back(make_pair("dispersion", LogicalType::DOUBLE));
	c.push_back(make_pair("n_parameters_full", LogicalType::BIGINT));
	c.push_back(make_pair("n_parameters_reduced", LogicalType::BIGINT));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

unique_ptr<FunctionData> LrTestBind(ClientContext &, ScalarBindFunction &, vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<LrTestBindData>();
	const bool two = arguments.size() >= 2;
	data->full = FindFields(arguments[0], two ? "model_full" : "model");
	if (two) {
		data->reduced = FindFields(arguments[1], "model_reduced");
		data->has_reduced = true;
	}
	PostHogTelemetry::Instance().RecordFunctionCall(kName);
	return std::move(data);
}

double DoubleAt(const StructEntries &entries, idx_t field, idx_t row) {
	if (field == kNone) {
		return std::nan("");
	}
	auto v = StructEntry(entries[field]).GetValue(row);
	if (v.IsNull()) {
		return std::nan("");
	}
	return v.GetValue<double>();
}

void CopyName(char (&dst)[ANOFOX_MODEL_NAME_LEN], const StructEntries &entries, idx_t field, idx_t row) {
	std::memset(dst, 0, ANOFOX_MODEL_NAME_LEN);
	if (field == kNone) {
		return;
	}
	auto v = StructEntry(entries[field]).GetValue(row);
	if (v.IsNull()) {
		return;
	}
	auto s = v.ToString();
	std::memcpy(dst, s.data(), MinValue<idx_t>(s.size(), ANOFOX_MODEL_NAME_LEN - 1));
}

//! The likelihood summary of the model at `row`; the parameters are the
//! coefficients that were estimated (finite) plus a fitted intercept.
AnofoxModelLikelihood ReadModel(Vector &model, const ModelFields &f, idx_t row) {
	auto &entries = StructVector::GetEntries(model);
	AnofoxModelLikelihood m {};
	CopyName(m.model_type, entries, f.model_type, row);
	CopyName(m.family, entries, f.family, row);
	m.log_likelihood = DoubleAt(entries, f.log_likelihood, row);
	m.null_log_likelihood = DoubleAt(entries, f.null_log_likelihood, row);
	m.deviance = DoubleAt(entries, f.deviance, row);
	m.null_deviance = DoubleAt(entries, f.null_deviance, row);
	m.dispersion = DoubleAt(entries, f.dispersion, row);
	double n = DoubleAt(entries, f.n_observations, row);
	m.n_observations = std::isfinite(n) ? static_cast<int64_t>(n) : -1;
	uint64_t k = std::isfinite(DoubleAt(entries, f.intercept, row)) ? 1 : 0;
	auto coef = StructEntry(entries[f.coefficients]).GetValue(row);
	if (!coef.IsNull()) {
		for (auto &c : ListValue::GetChildren(coef)) {
			if (!c.IsNull() && std::isfinite(c.GetValue<double>())) {
				k++;
			}
		}
	}
	m.n_parameters = k;
	return m;
}

void SetDouble(Vector &vec, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(vec)[row] = value;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

static void LrTestFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
void LrTestFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	LrTestFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void LrTestFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<LrTestBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &entries = StructVector::GetEntries(result);

	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(args.data[0], row) || (bind.has_reduced && FlatVector::IsNull(args.data[1], row))) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto full = ReadModel(args.data[0], bind.full, row);
		AnofoxModelLikelihood reduced {};
		if (bind.has_reduced) {
			reduced = ReadModel(args.data[1], bind.reduced, row);
		}
		AnofoxChiSqTestResult res {};
		AnofoxError error;
		if (!anofox_lr_test(full, bind.has_reduced ? &reduced : nullptr, &res, &error)) {
			ThrowUnlessDegenerate(kName, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		SetResultString(StructEntry(entries[k++]), row, "likelihood ratio");
		SetDouble(StructEntry(entries[k++]), row, res.statistic);
		SetDouble(StructEntry(entries[k++]), row, res.df);
		SetDouble(StructEntry(entries[k++]), row, res.p_value);
		SetDouble(StructEntry(entries[k++]), row, res.dispersion);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(full.n_parameters);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] =
		    static_cast<int64_t>(full.n_parameters) - static_cast<int64_t>(res.df);
		auto &n_vec = StructEntry(entries[k++]);
		if (res.n_observations >= 0) {
			FlatDataMut<int64_t>(n_vec)[row] = res.n_observations;
		} else {
			FlatVector::SetNull(n_vec, row, true);
		}
		WriteTestResultFields(result, row, k, res.unified);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

} // namespace

void RegisterLrTestFunction(ExtensionLoader &loader) {
	ScalarFunctionSet set(kName);
	ScalarFunction two({LogicalType::ANY, LogicalType::ANY}, LrTestResultType(),
	                   DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, LrTestFunction), ANOFOX_SCALAR_BIND(LrTestBind));
	ScalarFunction one({LogicalType::ANY}, LrTestResultType(), DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, LrTestFunction),
	                   ANOFOX_SCALAR_BIND(LrTestBind));
	for (auto *f : {&two, &one}) {
		MarkFallible(*f);
		SetNullHandlingOf(*f, FunctionNullHandling::SPECIAL_HANDLING);
		set.AddFunction(*f);
	}
	CreateScalarFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription d2;
	d2.description =
	    "Likelihood-ratio test of two nested fitted models (R's anova(reduced, full, test = 'LRT') / "
	    "lmtest::lrtest): the chi-square statistic, df1 = difference in estimated parameters, p_value. GLMs with an "
	    "estimated dispersion (Gamma, Tweedie) divide the deviance difference by the full model's dispersion. Both "
	    "models must be the same model type and family fitted on the same rows; model_full must have more "
	    "parameters.";
	d2.examples = {"lr_test(poisson_fit_agg(y, [x1, x2]), poisson_fit_agg(y, [x1]))"};
	d2.categories = {"regression", "hypothesis-test"};
	d2.parameter_names = {"model_full", "model_reduced"};
	d2.parameter_types = {LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	FunctionDescription d1;
	d1.description = "Model likelihood-ratio test of a fitted model against its null (intercept-only) model, on "
	                 "k - 1 degrees of freedom (the 'Model L.R.' of rms).";
	d1.examples = {"lr_test(logistic_fit_agg(y, [x1, x2]))"};
	d1.categories = {"regression", "hypothesis-test"};
	d1.parameter_names = {"model"};
	d1.parameter_types = {LogicalType::ANY};
	info.descriptions.push_back(std::move(d1));
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
