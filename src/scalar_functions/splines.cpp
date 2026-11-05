// Regression-spline bases (#317):
//
//   rcs(x, knots [, options])               -> DOUBLE[]  [x, x', x'', ...]  (rms::rcs, k - 1 columns)
//   lsp(x, knots)                           -> DOUBLE[]  [x, (x - t1)+, ...] (rms::lsp)
//   bspline(x, knots [, degree [, options]]) -> DOUBLE[] B-spline basis (sklearn SplineTransformer)
//   spline_spec(kind, knots [, options])    -> STRUCT    the basis definition (kind, knots, ...)
//   spline_basis(x, spec)                   -> DOUBLE[]  the basis of a spline_spec STRUCT
//   default_knot_quantiles(nk)              -> DOUBLE[]  RMS Table 2.3 knot quantiles
//   restricted_interaction(a, b [, a_nonlinear, b_nonlinear]) -> DOUBLE[] (rms %ia%)
//
// options: rcs {'norm': 0 | 1 | 2} (Hmisc::rcspline.eval norm, default 2);
// bspline {'extrapolation': 'constant' (default) | 'linear' | 'continue' |
// 'periodic' | 'error', 'include_bias': true}. A NULL x, knots, spec or basis
// row gives NULL, as does a NaN / infinite x (missing, as upstream); invalid
// knots or options raise. The bases are computed by anofox-regression
// (splines::{Rcs, Lsp, BSpline, restricted_interaction}).

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/duckdb_compat.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

static constexpr int32_t SPLINE_RCS = 0;
static constexpr int32_t SPLINE_LSP = 1;
static constexpr int32_t SPLINE_BSPLINE = 2;

static const char *const EXTRAPOLATIONS[] = {"constant", "linear", "continue", "periodic", "error"};

static int32_t ExtrapolationCode(const string &name) {
	for (int32_t i = 0; i < 5; i++) {
		if (name == EXTRAPOLATIONS[i]) {
			return i;
		}
	}
	throw InvalidInputException("Invalid extrapolation: '%s'. Valid values are 'constant', 'linear', 'continue', "
	                            "'periodic', 'error'",
	                            name);
}

static const char *KindName(int32_t kind) {
	return kind == SPLINE_RCS ? "rcs" : kind == SPLINE_LSP ? "lsp" : "bspline";
}

static int32_t KindCode(const string &raw, const char *fn) {
	auto kind = StringUtil::Lower(raw);
	if (kind == "rcs") {
		return SPLINE_RCS;
	}
	if (kind == "lsp") {
		return SPLINE_LSP;
	}
	if (kind == "bspline") {
		return SPLINE_BSPLINE;
	}
	throw InvalidInputException("%s: kind must be 'rcs', 'lsp' or 'bspline', got '%s'", fn, raw);
}

//! The spline parameters other than the knots.
struct SplineParams {
	int32_t kind = SPLINE_RCS;
	int32_t norm = 2;
	uint32_t degree = 3;
	int32_t extrapolation = 0;
	bool include_bias = true;

	bool operator==(const SplineParams &o) const {
		return kind == o.kind && norm == o.norm && degree == o.degree && extrapolation == o.extrapolation &&
		       include_bias == o.include_bias;
	}
};

static AnofoxSplineSpec MakeSpec(const SplineParams &p, const vector<double> &knots) {
	AnofoxSplineSpec spec;
	spec.kind = p.kind;
	spec.norm = p.norm;
	spec.extrapolation = p.extrapolation;
	spec.include_bias = p.include_bias;
	spec.degree = p.degree;
	spec.knots = knots.data();
	spec.n_knots = knots.size();
	return spec;
}

//! Calls an FFI function that writes a variable-length result into a buffer,
//! growing the buffer once when the result does not fit.
template <class T, class F>
static bool CallWithBuffer(vector<T> &buffer, F &&call) {
	size_t len = 0;
	if (!call(buffer.data(), buffer.size(), &len)) {
		return false;
	}
	if (len > buffer.size()) {
		buffer.resize(len);
		if (!call(buffer.data(), buffer.size(), &len)) {
			return false;
		}
	}
	buffer.resize(len);
	return true;
}

//! Reads the list at `row` of a flat LIST(DOUBLE) vector; NULL elements become NaN.
static vector<double> ReadDoubleList(Vector &list, idx_t row) {
	auto entry = ListVector::GetData(list)[row];
	auto &child = ListVector::GetEntry(list);
	child.Flatten(ListVector::GetListSize(list));
	auto data = FlatDataMut<double>(child);
	auto &validity = FlatVector::Validity(child);
	vector<double> out(entry.length);
	for (idx_t j = 0; j < entry.length; j++) {
		out[j] = validity.RowIsValid(entry.offset + j) ? data[entry.offset + j] : std::nan("");
	}
	return out;
}

//! Reads the list at `row` of a flat LIST(BOOLEAN) vector; a NULL element raises.
static vector<bool> ReadBoolList(Vector &list, idx_t row, const char *fn, const char *what) {
	auto entry = ListVector::GetData(list)[row];
	auto &child = ListVector::GetEntry(list);
	child.Flatten(ListVector::GetListSize(list));
	auto data = FlatDataMut<bool>(child);
	auto &validity = FlatVector::Validity(child);
	vector<bool> out(entry.length);
	for (idx_t j = 0; j < entry.length; j++) {
		if (!validity.RowIsValid(entry.offset + j)) {
			throw InvalidInputException("%s: %s must not contain NULL", fn, what);
		}
		out[j] = data[entry.offset + j];
	}
	return out;
}

//! Appends `values` as the list of `row` of a LIST(DOUBLE) result.
static void AppendDoubleList(Vector &result, idx_t row, const double *values, idx_t n) {
	auto offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, offset + n);
	auto &child = ListVector::GetEntry(result);
	auto data = FlatDataMut<double>(child);
	for (idx_t j = 0; j < n; j++) {
		data[offset + j] = values[j];
	}
	ListVector::SetListSize(result, offset + n);
	ListVector::GetData(result)[row] = {offset, n};
}

//! Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
//! asserts this in debug builds); the implementations read list arguments by
//! row index, so flatten first.
template <void (*IMPL)(DataChunk &, ExpressionState &, Vector &)>
static void Flattened(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	args.Flatten();
	IMPL(args, state, result);
	result.SetVectorType(all_constant ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
}

//! Evaluates the basis of one row; false for a NULL result.
static bool BasisRow(const char *fn, const SplineParams &p, const vector<double> &knots, double x,
                     vector<double> &buffer) {
	if (!std::isfinite(x)) {
		return false;
	}
	auto spec = MakeSpec(p, knots);
	AnofoxError error;
	bool ok = CallWithBuffer(buffer, [&](double *out, size_t cap, size_t *len) {
		return anofox_spline_basis(&spec, x, out, cap, len, &error);
	});
	if (!ok) {
		ThrowFromFfiError(fn, error);
	}
	return true;
}

// ---------------------------------------------------------------------------
// rcs / lsp / bspline
// ---------------------------------------------------------------------------

struct SplineBindData : public FunctionData {
	string name;
	SplineParams params;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SplineBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<SplineBindData>();
		return name == o.name && params == o.params;
	}
};

static Value ConstantArgument(ClientContext &context, Expression &expr, const string &fn, const char *what) {
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: %s must be a constant", fn, what);
	}
	return ExpressionExecutor::EvaluateScalar(context, expr);
}

static unique_ptr<FunctionData> SplineBind(ClientContext &context, ScalarBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments, const string &name,
                                           int32_t kind) {
	auto data = make_uniq<SplineBindData>();
	data->name = name;
	data->params.kind = kind;
	idx_t options_arg = 2;
	if (kind == SPLINE_BSPLINE && arguments.size() >= 3) {
		auto degree = ConstantArgument(context, *arguments[2], name, "degree");
		if (degree.IsNull()) {
			throw InvalidInputException("%s: degree must not be NULL", name);
		}
		auto d = degree.GetValue<int64_t>();
		if (d < 0 || d > 100) {
			throw InvalidInputException("%s: degree must be between 0 and 100, got %lld", name, (long long)d);
		}
		data->params.degree = static_cast<uint32_t>(d);
		options_arg = 3;
	}
	if (arguments.size() > options_arg) {
		Value options = EvaluateConstantOptions(context, *arguments[options_arg], name);
		auto opts = SplineMapOptions::ParseFromValue(options, name, kind);
		if (opts.norm) {
			data->params.norm = *opts.norm;
		}
		if (opts.extrapolation) {
			data->params.extrapolation = ExtrapolationCode(*opts.extrapolation);
		}
		if (opts.include_bias) {
			data->params.include_bias = *opts.include_bias;
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(data);
}

static unique_ptr<FunctionData> RcsBind(ClientContext &context, ScalarBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	return SplineBind(context, function, arguments, "rcs", SPLINE_RCS);
}
static unique_ptr<FunctionData> LspBind(ClientContext &context, ScalarBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	return SplineBind(context, function, arguments, "lsp", SPLINE_LSP);
}
static unique_ptr<FunctionData> BSplineBind(ClientContext &context, ScalarBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	return SplineBind(context, function, arguments, "bspline", SPLINE_BSPLINE);
}

static void SplineImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<SplineBindData>();
	const idx_t count = args.size();
	auto &x_vec = args.data[0];
	auto &knots_vec = args.data[1];
	auto x_data = FlatDataMut<double>(x_vec);
	vector<double> buffer(16);
	vector<double> knots;
	bool have_knots = false;
	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(x_vec, row) || FlatVector::IsNull(knots_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Reuse the knots of the previous row when they are the same list entry.
		if (!have_knots || row == 0 || ListVector::GetData(knots_vec)[row] != ListVector::GetData(knots_vec)[row - 1]) {
			knots = ReadDoubleList(knots_vec, row);
			have_knots = true;
		}
		if (!BasisRow(bind.name.c_str(), bind.params, knots, x_data[row], buffer)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AppendDoubleList(result, row, buffer.data(), buffer.size());
	}
}

// ---------------------------------------------------------------------------
// spline_spec / spline_basis
// ---------------------------------------------------------------------------

static LogicalType SplineSpecType() {
	child_list_t<LogicalType> children;
	children.push_back(MakeChild("kind", LogicalType::VARCHAR));
	children.push_back(MakeChild("knots", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(MakeChild("norm", LogicalType::INTEGER));
	children.push_back(MakeChild("degree", LogicalType::INTEGER));
	children.push_back(MakeChild("extrapolation", LogicalType::VARCHAR));
	children.push_back(MakeChild("include_bias", LogicalType::BOOLEAN));
	children.push_back(MakeChild("n_columns", LogicalType::BIGINT));
	children.push_back(MakeChild("nonlinear", LogicalType::LIST(LogicalType::BOOLEAN)));
	return LogicalType::STRUCT(std::move(children));
}

static unique_ptr<FunctionData> SplineSpecBind(ClientContext &context, ScalarBindFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	const string name = "spline_spec";
	auto data = make_uniq<SplineBindData>();
	data->name = name;
	auto kind = ConstantArgument(context, *arguments[0], name, "kind");
	if (kind.IsNull()) {
		throw InvalidInputException("%s: kind must not be NULL", name);
	}
	data->params.kind = KindCode(kind.ToString(), name.c_str());
	if (arguments.size() >= 3) {
		Value options = EvaluateConstantOptions(context, *arguments[2], name);
		auto opts = SplineMapOptions::ParseFromValue(options, name, 5);
		const char *kind_name = KindName(data->params.kind);
		if (opts.norm) {
			if (data->params.kind != SPLINE_RCS) {
				throw InvalidInputException("%s: norm applies only to kind 'rcs', not '%s'", name, kind_name);
			}
			data->params.norm = *opts.norm;
		}
		if ((opts.degree || opts.extrapolation || opts.include_bias) && data->params.kind != SPLINE_BSPLINE) {
			throw InvalidInputException(
			    "%s: degree, extrapolation and include_bias apply only to kind 'bspline', not '%s'", name, kind_name);
		}
		if (opts.degree) {
			if (*opts.degree > 100) {
				throw InvalidInputException("%s: degree must be between 0 and 100, got %u", name, *opts.degree);
			}
			data->params.degree = *opts.degree;
		}
		if (opts.extrapolation) {
			data->params.extrapolation = ExtrapolationCode(*opts.extrapolation);
		}
		if (opts.include_bias) {
			data->params.include_bias = *opts.include_bias;
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(data);
}

static void SplineSpecImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<SplineBindData>();
	const auto &p = bind.params;
	const idx_t count = args.size();
	auto &knots_vec = args.data[1];
	auto &entries = StructVector::GetEntries(result);
	auto &kind_vec = StructEntry(entries[0]);
	auto &knots_out = StructEntry(entries[1]);
	auto &norm_vec = StructEntry(entries[2]);
	auto &degree_vec = StructEntry(entries[3]);
	auto &extrap_vec = StructEntry(entries[4]);
	auto &bias_vec = StructEntry(entries[5]);
	auto &ncol_vec = StructEntry(entries[6]);
	auto &nonlinear_out = StructEntry(entries[7]);
	vector<double> knots_buf(16);
	vector<bool> flags;
	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(args.data[0], row) || FlatVector::IsNull(knots_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto knots = ReadDoubleList(knots_vec, row);
		auto spec = MakeSpec(p, knots);
		AnofoxError error;
		// bool is not a contiguous std::vector element type: use a plain buffer.
		vector<uint8_t> nl_buf(knots.size() + p.degree + 2);
		size_t n_knots = 0, n_columns = 0;
		for (int attempt = 0; attempt < 2; attempt++) {
			if (!anofox_spline_info(&spec, knots_buf.data(), knots_buf.size(), &n_knots,
			                        reinterpret_cast<bool *>(nl_buf.data()), nl_buf.size(), &n_columns, &error)) {
				ThrowFromFfiError(bind.name.c_str(), error);
			}
			if (n_knots <= knots_buf.size() && n_columns <= nl_buf.size()) {
				break;
			}
			knots_buf.resize(std::max<size_t>(n_knots, knots_buf.size()));
			nl_buf.resize(std::max<size_t>(n_columns, nl_buf.size()));
		}
		FlatDataMut<string_t>(kind_vec)[row] = StringVector::AddString(kind_vec, KindName(p.kind));
		AppendDoubleList(knots_out, row, knots_buf.data(), n_knots);
		if (p.kind == SPLINE_RCS) {
			FlatDataMut<int32_t>(norm_vec)[row] = p.norm;
		} else {
			FlatVector::SetNull(norm_vec, row, true);
		}
		if (p.kind == SPLINE_BSPLINE) {
			FlatDataMut<int32_t>(degree_vec)[row] = static_cast<int32_t>(p.degree);
			FlatDataMut<string_t>(extrap_vec)[row] = StringVector::AddString(extrap_vec, EXTRAPOLATIONS[p.extrapolation]);
			FlatDataMut<bool>(bias_vec)[row] = p.include_bias;
		} else {
			FlatVector::SetNull(degree_vec, row, true);
			FlatVector::SetNull(extrap_vec, row, true);
			FlatVector::SetNull(bias_vec, row, true);
		}
		FlatDataMut<int64_t>(ncol_vec)[row] = static_cast<int64_t>(n_columns);
		auto offset = ListVector::GetListSize(nonlinear_out);
		ListVector::Reserve(nonlinear_out, offset + n_columns);
		auto nl_data = FlatDataMut<bool>(ListVector::GetEntry(nonlinear_out));
		for (idx_t j = 0; j < n_columns; j++) {
			nl_data[offset + j] = nl_buf[j] != 0;
		}
		ListVector::SetListSize(nonlinear_out, offset + n_columns);
		ListVector::GetData(nonlinear_out)[row] = {offset, n_columns};
	}
}

//! Child positions of a spline spec STRUCT argument (-1: absent).
struct SplineBasisBindData : public FunctionData {
	int64_t kind = -1, knots = -1, norm = -1, degree = -1, extrapolation = -1, include_bias = -1;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SplineBasisBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<SplineBasisBindData>();
		return kind == o.kind && knots == o.knots && norm == o.norm && degree == o.degree &&
		       extrapolation == o.extrapolation && include_bias == o.include_bias;
	}
};

static unique_ptr<FunctionData> SplineBasisBind(ClientContext &context, ScalarBindFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
	const char *fn = "spline_basis";
	auto &type = arguments[1]->return_type;
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s: spec must be a STRUCT as returned by spline_spec, got %s", fn,
		                            type.ToString());
	}
	auto data = make_uniq<SplineBasisBindData>();
	auto &children = StructType::GetChildTypes(type);
	child_list_t<LogicalType> cast_children;
	for (idx_t i = 0; i < children.size(); i++) {
		auto child_name = StringUtil::Lower(NameString(children[i].first));
		LogicalType target = children[i].second;
		if (child_name == "kind") {
			data->kind = i;
			target = LogicalType::VARCHAR;
		} else if (child_name == "knots") {
			data->knots = i;
			target = LogicalType::LIST(LogicalType::DOUBLE);
		} else if (child_name == "norm") {
			data->norm = i;
			target = LogicalType::INTEGER;
		} else if (child_name == "degree") {
			data->degree = i;
			target = LogicalType::INTEGER;
		} else if (child_name == "extrapolation") {
			data->extrapolation = i;
			target = LogicalType::VARCHAR;
		} else if (child_name == "include_bias") {
			data->include_bias = i;
			target = LogicalType::BOOLEAN;
		}
		cast_children.push_back(MakeChild(NameString(children[i].first), target));
	}
	if (data->kind < 0 || data->knots < 0) {
		throw InvalidInputException("%s: spec must have the fields 'kind' and 'knots' (see spline_spec)", fn);
	}
	// Cast the struct so the fields arrive with the types read below.
	SetArgumentTypeOf(function, 1, LogicalType::STRUCT(std::move(cast_children)));
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	return std::move(data);
}

static void SplineBasisImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	const char *fn = "spline_basis";
	auto &bind = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<SplineBasisBindData>();
	const idx_t count = args.size();
	auto &x_vec = args.data[0];
	auto &spec_vec = args.data[1];
	auto x_data = FlatDataMut<double>(x_vec);
	auto &entries = StructVector::GetEntries(spec_vec);
	auto field = [&](int64_t i) -> Vector * { return i < 0 ? nullptr : &StructEntry(entries[i]); };
	Vector *kind_vec = field(bind.kind), *knots_vec = field(bind.knots), *norm_vec = field(bind.norm),
	       *degree_vec = field(bind.degree), *extrap_vec = field(bind.extrapolation),
	       *bias_vec = field(bind.include_bias);
	for (auto v : {kind_vec, knots_vec, norm_vec, degree_vec, extrap_vec, bias_vec}) {
		if (v) {
			v->Flatten(count);
		}
	}
	vector<double> buffer(16);
	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(x_vec, row) || FlatVector::IsNull(spec_vec, row) ||
		    FlatVector::IsNull(*kind_vec, row) || FlatVector::IsNull(*knots_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		SplineParams p;
		p.kind = KindCode(FlatDataMut<string_t>(*kind_vec)[row].GetString(), fn);
		if (norm_vec && !FlatVector::IsNull(*norm_vec, row)) {
			p.norm = FlatDataMut<int32_t>(*norm_vec)[row];
		}
		if (degree_vec && !FlatVector::IsNull(*degree_vec, row)) {
			auto d = FlatDataMut<int32_t>(*degree_vec)[row];
			if (d < 0 || d > 100) {
				throw InvalidInputException("%s: degree must be between 0 and 100, got %d", fn, d);
			}
			p.degree = static_cast<uint32_t>(d);
		}
		if (extrap_vec && !FlatVector::IsNull(*extrap_vec, row)) {
			p.extrapolation =
			    ExtrapolationCode(StringUtil::Lower(FlatDataMut<string_t>(*extrap_vec)[row].GetString()));
		}
		if (bias_vec && !FlatVector::IsNull(*bias_vec, row)) {
			p.include_bias = FlatDataMut<bool>(*bias_vec)[row];
		}
		auto knots = ReadDoubleList(*knots_vec, row);
		if (!BasisRow(fn, p, knots, x_data[row], buffer)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AppendDoubleList(result, row, buffer.data(), buffer.size());
	}
}

// ---------------------------------------------------------------------------
// default_knot_quantiles / restricted_interaction
// ---------------------------------------------------------------------------

static void DefaultKnotQuantilesImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	const char *fn = "default_knot_quantiles";
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	const idx_t count = args.size();
	auto &nk_vec = args.data[0];
	auto nk = FlatDataMut<int64_t>(nk_vec);
	vector<double> buffer(16);
	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(nk_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (nk[row] < 3 || nk[row] > 1000) {
			throw InvalidInputException("%s: nk must be between 3 and 1000, got %lld", fn, (long long)nk[row]);
		}
		AnofoxError error;
		bool ok = CallWithBuffer(buffer, [&](double *out, size_t cap, size_t *len) {
			return anofox_default_knot_quantiles(static_cast<size_t>(nk[row]), out, cap, len, &error);
		});
		if (!ok) {
			ThrowFromFfiError(fn, error);
		}
		AppendDoubleList(result, row, buffer.data(), buffer.size());
	}
}

//! Flags of an rcs / lsp basis: the first column linear, the others nonlinear.
static vector<bool> DefaultFlags(idx_t n) {
	vector<bool> flags(n, true);
	if (n > 0) {
		flags[0] = false;
	}
	return flags;
}

static void RestrictedInteractionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	const char *fn = "restricted_interaction";
	PostHogTelemetry::Instance().RecordFunctionCall(fn);
	const idx_t count = args.size();
	const bool with_flags = args.ColumnCount() == 4;
	vector<double> buffer(16);
	for (idx_t row = 0; row < count; row++) {
		bool any_null = false;
		for (idx_t c = 0; c < args.ColumnCount(); c++) {
			any_null = any_null || FlatVector::IsNull(args.data[c], row);
		}
		if (any_null) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto a = ReadDoubleList(args.data[0], row);
		auto b = ReadDoubleList(args.data[1], row);
		auto a_flags = with_flags ? ReadBoolList(args.data[2], row, fn, "a_nonlinear") : DefaultFlags(a.size());
		auto b_flags = with_flags ? ReadBoolList(args.data[3], row, fn, "b_nonlinear") : DefaultFlags(b.size());
		if (a_flags.size() != a.size() || b_flags.size() != b.size()) {
			throw InvalidInputException("%s: a_nonlinear / b_nonlinear must have one flag per element of a / b", fn);
		}
		// A NULL / non-finite basis value (missing x) gives a missing product row.
		bool finite = true;
		for (auto v : a) {
			finite = finite && std::isfinite(v);
		}
		for (auto v : b) {
			finite = finite && std::isfinite(v);
		}
		if (!finite) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		std::unique_ptr<bool[]> af(new bool[a.size() + 1]), bf(new bool[b.size() + 1]);
		for (idx_t j = 0; j < a.size(); j++) {
			af[j] = a_flags[j];
		}
		for (idx_t j = 0; j < b.size(); j++) {
			bf[j] = b_flags[j];
		}
		AnofoxError error;
		bool ok = CallWithBuffer(buffer, [&](double *out, size_t cap, size_t *len) {
			return anofox_restricted_interaction(a.data(), af.get(), a.size(), b.data(), bf.get(), b.size(), out,
			                                     cap, len, &error);
		});
		if (!ok) {
			ThrowFromFfiError(fn, error);
		}
		AppendDoubleList(result, row, buffer.data(), buffer.size());
	}
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

static void RegisterSplineScalar(ExtensionLoader &loader, const char *name, vector<ScalarFunction> overloads,
                                 const vector<string> &descriptions, const vector<string> &examples,
                                 const vector<vector<string>> &parameter_names) {
	ScalarFunctionSet set(name);
	for (auto &f : overloads) {
		MarkFallible(f);
		set.AddFunction(f);
	}
	CreateScalarFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	for (idx_t i = 0; i < overloads.size(); i++) {
		FunctionDescription desc;
		desc.description = descriptions[std::min(i, descriptions.size() - 1)];
		desc.examples = {examples[std::min(i, examples.size() - 1)]};
		desc.categories = {"regression", "splines"};
		desc.parameter_names = parameter_names[i];
		desc.parameter_types = overloads[i].arguments;
		info.descriptions.push_back(std::move(desc));
	}
	loader.RegisterFunction(std::move(info));
}

void RegisterSplineFunctions(ExtensionLoader &loader) {
	const auto D = LogicalType::DOUBLE;
	const auto DL = LogicalType::LIST(LogicalType::DOUBLE);
	const auto BL = LogicalType::LIST(LogicalType::BOOLEAN);
	const auto I = LogicalType::BIGINT;
	const auto V = LogicalType::VARCHAR;
	const auto ANY = LogicalType::ANY;

	RegisterSplineScalar(
	    loader, "rcs",
	    {ScalarFunction({D, DL}, DL, Flattened<SplineImpl>, ANOFOX_SCALAR_BIND(RcsBind)),
	     ScalarFunction({D, DL, ANY}, DL, Flattened<SplineImpl>, ANOFOX_SCALAR_BIND(RcsBind))},
	    {"Restricted cubic spline basis of x with the given knots, as rms::rcs / Hmisc::rcspline.eval(x, knots, "
	     "inclx = TRUE): [x, x', x'', ...] (k - 1 columns, linear beyond the outer knots). options: norm 0 | 1 | 2 "
	     "(default 2, the terms on the scale of x)."},
	    {"rcs(x, [10.0, 20.0, 30.0, 40.0])", "rcs(x, [10.0, 20.0, 30.0], {'norm': 0})"},
	    {{"x", "knots"}, {"x", "knots", "options"}});

	RegisterSplineScalar(loader, "lsp", {ScalarFunction({D, DL}, DL, Flattened<SplineImpl>, ANOFOX_SCALAR_BIND(LspBind))},
	                     {"Linear spline basis of x, as rms::lsp: [x, (x - t1)+, (x - t2)+, ...]."},
	                     {"lsp(x, [10.0, 20.0])"}, {{"x", "knots"}});

	RegisterSplineScalar(
	    loader, "bspline",
	    {ScalarFunction({D, DL}, DL, Flattened<SplineImpl>, ANOFOX_SCALAR_BIND(BSplineBind)),
	     ScalarFunction({D, DL, I}, DL, Flattened<SplineImpl>, ANOFOX_SCALAR_BIND(BSplineBind)),
	     ScalarFunction({D, DL, I, ANY}, DL, Flattened<SplineImpl>, ANOFOX_SCALAR_BIND(BSplineBind))},
	    {"B-spline basis of x on the given (strictly increasing) base knots, as sklearn SplineTransformer: "
	     "n_knots + degree - 1 columns (degree 3 by default). options: extrapolation 'constant' | 'linear' | "
	     "'continue' | 'periodic' | 'error', include_bias (default true)."},
	    {"bspline(x, [0.0, 2.5, 5.0, 7.5, 10.0])", "bspline(x, [0.0, 5.0, 10.0], 2)",
	     "bspline(x, [0.0, 5.0, 10.0], 3, {'extrapolation': 'linear', 'include_bias': false})"},
	    {{"x", "knots"}, {"x", "knots", "degree"}, {"x", "knots", "degree", "options"}});

	RegisterSplineScalar(
	    loader, "spline_spec",
	    {ScalarFunction({V, DL}, SplineSpecType(), Flattened<SplineSpecImpl>, ANOFOX_SCALAR_BIND(SplineSpecBind)),
	     ScalarFunction({V, DL, ANY}, SplineSpecType(), Flattened<SplineSpecImpl>,
	                    ANOFOX_SCALAR_BIND(SplineSpecBind))},
	    {"Validated definition of a spline basis: {kind, knots, norm, degree, extrapolation, include_bias, "
	     "n_columns, nonlinear}. kind 'rcs' | 'lsp' | 'bspline'; options norm (rcs), degree, extrapolation, "
	     "include_bias (bspline). Keep it next to a fit and expand new data with spline_basis(x, spec)."},
	    {"spline_spec('rcs', [10.0, 20.0, 30.0, 40.0])", "spline_spec('bspline', [0.0, 5.0, 10.0], {'degree': 2})"},
	    {{"kind", "knots"}, {"kind", "knots", "options"}});

	RegisterSplineScalar(
	    loader, "spline_basis",
	    {ScalarFunction({D, ANY}, DL, Flattened<SplineBasisImpl>, ANOFOX_SCALAR_BIND(SplineBasisBind))},
	    {"Basis of x for a spline_spec STRUCT (fields kind and knots, optional norm, degree, extrapolation, "
	     "include_bias)."},
	    {"spline_basis(x, spline_spec('rcs', [10.0, 20.0, 30.0]))"}, {{"x", "spec"}});

	RegisterSplineScalar(loader, "default_knot_quantiles",
	                     {ScalarFunction({I}, DL, Flattened<DefaultKnotQuantilesImpl>)},
	                     {"Default knot quantiles of a restricted cubic spline with nk knots (RMS Table 2.3, "
	                      "Hmisc::rcspline.eval)."},
	                     {"default_knot_quantiles(5)"}, {{"nk"}});

	RegisterSplineScalar(
	    loader, "restricted_interaction",
	    {ScalarFunction({DL, DL}, DL, Flattened<RestrictedInteractionImpl>),
	     ScalarFunction({DL, DL, BL, BL}, DL, Flattened<RestrictedInteractionImpl>)},
	    {"Restricted interaction of two basis rows, as rms %ia%: the products a_i * b_j without the doubly "
	     "nonlinear ones. Without flags the first element of each basis is linear and the others nonlinear "
	     "(rcs / lsp); pass spline_spec(...).nonlinear for other bases."},
	    {"restricted_interaction(rcs(x1, [1.0, 2.0, 3.0, 4.0]), rcs(x2, [1.0, 2.0, 3.0]))",
	     "restricted_interaction(a, b, [false, true], [true, true, true])"},
	    {{"a", "b"}, {"a", "b", "a_nonlinear", "b_nonlinear"}});
}

} // namespace duckdb
