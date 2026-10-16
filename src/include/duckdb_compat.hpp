#pragma once

// Helpers that compile against DuckDB 1.4 LTS, 1.5.x and 2.0.
//
// DuckDB's headers expose no version macro, so every helper detects the API it
// needs: from the shape of DuckDB's own callback typedefs (e.g. the arity of
// aggregate_initialize_t), or with SFINAE on the presence of a member (accessors
// such as GetReturnType() replace public fields across versions). Nothing here
// changes behaviour; on 1.4/1.5 the adapters forward to the callback unchanged.

#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parser.hpp"

// DuckDB 2.0 split the vector helpers (ListVector, StructVector, ...) out of
// vector.hpp into one header each.
#if defined(__has_include)
#if __has_include("duckdb/common/vector/list_vector.hpp")
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/string_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#endif
#endif

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Names
//===--------------------------------------------------------------------===//

//! The name type of STRUCT children and functions: std::string before DuckDB 2.0,
//! Identifier (explicitly constructible from a std::string) from 2.0 on.
using DuckName = typename child_list_t<LogicalType>::value_type::first_type;

//! A runtime string as a DuckDB name (struct child, function, function set).
inline DuckName AsDuckName(const std::string &name) {
	return DuckName(name);
}

//! Build a STRUCT child entry from a runtime string name.
inline std::pair<DuckName, LogicalType> MakeChild(const std::string &name, LogicalType type) {
	return std::make_pair(DuckName(name), std::move(type));
}

//! A DuckDB name (std::string or Identifier) as a std::string.
template <class NAME>
inline const std::string &NameString(const NAME &name) {
	return static_cast<const std::string &>(name);
}

//===--------------------------------------------------------------------===//
// Function accessors (public fields up to 1.5, accessors from 1.5 on)
//===--------------------------------------------------------------------===//

namespace compat_detail {
template <class T>
auto ReturnType(T &x, int) -> decltype(x.GetReturnType()) {
	return x.GetReturnType();
}
template <class T>
auto ReturnType(T &x, long) -> decltype((x.return_type)) {
	return x.return_type;
}
template <class T>
auto SetReturnType(T &x, LogicalType type, int) -> decltype(x.SetReturnType(type), void()) {
	x.SetReturnType(std::move(type));
}
template <class T>
void SetReturnType(T &x, LogicalType type, long) {
	x.return_type = std::move(type);
}
template <class T>
auto Arguments(T &x, int) -> decltype(x.GetArguments()) {
	return x.GetArguments();
}
template <class T>
auto Arguments(T &x, long) -> decltype((x.arguments)) {
	return x.arguments;
}
template <class T>
auto SetNullHandling(T &x, FunctionNullHandling value, int) -> decltype(x.SetNullHandling(value), void()) {
	x.SetNullHandling(value);
}
template <class T>
void SetNullHandling(T &x, FunctionNullHandling value, long) {
	x.null_handling = value;
}
template <class T>
auto SetOrderDependent(T &x, AggregateOrderDependent value, int) -> decltype(x.SetOrderDependent(value), void()) {
	x.SetOrderDependent(value);
}
template <class T>
void SetOrderDependent(T &x, AggregateOrderDependent value, long) {
	x.order_dependent = value;
}
template <class T>
auto SetArgumentType(T &x, idx_t i, LogicalType type, int) -> decltype(x.GetArguments()[i] = type, void()) {
	x.GetArguments()[i] = std::move(type);
}
template <class T>
auto SetArgumentType(T &x, idx_t i, LogicalType type, long)
    -> decltype(x.GetSignature().GetParameter(i).SetType(type), void()) {
	x.GetSignature().GetParameter(i).SetType(std::move(type));
}
template <class T>
auto SetArgumentType(T &x, idx_t i, LogicalType type, ...) -> decltype(x.arguments[i] = type, void()) {
	x.arguments[i] = std::move(type);
}
template <class T>
auto SetEntryName(T &x, const std::string &name, int) -> decltype(x.SetName(AsDuckName(name)), void()) {
	x.SetName(AsDuckName(name));
}
template <class T>
void SetEntryName(T &x, const std::string &name, long) {
	x.name = name;
}
template <class T>
auto SetSchema(T &x, const std::string &schema, int) -> decltype(x.SetSchema(AsDuckName(schema)), void()) {
	x.SetSchema(AsDuckName(schema));
}
template <class T>
void SetSchema(T &x, const std::string &schema, long) {
	x.schema = schema;
}
template <class T>
auto Name(T &x, int) -> decltype(x.GetName()) {
	return x.GetName();
}
template <class T>
auto Name(T &x, long) -> decltype((x.name)) {
	return x.name;
}
template <class T, class FN>
auto SetFunctionCallback(T &x, FN fn, int) -> decltype(x.SetFunctionCallback(fn), void()) {
	x.SetFunctionCallback(std::move(fn));
}
template <class T, class FN>
void SetFunctionCallback(T &x, FN fn, long) {
	x.function = std::move(fn);
}
template <class T>
auto BindInfo(T &x, int) -> decltype(x.BindInfo()) {
	return x.BindInfo();
}
template <class T>
auto BindInfo(T &x, long) -> decltype((x.bind_info)) {
	return x.bind_info;
}
} // namespace compat_detail

//! The return type of a function or an expression.
template <class T>
auto ReturnTypeOf(T &x) -> decltype(compat_detail::ReturnType(x, 0)) {
	return compat_detail::ReturnType(x, 0);
}
template <class T>
void SetReturnTypeOf(T &x, LogicalType type) {
	compat_detail::SetReturnType(x, std::move(type), 0);
}
//! The (mutable for a non-const function) argument types of a function.
template <class T>
auto ArgumentsOf(T &x) -> decltype(compat_detail::Arguments(x, 0)) {
	return compat_detail::Arguments(x, 0);
}
//! Change the type of argument i of a function.
template <class T>
void SetArgumentTypeOf(T &x, idx_t i, LogicalType type) {
	compat_detail::SetArgumentType(x, i, std::move(type), 0);
}
template <class T>
void SetNullHandlingOf(T &x, FunctionNullHandling value) {
	compat_detail::SetNullHandling(x, value, 0);
}
template <class T>
void SetOrderDependentOf(T &x, AggregateOrderDependent value) {
	compat_detail::SetOrderDependent(x, value, 0);
}
//! The name of a function or catalog entry as a std::string.
template <class T>
std::string NameOf(T &x) {
	return NameString(compat_detail::Name(x, 0));
}
//! Replace the execute callback of a scalar function.
template <class T, class FN>
void SetFunctionCallbackOf(T &x, FN fn) {
	compat_detail::SetFunctionCallback(x, std::move(fn), 0);
}
namespace compat_detail {
template <class P>
auto MakeParser(P *, int) -> decltype(P::GetBuiltinParser()) {
	return P::GetBuiltinParser();
}
template <class P>
P MakeParser(P *, long) {
	return P();
}
} // namespace compat_detail

//! A Parser with the default options (DuckDB 2.0 has no default constructor).
inline Parser MakeParser() {
	return compat_detail::MakeParser(static_cast<Parser *>(nullptr), 0);
}

//! Set the schema and name of a CreateInfo.
template <class T>
void SetCreateInfoName(T &info, const std::string &schema, const std::string &name) {
	compat_detail::SetSchema(info, schema, 0);
	compat_detail::SetEntryName(info, name, 0);
}
//! The bind data of a BoundFunctionExpression.
template <class T>
auto BindInfoOf(T &x) -> decltype(compat_detail::BindInfo(x, 0)) {
	return compat_detail::BindInfo(x, 0);
}

//===--------------------------------------------------------------------===//
// Vector writes (DuckDB 2.0 made FlatVector::GetData / Validity read-only)
//===--------------------------------------------------------------------===//

namespace compat_detail {
template <class T, class FV>
auto FlatData(FV *, Vector &v, int) -> decltype(FV::template GetDataMutable<T>(v)) {
	return FV::template GetDataMutable<T>(v);
}
template <class T, class FV>
T *FlatData(FV *, Vector &v, long) {
	return FV::template GetData<T>(v);
}
template <class FV>
auto FlatValidity(FV *, Vector &v, int) -> decltype(FV::ValidityMutable(v)) {
	return FV::ValidityMutable(v);
}
template <class FV>
ValidityMask &FlatValidity(FV *, Vector &v, long) {
	return FV::Validity(v);
}
} // namespace compat_detail

//! Writable data of a flat vector.
template <class T>
inline T *FlatDataMut(Vector &v) {
	return compat_detail::FlatData<T>(static_cast<FlatVector *>(nullptr), v, 0);
}

//! Writable validity mask of a flat vector.
inline ValidityMask &FlatValidityMut(Vector &v) {
	return compat_detail::FlatValidity(static_cast<FlatVector *>(nullptr), v, 0);
}

//! A STRUCT child vector: StructVector::GetEntries holds unique_ptr<Vector> before
//! DuckDB 2.0 and Vector from 2.0 on.
inline Vector &StructEntry(Vector &entry) {
	return entry;
}
inline const Vector &StructEntry(const Vector &entry) {
	return entry;
}
inline Vector &StructEntry(const unique_ptr<Vector> &entry) {
	return *entry;
}
//! The children of a STRUCT vector, as returned by StructVector::GetEntries.
using StructEntries = typename std::remove_reference<decltype(StructVector::GetEntries(std::declval<Vector &>()))>::type;

//===--------------------------------------------------------------------===//
// Aggregate and scalar callbacks
//===--------------------------------------------------------------------===//

//! Our aggregate Initialize callbacks placement-new one state.
typedef void (*anofox_aggregate_init_t)(data_ptr_t state);
//! Our aggregate Finalize callbacks (DuckDB 2.0 passes an AggregateFinalizeInputData,
//! which derives from AggregateInputData).
typedef void (*anofox_aggregate_finalize_t)(Vector &state, AggregateInputData &aggr_input_data, Vector &result,
                                            idx_t count, idx_t offset);

namespace compat_detail {
template <class FN>
struct FunctionTraits;
template <class R, class... ARGS>
struct FunctionTraits<R (*)(ARGS...)> {
	template <size_t I>
	using Arg = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
};

template <class FN>
struct InitAdapter;
// DuckDB < 2.0: void(const AggregateFunction &, data_ptr_t)
template <class FUNCTION, class STATE>
struct InitAdapter<void (*)(FUNCTION, STATE)> {
	template <anofox_aggregate_init_t INIT>
	static void Call(FUNCTION, STATE state) {
		INIT(state);
	}
};
// DuckDB 2.0: void(AggregateStateInput &, data_ptr_t *states, idx_t count)
template <class INPUT, class STATES, class COUNT>
struct InitAdapter<void (*)(INPUT, STATES, COUNT)> {
	template <anofox_aggregate_init_t INIT>
	static void Call(INPUT, STATES states, COUNT count) {
		for (idx_t i = 0; i < idx_t(count); i++) {
			INIT(states[i]);
		}
	}
};

template <class FN>
struct BindAdapter;
// DuckDB < 2.0: unique_ptr<FunctionData>(ClientContext &, FUNCTION &, vector<unique_ptr<Expression>> &)
template <class R, class FUNCTION>
struct BindAdapter<R (*)(ClientContext &, FUNCTION &, vector<unique_ptr<Expression>> &)> {
	using Function = FUNCTION;
	static constexpr bool kNewApi = false;
	template <R (*BIND)(ClientContext &, Function &, vector<unique_ptr<Expression>> &)>
	static R Call(ClientContext &context, Function &function, vector<unique_ptr<Expression>> &arguments) {
		return BIND(context, function, arguments);
	}
};
// DuckDB 2.0: unique_ptr<FunctionData>(Bind{Aggregate,Scalar}FunctionInput &)
template <class R, class INPUT>
struct BindAdapter<R (*)(INPUT &)> {
	using Function = typename std::remove_reference<decltype(std::declval<INPUT &>().GetBoundFunction())>::type;
	static constexpr bool kNewApi = true;
	template <R (*BIND)(ClientContext &, Function &, vector<unique_ptr<Expression>> &)>
	static R Call(INPUT &input) {
		return BIND(input.GetClientContext(), input.GetBoundFunction(), input.GetArguments());
	}
};
} // namespace compat_detail

//! The input of aggregate Finalize callbacks.
using AggregateFinalizeInput = typename std::remove_reference<
    typename compat_detail::FunctionTraits<aggregate_finalize_t>::template Arg<1>>::type;

//! Adapt one of our Finalize callbacks to aggregate_finalize_t.
template <anofox_aggregate_finalize_t FINALIZE>
void FinalizeAdapter(Vector &state, AggregateFinalizeInput &aggr_input_data, Vector &result, idx_t count,
                     idx_t offset) {
	FINALIZE(state, aggr_input_data, result, count, offset);
}

//! The function a bind callback may modify (e.g. to set the return type):
//! AggregateFunction / ScalarFunction before DuckDB 2.0, the Bound* variant from 2.0 on.
using AggregateBindFunction = compat_detail::BindAdapter<bind_aggregate_function_t>::Function;
using ScalarBindFunction = compat_detail::BindAdapter<bind_scalar_function_t>::Function;

namespace compat_detail {
template <class T>
auto SetFallible(T &x, int) -> decltype(x.SetFallible(), void()) {
	x.SetFallible();
}
template <class T>
void SetFallible(T &, long) {
}
template <class T, class = void>
struct HasSetFallible : std::false_type {};
template <class T>
struct HasSetFallible<T, decltype(std::declval<T &>().SetFallible(), void())> : std::true_type {};
} // namespace compat_detail

// On DuckDB 2.0 the no-op fallback above must never be chosen: an unmarked function
// that throws turns every user-facing error into an InternalException.
static_assert(!compat_detail::BindAdapter<bind_scalar_function_t>::kNewApi ||
                  compat_detail::HasSetFallible<ScalarFunction>::value,
              "DuckDB 2.0 API without ScalarFunction::SetFallible(): update MarkFallible");

//! Mark a scalar function that can throw at execution time. DuckDB 2.0 rejects an
//! execution error from a function that is not marked (InternalException); on
//! earlier versions this is a no-op, so their behaviour is unchanged.
inline void MarkFallible(ScalarFunction &function) {
	if (compat_detail::BindAdapter<bind_scalar_function_t>::kNewApi) {
		compat_detail::SetFallible(function, 0);
	}
}

} // namespace duckdb

//! Wrap our callbacks for an AggregateFunction / ScalarFunction registration.
#define ANOFOX_AGG_INIT(...) (&::duckdb::compat_detail::InitAdapter<::duckdb::aggregate_initialize_t>::Call<__VA_ARGS__>)
#define ANOFOX_AGG_FINALIZE(...) (&::duckdb::FinalizeAdapter<__VA_ARGS__>)
#define ANOFOX_AGG_BIND(...)                                                                                      \
	(&::duckdb::compat_detail::BindAdapter<::duckdb::bind_aggregate_function_t>::Call<__VA_ARGS__>)
#define ANOFOX_SCALAR_BIND(...)                                                                                   \
	(&::duckdb::compat_detail::BindAdapter<::duckdb::bind_scalar_function_t>::Call<__VA_ARGS__>)
