#pragma once
#include "duckdb.hpp"
#include <cmath>
#include <utility>

namespace duckdb {

/**
 * Helpers for the *_fit_predict_agg aggregates: rows whose whole x list is NULL.
 *
 * Every input row must produce one element of the output LIST so positions line
 * up with the group's rows. A row whose x is NULL cannot be fitted or
 * predicted, so it is kept as a non-training row whose yhat (and bounds) come
 * out NULL. Such rows are stored with an empty x vector because the feature
 * count may not be known yet (the NULL row can arrive before the first non-NULL
 * one); PadNullXRows widens them to all-NaN rows once it is.
 *
 * All helpers work on the per-row output vectors that every fit-predict state
 * has: y_all, y_is_null, is_training and x_all.
 */
template <class STATE>
static inline void PushNullXRow(STATE &state, bool y_valid, double y_val) {
    state.y_all.push_back(y_valid ? y_val : std::nan(""));
    state.y_is_null.push_back(!y_valid);
    state.is_training.push_back(false);
    state.x_all.emplace_back();
}

//! Output rows of a state, moved out so they can be re-attached after a merge.
template <class STATE>
struct FitPredictOutputRows {
    decltype(STATE::y_all) y_all;
    decltype(STATE::y_is_null) y_is_null;
    decltype(STATE::is_training) is_training;
    decltype(STATE::x_all) x_all;
};

template <class STATE>
static inline FitPredictOutputRows<STATE> TakeOutputRows(STATE &state) {
    FitPredictOutputRows<STATE> rows;
    rows.y_all = std::move(state.y_all);
    rows.y_is_null = std::move(state.y_is_null);
    rows.is_training = std::move(state.is_training);
    rows.x_all = std::move(state.x_all);
    state.y_all.clear();
    state.y_is_null.clear();
    state.is_training.clear();
    state.x_all.clear();
    return rows;
}

template <class V>
static inline void AppendTo(V &target, const V &source) {
    target.insert(target.end(), source.begin(), source.end());
}

template <class V>
static inline void PrependTo(V &target, V &&front) {
    front.insert(front.end(), target.begin(), target.end());
    target = std::move(front);
}

//! Combine, source not initialized: it can only hold NULL-x rows; append them.
template <class STATE>
static inline void AppendOutputRows(STATE &target, const STATE &source) {
    AppendTo(target.y_all, source.y_all);
    AppendTo(target.y_is_null, source.y_is_null);
    AppendTo(target.is_training, source.is_training);
    AppendTo(target.x_all, source.x_all);
}

//! Combine, target not initialized: put its earlier NULL-x rows back in front.
template <class STATE>
static inline void PrependOutputRows(STATE &target, FitPredictOutputRows<STATE> &&rows) {
    PrependTo(target.y_all, std::move(rows.y_all));
    PrependTo(target.y_is_null, std::move(rows.y_is_null));
    PrependTo(target.is_training, std::move(rows.is_training));
    PrependTo(target.x_all, std::move(rows.x_all));
}

//! Finalize: widen NULL-x rows to n_features NaNs so prediction yields NULL.
template <class STATE>
static inline void PadNullXRows(STATE &state) {
    for (auto &row : state.x_all) {
        if (row.size() != state.n_features) {
            row.assign(state.n_features, std::nan(""));
        }
    }
}

} // namespace duckdb
