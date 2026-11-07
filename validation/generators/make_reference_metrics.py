"""scikit-learn reference values for test/sql/reference/metrics_reference.test (#368, #369).

Usage: python3 validation/generators/make_reference_metrics.py
(scikit-learn 1.9.1 in a throwaway venv; sklearn is not installed globally.)

* classification_metrics_agg: accuracy_score, precision/recall/f1_score(average, zero_division),
  balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix.
  Specificity has no scikit-learn function: R yardstick::spec's definition (per-class
  tn / (tn + fp), averaged as the other metrics; micro = sum tn / sum (tn + fp)) is
  computed here from the scikit-learn confusion matrix (yardstick is not installed).
* regression_metrics_agg: mean_absolute_error, mean_squared_error, root_mean_squared_error,
  mean_absolute_percentage_error, median_absolute_error, r2_score, explained_variance_score,
  max_error; sMAPE = yardstick::smape / 100 = mean(|y - yhat| / ((|y| + |yhat|) / 2)).
* ranking_metrics_agg: ndcg_score(k) per query (averaged over the queries with a relevant
  item); MAP@k, MRR@k, precision@k, recall@k and hit rate@k follow evidently's
  definitions (legacy/metrics/recsys), computed here from the documented formulas.
"""

import numpy as np
from sklearn import metrics as M

np.set_printoptions(precision=17)


def show(name, value):
    print(f"{name}: {float(value)!r}")


def specificity(cm, average, pos=None):
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = n - tp - fp - fn
    per = np.where(tn + fp > 0, tn / np.where(tn + fp > 0, tn + fp, 1), 0.0)
    if average == "binary":
        return per[pos]
    if average == "macro":
        return per.mean()
    if average == "micro":
        return tn.sum() / (tn.sum() + fp.sum())
    support = cm.sum(axis=1)
    return (per * support).sum() / n


def classification(case, y, p, averages, pos=None):
    print(f"## classification {case}")
    labels = sorted(set(y) | set(p))
    cm = M.confusion_matrix(y, p, labels=labels)
    print("labels:", labels)
    print("confusion:", cm.tolist())
    show("accuracy", M.accuracy_score(y, p))
    show("balanced_accuracy", M.balanced_accuracy_score(y, p))
    show("mcc", M.matthews_corrcoef(y, p))
    show("cohen_kappa", M.cohen_kappa_score(y, p))
    for avg in averages:
        kw = {"average": avg, "zero_division": 0}
        if avg == "binary":
            kw["pos_label"] = pos
        print(
            f"{avg}: precision {M.precision_score(y, p, **kw)!r} recall {M.recall_score(y, p, **kw)!r} "
            f"f1 {M.f1_score(y, p, **kw)!r} specificity "
            f"{specificity(cm, avg, labels.index(pos) if avg == 'binary' else None)!r}"
        )


# Binary 0/1 labels (the scores of classification_reference.test, threshold 0.5).
y_bin = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
s_bin = [0.9, 0.1, 0.8, 0.65, 0.4, 0.3, 0.55, 0.55, 0.7, 0.2,
         0.35, 0.95, 0.6, 0.05, 0.45, 0.5, 0.25, 0.75, 0.55, 0.15]
p_bin = [int(s >= 0.5) for s in s_bin]
classification("binary (score >= 0.5)", y_bin, p_bin, ["binary", "macro", "micro", "weighted"], pos=1)
classification("binary pos_label 0", y_bin, p_bin, ["binary"], pos=0)

# Multiclass VARCHAR labels; 'fish' is never predicted, 'bird' never true.
y_mc = ["cat", "dog", "fish", "cat", "dog", "cat", "fish", "dog", "cat", "dog",
        "fish", "cat", "dog", "dog", "cat", "fish", "cat", "dog", "cat", "fish"]
p_mc = ["cat", "dog", "dog", "cat", "cat", "cat", "bird", "dog", "dog", "dog",
        "cat", "cat", "dog", "bird", "cat", "dog", "cat", "dog", "dog", "cat"]
classification("multiclass", y_mc, p_mc, ["macro", "micro", "weighted"])
print("macro zero_division=1: precision",
      repr(M.precision_score(y_mc, p_mc, average="macro", zero_division=1)))

print("## regression")
y = np.array([3.0, -0.5, 2.0, 7.0, 4.2, 0.0, 5.5, 1.3, 8.8, 2.2])
yhat = np.array([2.5, 0.0, 2.1, 7.8, 3.9, 0.4, 5.0, 1.0, 9.6, 2.2])
show("mae", M.mean_absolute_error(y, yhat))
show("mse", M.mean_squared_error(y, yhat))
show("rmse", M.root_mean_squared_error(y, yhat))
show("mape", M.mean_absolute_percentage_error(y, yhat))
d = (np.abs(y) + np.abs(yhat)) / 2
show("smape", np.mean(np.where(d > 0, np.abs(y - yhat) / np.where(d > 0, d, 1), 0.0)))
show("median_ae", M.median_absolute_error(y, yhat))
show("r2", M.r2_score(y, yhat))
show("explained_variance", M.explained_variance_score(y, yhat))
show("max_error", M.max_error(y, yhat))


# Ranking: 4 queries; q4 has no relevant item. No tied scores within a query.
queries = {
    1: ([3, 2, 0, 0, 1, 0], [0.9, 0.3, 0.8, 0.1, 0.5, 0.2]),
    2: ([0, 0, 1, 0, 2], [0.7, 0.6, 0.5, 0.4, 0.3]),
    3: ([1, 0, 0, 0], [0.2, 0.9, 0.8, 0.7]),
    4: ([0, 0, 0], [0.3, 0.2, 0.1]),
}


def evidently(rel, score, k, min_rel=None):
    order = np.argsort(-np.asarray(score), kind="stable")
    rb = np.array([(r >= min_rel) if min_rel is not None else (r > 0) for r in rel], dtype=float)[order]
    R = rb.sum()
    top = min(k, len(rb))
    hits = rb[:top].cumsum()
    ap = sum(hits[i] / (i + 1) * rb[i] for i in range(top)) / R if R else 0.0
    first = next((i for i in range(top) if rb[i]), None)
    rr = 1.0 / (first + 1) if first is not None else 0.0
    h = rb[:top].sum()
    return ap, rr, h / top, (h / R if R else 0.0), float(h > 0)


def ranking(k, include_no_feedback=False, min_rel=None):
    rows = []
    for rel, score in queries.values():
        rb = [(r >= min_rel) if min_rel is not None else (r > 0) for r in rel]
        if not any(rb):
            if include_no_feedback:
                rows.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        gain = [float(b) for b in rb] if min_rel is not None else rel
        nd = M.ndcg_score([gain], [score], k=k)
        rows.append((nd,) + evidently(rel, score, k, min_rel))
    a = np.mean(rows, axis=0)
    print(f"## ranking k={k} include_no_feedback={include_no_feedback} min_relevance={min_rel}: "
          f"n_queries {len(rows)}")
    for name, v in zip(["ndcg", "map", "mrr", "precision", "recall", "hit_rate"], a):
        show(name, v)


ranking(3)
ranking(10)
ranking(3, include_no_feedback=True)
ranking(2, min_rel=2)
