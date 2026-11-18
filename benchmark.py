"""Time-split seller handoff deadline-miss benchmark on Olist records."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "missed_seller_deadline"
NUMERIC = ["promised_lead_days", "price", "freight_value", "order_month", "order_weekday", "order_hour"]
CATEGORICAL = ["seller_id", "product_category"]


def load_data():
    orders = pd.read_csv("data/orders.csv", parse_dates=[
        "order_purchase_timestamp", "order_delivered_carrier_date"
    ])
    items = pd.read_csv("data/items.csv", parse_dates=["shipping_limit_date"])
    products = pd.read_csv("data/products.csv", usecols=["product_id", "product_category_name"])
    d = items.merge(orders, on="order_id", validate="many_to_one")
    d = d.merge(products, on="product_id", how="left", validate="many_to_one")
    d = d[d["order_status"].eq("delivered")].dropna(subset=[
        "order_purchase_timestamp", "order_delivered_carrier_date", "shipping_limit_date"
    ]).copy()
    d["promised_lead_days"] = (
        d["shipping_limit_date"] - d["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    d = d[d["promised_lead_days"].between(0, 60)].copy()
    d[TARGET] = (d["order_delivered_carrier_date"] > d["shipping_limit_date"]).astype(int)
    d["order_month"] = d["order_purchase_timestamp"].dt.month
    d["order_weekday"] = d["order_purchase_timestamp"].dt.day_of_week
    d["order_hour"] = d["order_purchase_timestamp"].dt.hour
    d["product_category"] = d["product_category_name"].fillna("UNKNOWN")
    return d.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)


def pipeline(c):
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("features", prep),
        ("classifier", LogisticRegression(C=c, max_iter=1500, solver="liblinear")),
    ])


def blend_probabilities(model_probability, seller_probability, logistic_weight):
    """Combine model and seller-history risk probabilities without thresholding."""
    model_probability = np.asarray(model_probability, dtype=float)
    seller_probability = np.asarray(seller_probability, dtype=float)
    if model_probability.shape != seller_probability.shape or model_probability.ndim != 1:
        raise ValueError("model and seller probabilities must be aligned vectors")
    if not np.isfinite(model_probability).all() or not np.isfinite(seller_probability).all():
        raise ValueError("probabilities must be finite")
    if not 0 <= logistic_weight <= 1:
        raise ValueError("logistic_weight must be in [0, 1]")
    return logistic_weight * model_probability + (1.0 - logistic_weight) * seller_probability


def capacity_metrics(y, p, shares=(0.01, 0.05, 0.10)):
    """Measure a score-ranked review queue at explicit capacity shares."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("labels and probabilities must be equally sized non-empty vectors")
    if not np.isfinite(p).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("probabilities must be finite and labels binary")
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("capacity recall requires at least one positive label")
    order = np.argsort(-p, kind="stable")
    result = {}
    for share in shares:
        if not 0 < share <= 1:
            raise ValueError("capacity shares must be in (0, 1]")
        k = max(1, int(np.ceil(len(y) * share)))
        true_positives = int(y[order[:k]].sum())
        precision = true_positives / k
        result[f"{share:.0%}"] = {
            "rows_reviewed": k,
            "true_positives": true_positives,
            "precision": float(precision),
            "recall": float(true_positives / positives),
            "lift_over_test_prevalence": float(precision / y.mean()) if y.mean() else None,
        }
    return result


def paired_order_cluster_capacity_bootstrap(
    y,
    model_scores,
    baseline_scores,
    order_ids,
    shares=(0.01, 0.05, 0.10),
    *,
    n_resamples=2000,
    random_state=42,
):
    """Bootstrap paired capacity-metric differences by order, keeping items clustered."""
    y = np.asarray(y, dtype=int)
    model_scores = np.asarray(model_scores, dtype=float)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    order_ids = np.asarray(order_ids)
    if y.ndim != 1 or not len(y) or model_scores.shape != y.shape or baseline_scores.shape != y.shape:
        raise ValueError("labels and both score arrays must be aligned non-empty vectors")
    if order_ids.shape != y.shape or pd.isna(order_ids).any():
        raise ValueError("every label row must have one non-missing order id")
    if not np.isfinite(model_scores).all() or not np.isfinite(baseline_scores).all():
        raise ValueError("score arrays must be finite")
    if not set(np.unique(y)).issubset({0, 1}) or y.sum() == 0:
        raise ValueError("labels must be binary and include a positive case")
    if n_resamples < 50:
        raise ValueError("at least 50 cluster resamples are required")

    cluster_codes, cluster_values = pd.factorize(order_ids, sort=False)
    rows_by_cluster = [np.flatnonzero(cluster_codes == code) for code in range(len(cluster_values))]
    baseline_metrics = capacity_metrics(y, baseline_scores, shares=shares)
    model_metrics = capacity_metrics(y, model_scores, shares=shares)
    samples = {
        f"{share:.0%}": {"precision": [], "recall": []}
        for share in shares
    }
    rng = np.random.default_rng(random_state)
    valid_resamples = 0
    for _ in range(n_resamples):
        sampled_clusters = rng.integers(0, len(rows_by_cluster), size=len(rows_by_cluster))
        sampled_rows = np.concatenate([rows_by_cluster[code] for code in sampled_clusters])
        sampled_y = y[sampled_rows]
        if sampled_y.sum() == 0:
            continue
        sampled_model = capacity_metrics(sampled_y, model_scores[sampled_rows], shares=shares)
        sampled_baseline = capacity_metrics(sampled_y, baseline_scores[sampled_rows], shares=shares)
        valid_resamples += 1
        for share in shares:
            key = f"{share:.0%}"
            for measure in ("precision", "recall"):
                samples[key][measure].append(
                    sampled_model[key][measure] - sampled_baseline[key][measure]
                )

    if valid_resamples < 50:
        raise ValueError("fewer than 50 valid positive-containing order-cluster resamples")
    report = {}
    for share in shares:
        key = f"{share:.0%}"
        report[key] = {}
        for measure in ("precision", "recall"):
            values = np.asarray(samples[key][measure], dtype=float)
            report[key][f"model_minus_baseline_{measure}"] = {
                "estimate": float(model_metrics[key][measure] - baseline_metrics[key][measure]),
                "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            }
    return {
        "cluster": "order_id",
        "unique_clusters": int(len(cluster_values)),
        "requested_resamples": int(n_resamples),
        "valid_resamples": int(valid_resamples),
        "random_state": int(random_state),
        "differences": report,
        "method_note": "Paired percentile bootstrap; resample test orders with replacement and retain all their item rows.",
    }


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision_at_0_5": float(precision_score(y, p >= 0.5, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, p >= 0.5, zero_division=0)),
        "predicted_positive_at_0_5": int(pred.sum()),
        "precision_recall_at_capacity": capacity_metrics(y, p),
        "confusion_matrix_tn_fp_fn_tp_at_0_5": [int(tn), int(fp), int(fn), int(tp)],
    }


def main():
    d = load_data()
    # Hold back the latest 20% of observed purchase time; the middle 10% is for C selection.
    q70, q80 = d["order_purchase_timestamp"].quantile([0.70, 0.80])
    train = d[d["order_purchase_timestamp"] <= q70].copy()
    validation = d[(d["order_purchase_timestamp"] > q70) & (d["order_purchase_timestamp"] <= q80)].copy()
    test = d[d["order_purchase_timestamp"] > q80].copy()
    if min(part[TARGET].nunique() for part in (train, validation, test)) < 2:
        raise SystemExit("Each chronological partition must contain both label classes.")

    x_cols = NUMERIC + CATEGORICAL
    train_rate = float(train[TARGET].mean())
    train_counts = train.groupby("seller_id")[TARGET].agg(["sum", "count"])
    train_smooth = (train_counts["sum"] + 20 * train_rate) / (train_counts["count"] + 20)
    validation_seller_rate = validation["seller_id"].map(train_smooth).fillna(train_rate).to_numpy()

    # Select both regularization and seller/model blend weight on validation only.
    chosen_c, chosen_logistic_weight, best_ap, validation_metrics = None, None, -1.0, None
    for c in (0.1, 1.0, 10.0):
        candidate = pipeline(c)
        candidate.fit(train[x_cols], train[TARGET])
        validation_model_probability = candidate.predict_proba(validation[x_cols])[:, 1]
        for logistic_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            validation_score = blend_probabilities(
                validation_model_probability, validation_seller_rate, logistic_weight
            )
            ap = average_precision_score(validation[TARGET], validation_score)
            if ap > best_ap:
                chosen_c = c
                chosen_logistic_weight = logistic_weight
                best_ap = float(ap)
                validation_metrics = metrics(validation[TARGET], validation_score)

    development = pd.concat([train, validation], ignore_index=True)
    model = pipeline(chosen_c)
    model.fit(development[x_cols], development[TARGET])
    test_model_probability = model.predict_proba(test[x_cols])[:, 1]

    # Seller-rate component is estimated only on the pre-test development window.
    overall_rate = float(development[TARGET].mean())
    counts = development.groupby("seller_id")[TARGET].agg(["sum", "count"])
    smooth = (counts["sum"] + 20 * overall_rate) / (counts["count"] + 20)
    seller_baseline = test["seller_id"].map(smooth).fillna(overall_rate).to_numpy()
    pred = blend_probabilities(test_model_probability, seller_baseline, chosen_logistic_weight)

    order_sets = [set(part["order_id"]) for part in (train, validation, test)]
    if any(order_sets[i] & order_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("An order crosses chronological partitions.")

    report = {
        "dataset": "Olist Brazilian E-Commerce Public Dataset, item/order/product tables",
        "estimand": "Whether carrier handoff for a delivered order item occurred after its seller shipping_limit_date.",
        "rows": {
            "eligible_delivered_items": int(len(d)),
            "train": int(len(train)), "validation": int(len(validation)), "test": int(len(test)),
            "unique_test_orders": int(test.order_id.nunique()),
            "train_dates": [str(train.order_purchase_timestamp.min()), str(train.order_purchase_timestamp.max())],
            "validation_dates": [str(validation.order_purchase_timestamp.min()), str(validation.order_purchase_timestamp.max())],
            "test_dates": [str(test.order_purchase_timestamp.min()), str(test.order_purchase_timestamp.max())],
            "order_overlap_across_splits": 0,
        },
        "label_rate": {"train": float(train[TARGET].mean()), "validation": float(validation[TARGET].mean()),
                       "test": float(test[TARGET].mean())},
        "model": "Chronological-validation-selected blend of logistic probability and smoothed seller late rate; selection metric is average precision.",
        "selected_C": chosen_c,
        "selected_logistic_weight": chosen_logistic_weight,
        "selected_seller_rate_weight": 1.0 - chosen_logistic_weight,
        "features": x_cols,
        "excluded_post_order_fields": [
            "order_delivered_carrier_date", "order_delivered_customer_date", "shipping_limit_date",
            "order_status", "order_id", "product_id", "customer_id",
        ],
        "test_metrics": {
            "validation_selected_blend": metrics(test[TARGET], pred),
            "logistic_model": metrics(test[TARGET], test_model_probability),
            "smoothed_seller_historical_late_rate_baseline": metrics(test[TARGET], seller_baseline),
            "global_rate_baseline_average_precision": overall_rate,
        },
        "paired_order_cluster_capacity_comparison": paired_order_cluster_capacity_bootstrap(
            test[TARGET].to_numpy(), pred, seller_baseline, test["order_id"].to_numpy()
        ),
        "validation_metrics": validation_metrics,
        "source_zip_sha256": hashlib.sha256(Path("data/olist.zip").read_bytes()).hexdigest(),
        "limitations": [
            "The carrier handoff timestamp is order-level while the shipping deadline is item-level; multi-item orders can repeat the handoff event.",
            "A deadline miss is not a causal finding that the seller was responsible.",
            "Only delivered historical orders are included; cancellations and undelivered orders are outside this estimand.",
            "This 2016-2018 Brazilian marketplace sample may not transfer to other regions, platforms, or current supply chains.",
            "The dataset has anonymized seller/order identifiers and provides no intervention or operational capacity evaluation.",
            "Capacity-comparison intervals are paired order-cluster bootstrap intervals and are conditional on this model-selection run; they are not adjusted for trying several capacity levels.",
        ],
    }
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


