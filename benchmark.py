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
    chosen_c, best_ap, validation_metrics = None, -1.0, None
    for c in (0.1, 1.0, 10.0):
        candidate = pipeline(c)
        candidate.fit(train[x_cols], train[TARGET])
        p = candidate.predict_proba(validation[x_cols])[:, 1]
        ap = average_precision_score(validation[TARGET], p)
        if ap > best_ap:
            chosen_c, best_ap, validation_metrics = c, float(ap), metrics(validation[TARGET], p)

    development = pd.concat([train, validation], ignore_index=True)
    model = pipeline(chosen_c)
    model.fit(development[x_cols], development[TARGET])
    pred = model.predict_proba(test[x_cols])[:, 1]

    # Seller-rate baseline is estimated only on the pre-test development window.
    overall_rate = float(development[TARGET].mean())
    counts = development.groupby("seller_id")[TARGET].agg(["sum", "count"])
    smooth = (counts["sum"] + 20 * overall_rate) / (counts["count"] + 20)
    seller_baseline = test["seller_id"].map(smooth).fillna(overall_rate).to_numpy()

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
        "model": "Unweighted logistic regression; C selected on chronological validation by average precision.",
        "selected_C": chosen_c,
        "features": x_cols,
        "excluded_post_order_fields": [
            "order_delivered_carrier_date", "order_delivered_customer_date", "shipping_limit_date",
            "order_status", "order_id", "product_id", "customer_id",
        ],
        "test_metrics": {
            "model": metrics(test[TARGET], pred),
            "smoothed_seller_historical_late_rate_baseline": metrics(test[TARGET], seller_baseline),
            "global_rate_baseline_average_precision": overall_rate,
        },
        "validation_metrics": validation_metrics,
        "source_zip_sha256": hashlib.sha256(Path("data/olist.zip").read_bytes()).hexdigest(),
        "limitations": [
            "The carrier handoff timestamp is order-level while the shipping deadline is item-level; multi-item orders can repeat the handoff event.",
            "A deadline miss is not a causal finding that the seller was responsible.",
            "Only delivered historical orders are included; cancellations and undelivered orders are outside this estimand.",
            "This 2016-2018 Brazilian marketplace sample may not transfer to other regions, platforms, or current supply chains.",
            "The dataset has anonymized seller/order identifiers and provides no intervention or operational capacity evaluation.",
        ],
    }
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

