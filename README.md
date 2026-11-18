# Supplier Delay Early Warning

A chronological research benchmark for seller shipment-deadline misses using the Olist Brazilian E-Commerce Public Dataset.

## Data and run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py
python benchmark.py
```

The downloader uses Kaggle's public Olist dataset endpoint and extracts only order, item, and product tables. Raw records are local and excluded from Git.

## Outcome definition

For delivered items with valid timestamps, the label is whether the order-level `order_delivered_carrier_date` is later than the item-level `shipping_limit_date`. The latter is the seller's shipping deadline in this dataset. The prediction point is order purchase; features include seller/product category, item price/freight, promised lead time, and purchase calendar fields.

A late carrier handoff can have many causes. This is a historical deadline-miss proxy, not proof of seller fault or a causal supplier-risk estimate. Multi-item orders can repeat a shared carrier handoff event.

## Observed benchmark

The run contains 110,189 eligible delivered item rows. It uses chronological train/validation/test periods and has no order overlap across partitions. The latest holdout has 22,038 items across 19,414 orders, with an 8.2% deadline-miss rate.

Chronological validation selected a blend of 25% logistic probability (C=10) and 75% smoothed seller-history rate by average precision. On the untouched test window, the blend achieved AP 0.156, ROC-AUC 0.629, and Brier 0.0755; the seller-history baseline achieved AP 0.144, ROC-AUC 0.622, and Brier 0.0755. At the arbitrary 0.5 threshold, the blend found 40 of 1,813 misses (2.2% recall).

| Review capacity | Blend precision / recall | Seller-history precision / recall |
|---:|---:|---:|
| Top 1% | 31.7% / 3.9% | 22.6% / 2.8% |
| Top 5% | 21.1% / 12.8% | 22.3% / 13.6% |
| Top 10% | 19.4% / 23.6% | 19.8% / 24.1% |

The selected blend improves average precision and the top-1% slice, but it does not beat the seller-history baseline at 5% or 10% review capacity. This is not a usable alert or intervention system. No operating capacity, intervention, costs, or external retailer were validated. Full metrics and the source archive fingerprint are in `reports/benchmark.json`.

## Limits

The carrier handoff timestamp is order-level while deadlines are item-level, so multi-item orders may repeat the same handoff event. Only delivered orders are included; cancellations and undelivered orders are out of scope. The data cover a Brazilian marketplace from 2016–2018, so results do not establish performance for current or other supply chains. Scores do not attribute fault or evaluate an intervention.

## Source

Olist Brazilian E-Commerce Public Dataset: [Kaggle source](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Follow the dataset's terms and attribution requirements. Raw tables are not redistributed here.

