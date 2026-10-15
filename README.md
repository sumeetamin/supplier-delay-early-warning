# Supplier Delay Early Warning

A time-ordered late-delivery classifier using order-time data only, with probability queue and chronological holdout metrics.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A time-ordered late-delivery classifier using order-time data only, with probability queue and chronological holdout metrics. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

Your de-identified procurement/ERP delivery export. CSV columns for order date, promised date, actual delivery date, supplier ID, and order-time predictors. See argument defaults; dates are excluded from predictors after deriving lead time/calendar features. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/orders.csv --ordered order_date --promised promised_date --delivered actual_delivery_date --supplier supplier_id --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

The actual delivery date defines the label and is excluded from the model. Use a chronological holdout, monitor supplier mix shifts, and never interpret a risk score as supplier fault.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

Code: MIT. Third-party datasets remain subject to their original terms.
