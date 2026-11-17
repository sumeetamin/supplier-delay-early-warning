"""Download the public Olist dataset and extract only required source tables."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
ARCHIVE = Path("data/olist.zip")
FILES = {
    "olist_orders_dataset.csv": "data/orders.csv",
    "olist_order_items_dataset.csv": "data/items.csv",
    "olist_products_dataset.csv": "data/products.csv",
}


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urlretrieve(URL, ARCHIVE)
    with ZipFile(ARCHIVE) as archive:
        for member, destination in FILES.items():
            Path(destination).write_bytes(archive.read(member))
    print("Extracted the Olist order, item and product tables; raw files stay local.")


if __name__ == "__main__":
    main()

