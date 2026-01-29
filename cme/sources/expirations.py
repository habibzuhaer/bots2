# cme/sources/expirations.py

import requests
from datetime import datetime
from cme.models.expiration import CMEExpiration


CME_EXPIRATION_URL = (
    "https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/Futures/"
)


def fetch_expirations(product_code: str):
    resp = requests.get(
        CME_EXPIRATION_URL + product_code,
        timeout=15
    )
    resp.raise_for_status()

    data = resp.json()

    for row in data.get("contracts", []):
        yield CMEExpiration(
            symbol=product_code,
            expiration_date=datetime.strptime(
                row["expirationDate"], "%Y-%m-%d"
            ).date()
        )
