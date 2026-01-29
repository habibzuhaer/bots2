# cme/loader.py

from cme.sources.settlements import fetch_settlements
from cme.sources.expirations import fetch_expirations


def load_cme_context(product_code: str):
    return {
        "settlements": list(fetch_settlements(product_code)),
        "expirations": list(fetch_expirations(product_code)),
    }
