# cme/sync.py

from cme.loader import load_cme_context
from cme.storage.sqlite import (
    init_db,
    insert_settlement,
    insert_expiration,
)
from cme.storage.cache import cache_is_valid, mark_cache_updated


def sync_cme(product_code: str, force: bool = False):
    init_db()

    if not force and cache_is_valid():
        return "CME cache valid — sync skipped"

    ctx = load_cme_context(product_code)

    for s in ctx["settlements"]:
        insert_settlement(
            s.symbol,
            s.trade_date.isoformat(),
            s.low,
            s.high,
        )

    for e in ctx["expirations"]:
        insert_expiration(
            e.symbol,
            e.expiration_date.isoformat(),
        )

    mark_cache_updated()
    return "CME sync completed"
