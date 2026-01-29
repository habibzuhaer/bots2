# cme/storage/cache.py

from datetime import datetime, timedelta
from cme.storage.sqlite import get_meta, set_meta

CACHE_KEY = "last_cme_sync"
CACHE_TTL_HOURS = 12


def cache_is_valid() -> bool:
    ts = get_meta(CACHE_KEY)
    if not ts:
        return False

    last = datetime.fromisoformat(ts)
    return datetime.utcnow() - last < timedelta(hours=CACHE_TTL_HOURS)


def mark_cache_updated():
    set_meta(CACHE_KEY, datetime.utcnow().isoformat())
