# engine/confluence/contracts.py

from typing import TypedDict


class ConfluenceEvent(TypedDict):
    symbol: str
    timeframe: str

    level_key: str        # A / C / D / F / X / Y
    level_price: float

    margin_zone: {
        "low": float,
        "high": float,
        "side": str
    }

    level_ts: int
    margin_ts: int

    meta: {
        "strength": float
    }