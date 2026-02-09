# engine/margin/contracts.py

from typing import Literal, TypedDict


class MarginZone(TypedDict):
    symbol: str
    timeframe: Literal["1h", "4h"]
    side: Literal["long", "short"]

    zone: {
        "low": float,
        "high": float
    }

    source: {
        "ts": int,
        "price": float
    }

    meta: {
        "type": Literal["CME"],
        "strength": float
    }