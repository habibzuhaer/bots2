# engine/state/contracts.py

from typing import TypedDict


class CandleState(TypedDict):
    last_ts: int
    last_range: float


class SymbolState(TypedDict):
    symbol: str
    timeframe: str

    candle: CandleState

    active_levels: set
    active_zones: set