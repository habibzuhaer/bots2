# engine/levels/contracts.py

from typing import Dict, Literal, TypedDict


class Candle(TypedDict):
    ts: int
    open: float
    high: float
    low: float
    close: float
    color: Literal["green", "red"]


class LevelContract(TypedDict):
    symbol: str
    timeframe: str
    type: Literal["STF", "MTF"]

    base_candle: Candle

    levels: Dict[str, float]  # A, C, D, F, X, Y

    meta: Dict[str, float | str]