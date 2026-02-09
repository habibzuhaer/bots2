# engine/state/machine.py

from math import fabs


class StateMachine:
    def __init__(self, tolerance: float = 0.15):
        self.states = {}
        self.tolerance = tolerance

    def _range(self, c) -> float:
        return fabs(c["high"] - c["low"])

    def on_candle(self, symbol: str, timeframe: str, candle: dict) -> str:
        key = f"{symbol}:{timeframe}"

        r = self._range(candle)
        ts = candle["timestamp"]

        if key not in self.states:
            self.states[key] = {
                "last_ts": ts,
                "last_range": r
            }
            return "INIT"

        prev = self.states[key]
        prev_r = prev["last_range"]

        self.states[key]["last_ts"] = ts
        self.states[key]["last_range"] = r

        if r >= prev_r:
            return "NEW_CANDLE"

        if abs(r - prev_r) / prev_r <= self.tolerance:
            return "CANDLE_CHANGED"

        return "IGNORED"