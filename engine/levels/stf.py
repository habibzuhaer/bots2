# engine/levels/stf.py

from engine.levels.contracts import LevelContract, Candle


def build_stf_levels(
    candles: list[dict],
    symbol: str,
    timeframe: str,
    price_decimal: int
) -> list[LevelContract]:
    """
    candles: список свечей (OHLCV), уже загруженных
    логика определения импульсной свечи — ТВОЯ (переносится 1 в 1)
    """

    results: list[LevelContract] = []

    for c in candles:
        open_ = c["open"]
        high = c["high"]
        low = c["low"]
        close = c["close"]

        color = "green" if close >= open_ else "red"

        # === ИМПУЛЬСНАЯ СВЕЧА (пример, заменяется твоей логикой) ===
        body = abs(close - open_)
        full_range = high - low
        if full_range == 0:
            continue

        body_pct = round((body / full_range) * 100, 2)

        if body_pct < 60:   # <-- здесь твои реальные условия
            continue

        # === РАСЧЁТ УРОВНЕЙ (ТВОЯ ЛОГИКА) ===
        if color == "green":
            base = open_
            candle_up = high
            candle_down = open_
        else:
            base = open_
            candle_up = open_
            candle_down = low

        levels = {
            "A": round(candle_up, price_decimal),
            "C": round(base, price_decimal),
            "D": round(candle_down, price_decimal),
            "F": round((candle_up + candle_down) / 2, price_decimal),
            "X": round(high, price_decimal),
            "Y": round(low, price_decimal),
        }

        level: LevelContract = {
            "symbol": symbol,
            "timeframe": timeframe,
            "type": "STF",

            "base_candle": {
                "ts": c["timestamp"],
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "color": color,
            },

            "levels": levels,

            "meta": {
                "body_pct": body_pct,
                "direction": "long" if color == "green" else "short",
            }
        }

        results.append(level)

    return results