# engine/levels/mtf.py

from engine.levels.contracts import LevelContract


def build_mtf_levels(
    candles: list[dict],
    parent_level: LevelContract,
    price_decimal: int
) -> list[LevelContract]:
    """
    candles — свечи МТФ внутри диапазона STF
    parent_level — уровень STF
    """

    results: list[LevelContract] = []

    for c in candles:
        open_ = c["open"]
        high = c["high"]
        low = c["low"]
        close = c["close"]

        color = "green" if close >= open_ else "red"

        body = abs(close - open_)
        full_range = high - low
        if full_range == 0:
            continue

        body_pct = round((body / full_range) * 100, 2)

        # фильтр МТФ — твой
        if body_pct < 50:
            continue

        levels = {
            "A": round(high, price_decimal),
            "C": round(open_, price_decimal),
            "D": round(low, price_decimal),
            "F": round((high + low) / 2, price_decimal),
            "X": round(high, price_decimal),
            "Y": round(low, price_decimal),
        }

        level: LevelContract = {
            "symbol": parent_level["symbol"],
            "timeframe": parent_level["timeframe"],
            "type": "MTF",

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
                "parent": parent_level["base_candle"]["ts"],
            }
        }

        results.append(level)

    return results