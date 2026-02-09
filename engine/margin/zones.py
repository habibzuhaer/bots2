# engine/margin/zones.py

from engine.margin.contracts import MarginZone


def build_margin_zones(
    candles: list[dict],
    symbol: str,
    timeframe: str,
    price_decimal: int
) -> list[MarginZone]:
    """
    candles — свечи STF (1h / 4h)
    логика CME и маржинальных требований — ТВОЯ
    """

    if timeframe not in ("1h", "4h"):
        return []

    zones: list[MarginZone] = []

    for c in candles:
        open_ = c["open"]
        high = c["high"]
        low = c["low"]
        close = c["close"]

        # === ТВОЯ ЛОГИКА CME / MARGIN ===
        impulse = abs(close - open_) / open_ * 100

        if impulse < 1.2:   # пример, заменить на твою формулу
            continue

        if close > open_:
            side = "long"
            zone_low = round(open_, price_decimal)
            zone_high = round(high, price_decimal)
        else:
            side = "short"
            zone_low = round(low, price_decimal)
            zone_high = round(open_, price_decimal)

        zone: MarginZone = {
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,

            "zone": {
                "low": min(zone_low, zone_high),
                "high": max(zone_low, zone_high)
            },

            "source": {
                "ts": c["timestamp"],
                "price": round(close, price_decimal)
            },

            "meta": {
                "type": "CME",
                "strength": round(impulse, 2)
            }
        }

        zones.append(zone)

    return zones