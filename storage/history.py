# storage/history.py

import sqlite3
import time


class History:
    def __init__(self, path="crypto_lite.db"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row

    def save_level(self, level, key, price):
        self.db.execute(
            """
            INSERT INTO levels
            (symbol, timeframe, type, level_key, price, candle_ts, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                level["symbol"],
                level["timeframe"],
                level["type"],
                key,
                price,
                level["base_candle"]["ts"],
                int(time.time())
            )
        )
        self.db.commit()

    def save_margin_zone(self, zone):
        self.db.execute(
            """
            INSERT INTO margin_zones
            (symbol, timeframe, side, low, high, source_ts, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                zone["symbol"],
                zone["timeframe"],
                zone["side"],
                zone["zone"]["low"],
                zone["zone"]["high"],
                zone["source"]["ts"],
                zone["meta"]["strength"],
                int(time.time())
            )
        )
        self.db.commit()

    def save_confluence(self, event):
        self.db.execute(
            """
            INSERT INTO confluence
            (symbol, timeframe, level_key, level_price,
             zone_low, zone_high, level_ts, margin_ts, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["symbol"],
                event["timeframe"],
                event["level_key"],
                event["level_price"],
                event["margin_zone"]["low"],
                event["margin_zone"]["high"],
                event["level_ts"],
                event["margin_ts"],
                event["meta"]["strength"],
                int(time.time())
            )
        )
        self.db.commit()

    def save_candle_event(self, symbol, timeframe, event, candle_ts, range_):
        self.db.execute(
            """
            INSERT INTO candle_events
            (symbol, timeframe, event, candle_ts, range, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                timeframe,
                event,
                candle_ts,
                range_,
                int(time.time())
            )
        )
        self.db.commit()