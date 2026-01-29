# cme/storage/sqlite.py

import sqlite3
from pathlib import Path
from cme.storage.schema import SCHEMA

DB_PATH = Path("data/cme.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_settlement(symbol, trade_date, low, high):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO cme_settlements
            (symbol, trade_date, low, high)
            VALUES (?, ?, ?, ?)
            """,
            (symbol, trade_date, low, high),
        )


def insert_expiration(symbol, expiration_date):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO cme_expirations
            (symbol, expiration_date)
            VALUES (?, ?)
            """,
            (symbol, expiration_date),
        )


def set_meta(key, value):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cme_meta (key, value)
            VALUES (?, ?)
            """,
            (key, value),
        )


def get_meta(key):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT value FROM cme_meta WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else None
