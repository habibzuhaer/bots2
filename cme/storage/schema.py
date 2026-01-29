# cme/storage/schema.py

SCHEMA = """
CREATE TABLE IF NOT EXISTS cme_settlements (
    symbol TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    low REAL NOT NULL,
    high REAL NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS cme_expirations (
    symbol TEXT NOT NULL,
    expiration_date TEXT NOT NULL,
    PRIMARY KEY (symbol, expiration_date)
);

CREATE TABLE IF NOT EXISTS cme_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
