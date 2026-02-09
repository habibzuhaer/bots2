CREATE TABLE IF NOT EXISTS levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timeframe TEXT,
    type TEXT,              -- STF / MTF
    level_key TEXT,         -- A C D F X Y
    price REAL,
    candle_ts INTEGER,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS margin_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timeframe TEXT,         -- 1h / 4h
    side TEXT,              -- long / short
    low REAL,
    high REAL,
    source_ts INTEGER,
    strength REAL,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS confluence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timeframe TEXT,
    level_key TEXT,
    level_price REAL,
    zone_low REAL,
    zone_high REAL,
    level_ts INTEGER,
    margin_ts INTEGER,
    strength REAL,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS candle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timeframe TEXT,
    event TEXT,             -- NEW_CANDLE / CANDLE_CHANGED
    candle_ts INTEGER,
    range REAL,
    created_at INTEGER
);