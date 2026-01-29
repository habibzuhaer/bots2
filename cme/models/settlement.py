# cme/models/settlement.py

from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class CMESettlement:
    symbol: str
    trade_date: date
    low: float
    high: float
