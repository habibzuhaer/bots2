# cme/models/expiration.py

from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class CMEExpiration:
    symbol: str
    expiration_date: date
