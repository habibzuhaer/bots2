# backtest/guard.py

from contracts.SYSTEM_CONTRACT import validate_contract

def ensure_contract_passed():
    validate_contract()
