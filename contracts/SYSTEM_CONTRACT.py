# contracts/SYSTEM_CONTRACT.py
"""
SYSTEM CONTRACT — CME CONTEXT ENGINE

Этот файл — источник истины.
Если код ему противоречит — код неверен.

------------------------------------------------
1. CME НИКОГДА не является сигналом
2. CME — обязательный контекст
3. Без CME → торговля запрещена
4. Decision = gate, не стратегия
5. Техника не вызывается без can_trade == True
6. Экспирации не зависят от цены
------------------------------------------------
"""

from engine.decision import can_trade
from cme.expirations import days_to_expiration, is_expiration_near
import inspect


# --- CONTRACT 1: CME required ---
def _contract_cme_required():
    try:
        can_trade(0, None)
    except Exception:
        return
    assert False, "Trading allowed without CME context"


# --- CONTRACT 2: Decision purity ---
def _contract_decision_pure():
    src = inspect.getsource(can_trade).lower()
    forbidden = [
        "entry", "order", "position",
        "tp", "sl", "buy", "sell"
    ]
    for w in forbidden:
        assert w not in src, f"Decision leaks execution: {w}"


# --- CONTRACT 3: Expiration isolation ---
def _contract_expiration_independent():
    for fn in (days_to_expiration, is_expiration_near):
        params = inspect.signature(fn).parameters
        assert "price" not in params, "Expiration depends on price"


def validate_contract():
    _contract_cme_required()
    _contract_decision_pure()
    _contract_expiration_independent()
    print("SYSTEM CONTRACT VALID")


if __name__ == "__main__":
    validate_contract()
