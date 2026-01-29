# diag/diag_acdf_cme_isolation.py

from technical.acdf.adapter import build_acdf_levels
from engine.decision import can_trade

def run():
    print("\nACDF / CME ISOLATION DIAGNOSTIC\n")

    market_state = {
        "impulse_high": 10,
        "impulse_low": 5,
    }

    # 1. ACDF сам по себе работает
    levels = build_acdf_levels(market_state)
    assert levels.A > levels.C
    print("OK   | ACDF independent")

    # 2. вне CME → ACDF не должен вызываться
    try:
        allowed = can_trade(0, None)
    except Exception:
        print("OK   | CME gate blocks ACDF")
        print("\nACDF / CME DIAGNOSTIC PASSED")
        return

    raise RuntimeError("FAIL | ACDF reachable without CME gate")

if __name__ == "__main__":
    run()
