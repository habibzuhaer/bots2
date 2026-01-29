# diag/diag_hard_stop_expiration.py

from engine.decision import can_trade

def run():
    print("\nHARD STOP EXPIRATION DIAGNOSTIC\n")

    context = {
        "range": None,
        "bias": "NEUTRAL",
        "expiration_near": True,
    }

    try:
        can_trade(0, context)
    except Exception:
        print("OK   | hard-stop blocks trade")
        print("\nHARD STOP DIAGNOSTIC PASSED")
        return

    raise RuntimeError(
        "FAIL | trade allowed with exp + out of range"
    )

if __name__ == "__main__":
    run()
