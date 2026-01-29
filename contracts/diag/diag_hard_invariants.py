# diag/diag_hard_invariants.py
"""
HARD INVARIANTS DIAGNOSTIC

Инварианты:
1. Без CME-контекста → всегда SKIP
2. Decision не может работать без range
3. CME — обязательный gate
"""

from engine.decision import can_trade


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def invariant_no_cme_always_skip():
    header("INVARIANT: NO CME → ALWAYS SKIP")

    invalid_contexts = [
        None,
        {},
        {"bias": "BULLISH"},
        {"range": None},
        {"range": {}, "bias": "NEUTRAL"},
    ]

    for ctx in invalid_contexts:
        try:
            can_trade(0, ctx)
        except Exception:
            print("OK   | rejected invalid CME context")
            continue

        raise RuntimeError(
            "FAIL | trading allowed without CME context"
        )


def invariant_gate_only():
    header("INVARIANT: DECISION IS GATE ONLY")

    src = can_trade.__code__.co_names

    forbidden = {
        "order", "position", "tp", "sl",
        "buy", "sell", "long", "short"
    }

    if forbidden & set(src):
        raise RuntimeError(
            f"FAIL | decision leaks execution logic: {forbidden & set(src)}"
        )

    print("OK   | decision is pure gate")


def run():
    invariant_no_cme_always_skip()
    invariant_gate_only()

    print("\nALL HARD INVARIANTS PASSED")


if __name__ == "__main__":
    run()
