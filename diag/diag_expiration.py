# diag/diag_expiration.py

from cme.expirations import days_to_expiration, is_expiration_near

def run():
    now = "2026-01-10"

    expirations = [
        "2026-01-11",
        "2026-01-13",
        "2026-01-20",
        "2026-02-01"
    ]

    print("=== EXPIRATION DIAGNOSTIC ===")

    for e in expirations:
        days = days_to_expiration(e, now)
        near = is_expiration_near(e, now, window=3)

        print(
            f"exp={e} | days_left={days:2} | expiration_near={near}"
        )


if __name__ == "__main__":
    run()
