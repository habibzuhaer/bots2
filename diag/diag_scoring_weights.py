# diag/diag_scoring_weights.py

from context.scoring.scorer import WEIGHTS, compute_cme_score

def run():
    print("\nSCORING WEIGHTS DIAGNOSTIC\n")

    # 1. веса существуют
    required = {"range_position", "range_width", "expiration"}
    assert required.issubset(WEIGHTS), "Missing scoring weights"

    # 2. сумма весов
    total = sum(WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum != 1.0 ({total})"
    print("OK   | weights sum = 1.0")

    # 3. score в диапазоне
    score = compute_cme_score(
        price=1,
        low=0,
        high=2,
        days_to_expiration=10,
    )
    assert 0.0 <= score.score <= 1.0, "Score out of bounds"
    print("OK   | score bounded [0,1]")

    # 4. нет доминирующего фактора
    for k, w in WEIGHTS.items():
        assert w < 0.8, f"Weight {k} dominates system"

    print("OK   | no dominant factor")
    print("\nSCORING DIAGNOSTIC PASSED")

if __name__ == "__main__":
    run()
