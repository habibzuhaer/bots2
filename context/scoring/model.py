# context/scoring/model.py

from dataclasses import dataclass

@dataclass(frozen=True)
class ScoreResult:
    score: float          # 0.0 .. 1.0
    components: dict      # детализация факторов
