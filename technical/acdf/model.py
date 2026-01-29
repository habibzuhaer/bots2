# technical/acdf/model.py

from dataclasses import dataclass

@dataclass(frozen=True)
class ACDFLevels:
    A: float
    C: float
    D: float
    F: float
