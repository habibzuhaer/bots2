# engine/state/levels.py

class LevelState:
    def __init__(self):
        self.fired = set()

    def key(self, level, key) -> str:
        return (
            f"{level['symbol']}:"
            f"{level['timeframe']}:"
            f"{level['base_candle']['ts']}:"
            f"{key}"
        )

    def is_new(self, level, key) -> bool:
        k = self.key(level, key)
        if k in self.fired:
            return False
        self.fired.add(k)
        return True