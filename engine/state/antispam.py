# engine/state/antispam.py

class AntiSpam:
    def __init__(self):
        self.seen = set()

    def key(self, event) -> str:
        return (
            f"{event['symbol']}:"
            f"{event['timeframe']}:"
            f"{event['level_key']}:"
            f"{event['margin_ts']}"
        )

    def is_new(self, event) -> bool:
        k = self.key(event)
        if k in self.seen:
            return False
        self.seen.add(k)
        return True