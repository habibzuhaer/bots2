# bots2/engine/confluence.py
class ConfluenceCalculator:
    """Оценивает силу уровней на основе их повторения на разных таймфреймах."""
    
    def evaluate(self, all_levels: dict):
        """all_levels: {'1h': {levels}, '4h': {levels}, ...}"""
        confluence_report = {
            'strong_resistances': [],
            'strong_supports': [],
            'weak_resistances': [],
            'weak_supports': [],
            'score': 0
        }
        
        # Собираем все уровни со всех таймфреймов
        all_res = []
        all_sup = []
        
        for tf, levels in all_levels.items():
            all_res.extend(levels.get('resistances', []))
            all_sup.extend(levels.get('supports', []))
        
        # Ищем уровни, которые повторяются (конфлюэнс)
        from collections import Counter
        res_counter = Counter([round(r, 2) for r in all_res])
        sup_counter = Counter([round(s, 2) for s in all_sup])
        
        # Уровень считается сильным, если встречается на 2+ таймфреймах
        confluence_report['strong_resistances'] = [price for price, count in res_counter.items() if count >= 2]
        confluence_report['strong_supports'] = [price for price, count in sup_counter.items() if count >= 2]
        
        confluence_report['score'] = len(confluence_report['strong_resistances']) + len(confluence_report['strong_supports'])
        
        return confluence_report