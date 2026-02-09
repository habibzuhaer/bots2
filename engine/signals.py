# bots2/engine/signals.py
class SignalGenerator:
    """
    Анализирует уровни, цену и конфлюэнс для принятия решений.
    Простая, но рабочая логика.
    """
    def __init__(self, 
                 rsi_overbought=70, 
                 rsi_oversold=30,
                 volatility_threshold=0.02):
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volatility_threshold = volatility_threshold

    def calculate_rsi(self, prices, period=14):
        """Рассчитывает RSI по ценам закрытия."""
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50

    def generate(self, symbol, price, levels, confluence):
        """
        Основная функция генерации сигнала.
        
        :return: dict с ключами 'direction' (BUY/SELL), 'strength', 'confidence'
        """
        if not levels:
            return None
        
        # 1. Берем ближайшие ключевые уровни с основного таймфрейма (например, 1h)
        main_tf = '1h'
        if main_tf not in levels:
            main_tf = list(levels.keys())[0]
        
        sup_levels = levels[main_tf].get('supports', [])
        res_levels = levels[main_tf].get('resistances', [])
        
        if not sup_levels or not res_levels:
            return None
        
        nearest_support = sup_levels[0]  # Самый ближайший уровень поддержки
        nearest_resistance = res_levels[0]  # Самый ближайший уровень сопротивления
        
        # 2. Определяем дистанцию до уровней в процентах
        dist_to_support = abs(price - nearest_support) / price if nearest_support else 1.0
        dist_to_resistance = abs(price - nearest_resistance) / price if nearest_resistance else 1.0
        
        # 3. Логика сигналов (пример: отскок от поддержки/сопротивления)
        signal = None
        strength = 'MEDIUM'
        
        # Если цена близко к поддержке и есть конфлюэнс (уровни совпадают на нескольких TF)
        if dist_to_support < 0.005 and confluence.get('strong_supports'):  # В пределах 0.5%
            signal = 'BUY'
            strength = 'STRONG' if confluence.get('strong_supports') > 1 else 'MEDIUM'
        # Если цена близко к сопротивлению
        elif dist_to_resistance < 0.005 and confluence.get('strong_resistances'):
            signal = 'SELL'
            strength = 'STRONG' if confluence.get('strong_resistances') > 1 else 'MEDIUM'
        # Если цена посередине диапазона - смотрим тренд по SMА
        else:
            # (Здесь можно добавить анализ тренда из данных)
            pass
        
        if signal:
            return {
                'direction': signal,
                'strength': strength,
                'price': price,
                'timestamp': pd.Timestamp.now().isoformat(),
                'levels_used': {
                    'support': nearest_support,
                    'resistance': nearest_resistance
                }
            }
        
        return None