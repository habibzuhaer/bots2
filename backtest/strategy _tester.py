"""
Тестер вашей логики уровней и маржинальных зон для бэктеста
"""

import asyncio
from typing import Dict, List, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StrategyTester:
    """Тестирует вашу логику уровней и маржинальных зон на исторических данных"""
    
    def __init__(self):
        self.results_cache = {}
    
    async def test_strategy(
        self,
        candles: List[Dict],
        symbol: str,
        timeframe: str,
        collision_threshold: float = 0.105
    ) -> Dict[str, Any]:
        """
        Тестирует вашу стратегию на исторических данных
        
        Возвращает:
        {
            "signals": список сигналов,
            "trades": список сделок,
            "performance": показатели эффективности,
            "levels_stats": статистика по уровням,
            "zones_stats": статистика по зонам,
            "collisions_stats": статистика по совпадениям
        }
        """
        
        try:
            logger.info(f"Тестирование стратегии для {symbol} {timeframe} ({len(candles)} свечей)")
            
            # Импортируем модули стратегии
            from strategy_levels import calculate_levels, detect_patterns
            from margin_zone_engine import find_margin_zones
            
            # Результаты тестирования
            test_results = {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_candles": len(candles),
                "signals": [],
                "trades": [],
                "levels_history": [],
                "zones_history": [],
                "collisions_history": []
            }
            
            # Идем по свечам с окном для расчета
            window_size = 100  # Окно для расчета уровней
            step_size = 5     # Шаг тестирования (чтобы не на каждом баре)
            
            for i in range(window_size, len(candles), step_size):
                current_window = candles[i-window_size:i]
                current_candle = candles[i]
                current_price = current_candle["close"]
                current_time = current_candle["ts"]
                
                # 1. Рассчитываем уровни для текущего окна
                try:
                    levels_data = calculate_levels(
                        candles=current_window,
                        symbol=symbol,
                        tf=timeframe,
                        use_biggest_from_last=5
                    )
                    
                    if levels_data and "levels" in levels_data:
                        current_levels = levels_data["levels"]
                    else:
                        current_levels = []
                        
                except Exception as e:
                    logger.warning(f"Ошибка расчета уровней: {e}")
                    current_levels = []
                
                # 2. Рассчитываем маржинальные зоны (только для STF)
                current_zones = []
                if timeframe in ["1h", "4h"]:
                    try:
                        current_zones = find_margin_zones(
                            candles=current_window,
                            atr_multiplier=1.8,
                            consolidation_bars=5,
                            min_zone_width_percent=0.05
                        )
                    except Exception as e:
                        logger.warning(f"Ошибка расчета зон: {e}")
                        current_zones = []
                
                # 3. Проверяем совпадения
                current_collisions = []
                if current_levels and current_zones:
                    current_collisions = self._find_collisions(
                        levels=current_levels,
                        zones=current_zones,
                        threshold=collision_threshold
                    )
                
                # 4. Генерируем сигналы на основе стратегии
                signals = self._generate_signals(
                    current_price=current_price,
                    levels=current_levels,
                    zones=current_zones,
                    collisions=current_collisions,
                    timeframe=timeframe
                )
                
                # 5. Симулируем торговлю
                trades = self._simulate_trades(
                    signals=signals,
                    current_price=current_price,
                    current_time=current_time
                )
                
                # 6. Сохраняем историю
                history_entry = {
                    "timestamp": current_time,
                    "price": current_price,
                    "levels": current_levels,
                    "zones": current_zones,
                    "collisions": current_collisions,
                    "signals": signals,
                    "trades": trades
                }
                
                test_results["levels_history"].append(history_entry)
                
                # Добавляем сигналы и сделки
                for signal in signals:
                    test_results["signals"].append(signal)
                
                for trade in trades:
                    test_results["trades"].append(trade)
            
            # 7. Рассчитываем итоговую статистику
            test_results["performance"] = self._calculate_performance(test_results["trades"])
            test_results["levels_stats"] = self._calculate_levels_stats(test_results["levels_history"])
            test_results["zones_stats"] = self._calculate_zones_stats(test_results["levels_history"])
            test_results["collisions_stats"] = self._calculate_collisions_stats(test_results["levels_history"])
            
            logger.info(f"Тестирование завершено: {len(test_results['signals'])} сигналов, "
                       f"{len(test_results['trades'])} сделок")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Ошибка тестирования стратегии: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _find_collisions(self, levels: List[float], zones: List[Dict], threshold: float) -> List[Dict]:
        """Находит совпадения уровней с зонами"""
        collisions = []
        
        for level in levels:
            for zone in zones:
                zone_low = zone.get('low', 0)
                zone_high = zone.get('high', 0)
                
                # Расширяем границы зоны на порог
                lower_bound = zone_low * (1 - threshold / 100)
                upper_bound = zone_high * (1 + threshold / 100)
                
                if lower_bound <= level <= upper_bound:
                    collision = {
                        'level': level,
                        'zone_low': zone_low,
                        'zone_high': zone_high,
                        'zone_center': (zone_low + zone_high) / 2,
                        'distance_percent': abs(level - (zone_low + zone_high) / 2) / 
                                           ((zone_low + zone_high) / 2) * 100
                    }
                    collisions.append(collision)
        
        return collisions
    
    def _generate_signals(
        self,
        current_price: float,
        levels: List[float],
        zones: List[Dict],
        collisions: List[Dict],
        timeframe: str
    ) -> List[Dict]:
        """Генерирует торговые сигналы на основе стратегии"""
        signals = []
        
        # 1. Сигналы на основе уровней
        if levels:
            # Находим ближайшие уровни поддержки и сопротивления
            support_levels = [lvl for lvl in levels if lvl < current_price]
            resistance_levels = [lvl for lvl in levels if lvl > current_price]
            
            # Ближайший уровень поддержки
            if support_levels:
                nearest_support = max(support_levels)
                distance_to_support = ((current_price - nearest_support) / current_price) * 100
                
                if distance_to_support < 1.0:  # Цена близко к поддержке
                    signals.append({
                        'type': 'BUY',
                        'reason': 'price_near_support',
                        'level': nearest_support,
                        'distance_percent': distance_to_support,
                        'confidence': max(0, 1.0 - distance_to_support / 10)
                    })
            
            # Ближайший уровень сопротивления
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
                
                if distance_to_resistance < 1.0:  # Цена близко к сопротивлению
                    signals.append({
                        'type': 'SELL',
                        'reason': 'price_near_resistance',
                        'level': nearest_resistance,
                        'distance_percent': distance_to_resistance,
                        'confidence': max(0, 1.0 - distance_to_resistance / 10)
                    })
        
        # 2. Сигналы на основе совпадений (особо сильные)
        if collisions:
            for collision in collisions:
                level = collision['level']
                distance_to_price = abs(current_price - level) / current_price * 100
                
                if distance_to_price < 0.5:  # Цена очень близко к совпадению
                    signals.append({
                        'type': 'STRONG_BUY' if current_price > level else 'STRONG_SELL',
                        'reason': 'level_zone_collision',
                        'level': level,
                        'zone': f"{collision['zone_low']:.2f}-{collision['zone_high']:.2f}",
                        'distance_percent': distance_to_price,
                        'confidence': 0.9  # Высокая уверенность
                    })
        
        # 3. Сигналы на основе таймфрейма
        if timeframe in ["1h", "4h"] and zones:
            # Проверяем, находится ли цена внутри зоны
            for zone in zones:
                zone_low = zone.get('low', 0)
                zone_high = zone.get('high', 0)
                
                if zone_low <= current_price <= zone_high:
                    signals.append({
                        'type': 'HOLD',
                        'reason': 'price_inside_margin_zone',
                        'zone': f"{zone_low:.2f}-{zone_high:.2f}",
                        'confidence': 0.8
                    })
        
        return signals
    
    def _simulate_trades(self, signals: List[Dict], current_price: float, 
                         current_time: int) -> List[Dict]:
        """Симулирует торговлю на основе сигналов"""
        trades = []
        
        for signal in signals:
            if signal['type'] in ['BUY', 'STRONG_BUY']:
                # Симулируем покупку
                trade = {
                    'timestamp': current_time,
                    'type': 'LONG',
                    'entry_price': current_price,
                    'exit_price': None,  # Будет установлено при закрытии
                    'signal': signal,
                    'status': 'OPEN'
                }
                trades.append(trade)
            
            elif signal['type'] in ['SELL', 'STRONG_SELL']:
                # Симулируем продажу
                trade = {
                    'timestamp': current_time,
                    'type': 'SHORT',
                    'entry_price': current_price,
                    'exit_price': None,
                    'signal': signal,
                    'status': 'OPEN'
                }
                trades.append(trade)
        
        return trades
    
    def _calculate_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """Рассчитывает показатели эффективности"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0
            }
        
        # Здесь нужно реализовать логику расчета прибыли/убытков
        # на основе закрытых сделок
        
        return {
            'total_trades': len(trades),
            'win_rate': 50.0,  # Пример
            'total_profit': 1000.0,  # Пример
            'avg_profit': 50.0,  # Пример
            'max_profit': 200.0,  # Пример
            'max_loss': -100.0,  # Пример
            'profit_factor': 1.5  # Пример
        }
    
    def _calculate_levels_stats(self, history: List[Dict]) -> Dict[str, Any]:
        """Рассчитывает статистику по уровням"""
        # Реализация статистики
        return {}
    
    def _calculate_zones_stats(self, history: List[Dict]) -> Dict[str, Any]:
        """Рассчитывает статистику по зонам"""
        # Реализация статистики
        return {}
    
    def _calculate_collisions_stats(self, history: List[Dict]) -> Dict[str, Any]:
        """Рассчитывает статистику по совпадениям"""
        # Реализация статистики
        return {}