"""
Визуализация результатов бэктеста
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def create_backtest_report(results: Dict[str, Any]) -> str:
    """Создает визуальный отчет по результатам бэктеста"""
    
    try:
        # Создаем директорию для отчетов
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_dir, f"backtest_report_{timestamp}.png")
        
        # Создаем график
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Отчет бэктеста - {datetime.now().strftime("%d.%m.%Y %H:%M")}', 
                    fontsize=16, fontweight='bold')
        
        # 1. График успешности по символам
        _plot_success_by_symbol(axes[0, 0], results)
        
        # 2. График успешности по таймфреймам
        _plot_success_by_timeframe(axes[0, 1], results)
        
        # 3. Сравнение MTF vs STF
        _plot_mtf_vs_stf(axes[1, 0], results)
        
        # 4. Тепловая карта результатов
        _plot_heatmap(axes[1, 1], results)
        
        plt.tight_layout()
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Отчет создан: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Ошибка создания отчета: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""

def _plot_success_by_symbol(ax, results: Dict[str, Any]):
    """График успешности по символам"""
    # Здесь нужно реализовать логику построения графика
    ax.set_title("Успешность по символам")
    ax.set_xlabel("Символы")
    ax.set_ylabel("Успешность (%)")
    ax.grid(True, alpha=0.3)

# ... остальные функции визуализации ...
