# bots2/notify/telegram.py
import asyncio
import logging
from typing import Dict, Any, Optional
import aiohttp
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    parse_mode: str = "HTML"
    disable_notifications: bool = False

class TelegramNotifier:
    """
    Асинхронный клиент для отправки уведомлений в Telegram.
    Поддерживает форматирование, кнопки и обработку ошибок.
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    
    def __init__(self, config: Optional[TelegramConfig] = None):
        self.config = config or self._load_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self._message_queue = asyncio.Queue()
        self._is_running = False
        
    def _load_config(self) -> TelegramConfig:
        """Загружает конфигурацию из переменных окружения или файла."""
        import os
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        if not token or not chat_id:
            logger.warning("Telegram конфигурация не найдена. Уведомления отключены.")
        
        return TelegramConfig(
            bot_token=token,
            chat_id=chat_id
        )
    
    async def start(self):
        """Инициализирует HTTP сессию и запускает фоновую задачу обработки очереди."""
        if not self.config.bot_token or not self.config.chat_id:
            logger.error("Не заданы токен или chat_id. Telegram не активирован.")
            return
            
        self.session = aiohttp.ClientSession()
        self._is_running = True
        asyncio.create_task(self._process_queue())
        logger.info("TelegramNotifier запущен")
    
    async def stop(self):
        """Корректно останавливает клиент."""
        self._is_running = False
        if self.session:
            await self.session.close()
        logger.info("TelegramNotifier остановлен")
    
    async def send_signal(self, signal_data: Dict[str, Any]):
        """
        Форматирует и отправляет торговый сигнал.
        
        Пример signal_data:
        {
            'symbol': 'BTC/USDT',
            'direction': 'BUY',
            'strength': 'STRONG',
            'price': 50000.50,
            'levels': {'support': 49000, 'resistance': 51000},
            'confidence': 0.85,
            'timestamp': '2024-01-01T12:00:00'
        }
        """
        if not self.config.bot_token:
            return False
            
        message = self._format_signal_message(signal_data)
        keyboard = self._create_inline_keyboard(signal_data)
        
        payload = {
            'chat_id': self.config.chat_id,
            'text': message,
            'parse_mode': self.config.parse_mode,
            'reply_markup': keyboard,
            'disable_notification': self.config.disable_notifications
        }
        
        await self._send_message(payload)
        return True
    
    def _format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Форматирует красивое сообщение со смайлами."""
        symbol = signal.get('symbol', 'N/A')
        direction = signal.get('direction', 'UNKNOWN')
        price = signal.get('price', 0)
        strength = signal.get('strength', 'MEDIUM')
        confidence = signal.get('confidence', 0) * 100
        
        # Смайлы для направлений
        emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        
        # Цветные теги HTML
        direction_tag = f"<b>{direction}</b>"
        strength_color = {
            'STRONG': '#00ff00',
            'MEDIUM': '#ffff00',
            'WEAK': '#ff6600'
        }.get(strength, '#ffffff')
        
        return f"""
{emoji} <b>ТОРГОВЫЙ СИГНАЛ</b> {emoji}

<b>Пара:</b> {symbol}
<b>Направление:</b> {direction_tag}
<b>Сила сигнала:</b> <code>{strength}</code>
<b>Уверенность:</b> {confidence:.1f}%

<b>Текущая цена:</b> ${price:,.2f}

<b>Ближайшие уровни:</b>
• Поддержка: ${signal.get('levels', {}).get('support', 0):,.2f}
• Сопротивление: ${signal.get('levels', {}).get('resistance', 0):,.2f}

<i>Время: {datetime.now().strftime('%H:%M:%S')}</i>
"""
    
    def _create_inline_keyboard(self, signal: Dict[str, Any]) -> Dict:
        """Создает инлайн-кнопки для быстрых действий."""
        symbol = signal.get('symbol', '').replace('/', '')
        return {
            "inline_keyboard": [[
                {
                    "text": "📊 График на TradingView",
                    "url": f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"
                }
            ], [
                {
                    "text": "✅ Сигнал принят",
                    "callback_data": f"signal_ack_{symbol}"
                },
                {
                    "text": "❌ Отклонить",
                    "callback_data": f"signal_reject_{symbol}"
                }
            ]]
        }
    
    async def _send_message(self, payload: Dict):
        """Асинхронная отправка сообщения через Telegram API."""
        url = self.BASE_URL.format(
            token=self.config.bot_token,
            method="sendMessage"
        )
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug("Сообщение отправлено в Telegram")
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка Telegram API: {error_text}")
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")
    
    async def _process_queue(self):
        """Фоновая обработка очереди сообщений (для защиты от спама)."""
        while self._is_running:
            try:
                message = await self._message_queue.get()
                await self._send_message(message)
                await asyncio.sleep(0.5)  # Задержка между сообщениями
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка обработки очереди: {e}")

# Синглтон для удобного использования
_notifier_instance = None

async def get_notifier() -> TelegramNotifier:
    """Возвращает глобальный экземпляр уведомителя."""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = TelegramNotifier()
        await _notifier_instance.start()
    return _notifier_instance

async def send_alert(message: str, level: str = "INFO"):
    """Упрощенная функция для отправки текстовых уведомлений."""
    notifier = await get_notifier()
    
    level_icons = {
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "🔴",
        "SUCCESS": "✅"
    }
    
    formatted_msg = f"{level_icons.get(level, '📢')} <b>{level}</b>\n\n{message}"
    
    payload = {
        'chat_id': notifier.config.chat_id,
        'text': formatted_msg,
        'parse_mode': 'HTML'
    }
    
    await notifier._send_message(payload)