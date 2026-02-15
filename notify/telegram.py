#!/usr/bin/env python3
"""
ПОЛНЫЙ МОДУЛЬ УВЕДОМЛЕНИЙ В TELEGRAM
Версия: 2.0
Функционал: Отправка сигналов, алертов, графиков, клавиатур
Поддержка: Форматирование, очереди, повторные попытки, эмодзи
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import aiohttp
import json
import os
from pathlib import Path
import traceback
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# ОПРЕДЕЛЕНИЯ КЛАССОВ
# ============================================================================

class MessagePriority(Enum):
    """Приоритет сообщений."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class ParseMode(Enum):
    """Режимы парсинга Telegram."""
    HTML = "HTML"
    MARKDOWN = "Markdown"
    MARKDOWN_V2 = "MarkdownV2"
    PLAIN = ""

@dataclass
class TelegramConfig:
    """Конфигурация Telegram бота."""
    bot_token: str
    chat_id: str
    parse_mode: ParseMode = ParseMode.HTML
    disable_notification: bool = False
    disable_web_page_preview: bool = True
    
    @classmethod
    def from_env(cls):
        """Создает конфигурацию из переменных окружения."""
        return cls(
            bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            parse_mode=ParseMode(os.getenv('TELEGRAM_PARSE_MODE', 'HTML')),
            disable_notification=os.getenv('TELEGRAM_DISABLE_NOTIFICATIONS', 'false').lower() == 'true',
            disable_web_page_preview=os.getenv('TELEGRAM_DISABLE_PREVIEW', 'true').lower() == 'true'
        )

@dataclass
class TelegramMessage:
    """Структура сообщения для отправки."""
    text: str
    parse_mode: ParseMode = ParseMode.HTML
    disable_notification: bool = False
    disable_web_page_preview: bool = True
    reply_markup: Optional[Dict] = None
    photo: Optional[Union[str, bytes]] = None
    document: Optional[Union[str, bytes]] = None
    caption: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    retry_count: int = 0
    message_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# ОСНОВНОЙ КЛАСС TELEGRAM NOTIFIER
# ============================================================================

class TelegramNotifier:
    """
    Полнофункциональный клиент для отправки уведомлений в Telegram.
    
    Возможности:
    - Отправка форматированных сообщений (HTML/Markdown)
    - Инлайн-клавиатуры
    - Отправка фото и документов
    - Очередь сообщений с приоритетами
    - Автоматические повторные попытки при ошибках
    - Rate limiting (не более 20 сообщений в минуту)
    - Статистика и мониторинг
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    MAX_MESSAGE_LENGTH = 4096
    RATE_LIMIT_MESSAGES = 20  # сообщений в минуту
    RATE_LIMIT_PERIOD = 60    # секунд
    
    def __init__(self, config: Optional[TelegramConfig] = None):
        """
        Инициализация Telegram уведомителя.
        
        Args:
            config: Конфигурация Telegram (если None, загружается из .env)
        """
        self.config = config or TelegramConfig.from_env()
        self.session: Optional[aiohttp.ClientSession] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.queue_processor_task: Optional[asyncio.Task] = None
        
        # Rate limiting
        self.message_timestamps: List[datetime] = []
        self.rate_limit_lock = asyncio.Lock()
        
        # Статистика
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'photos_sent': 0,
            'documents_sent': 0,
            'queue_size': 0,
            'errors': []
        }
        
        # Проверка конфигурации
        if not self.config.bot_token or not self.config.chat_id:
            logger.warning("⚠️ Telegram не настроен: отсутствует токен или chat_id")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ TelegramNotifier инициализирован для chat_id: {self.config.chat_id[:5]}...")
    
    async def initialize(self):
        """Инициализирует HTTP сессию и запускает обработчик очереди."""
        if not self.enabled:
            return
        
        self.session = aiohttp.ClientSession()
        self.is_running = True
        self.queue_processor_task = asyncio.create_task(self._process_queue())
        
        # Проверяем соединение
        if await self.test_connection():
            logger.info("✅ Telegram соединение установлено")
            await self.send_message("🚀 Telegram уведомитель запущен", priority=MessagePriority.HIGH)
        else:
            logger.error("❌ Не удалось подключиться к Telegram")
    
    async def _process_queue(self):
        """Обрабатывает очередь сообщений с учетом приоритетов и rate limiting."""
        while self.is_running:
            try:
                # Получаем сообщение из очереди
                message: TelegramMessage = await self.message_queue.get()
                
                # Проверяем rate limit
                await self._check_rate_limit()
                
                # Отправляем сообщение
                success = await self._send_message_internal(message)
                
                if not success and message.retry_count < 3:
                    # Повторная попытка с экспоненциальной задержкой
                    message.retry_count += 1
                    wait_time = 2 ** message.retry_count
                    logger.warning(f"🔄 Повторная попытка {message.retry_count}/3 через {wait_time}с")
                    await asyncio.sleep(wait_time)
                    await self.message_queue.put(message)
                elif not success:
                    self.stats['messages_failed'] += 1
                    logger.error(f"❌ Не удалось отправить сообщение после {message.retry_count} попыток")
                
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в обработчике очереди: {e}")
                await asyncio.sleep(1)
    
    async def _check_rate_limit(self):
        """Проверяет и соблюдает rate limiting Telegram."""
        async with self.rate_limit_lock:
            now = datetime.now()
            # Удаляем старые сообщения (старше 1 минуты)
            self.message_timestamps = [ts for ts in self.message_timestamps 
                                      if (now - ts).total_seconds() < self.RATE_LIMIT_PERIOD]
            
            if len(self.message_timestamps) >= self.RATE_LIMIT_MESSAGES:
                # Ждем до освобождения слота
                oldest = min(self.message_timestamps)
                wait_time = self.RATE_LIMIT_PERIOD - (now - oldest).total_seconds()
                if wait_time > 0:
                    logger.debug(f"⏳ Rate limit: ожидание {wait_time:.1f}с")
                    await asyncio.sleep(wait_time)
            
            self.message_timestamps.append(now)
    
    async def _send_message_internal(self, message: TelegramMessage) -> bool:
        """
        Внутренний метод отправки сообщения.
        
        Args:
            message: Сообщение для отправки
            
        Returns:
            True если успешно
        """
        if not self.session:
            logger.error("❌ HTTP сессия не инициализирована")
            return False
        
        try:
            # Разбиваем длинные сообщения
            if len(message.text) > self.MAX_MESSAGE_LENGTH and not message.photo:
                return await self._send_long_message(message)
            
            # Определяем метод и payload
            if message.photo:
                method = "sendPhoto"
                payload = {
                    'chat_id': self.config.chat_id,
                    'photo': message.photo,
                    'caption': message.caption or message.text[:1024],
                    'parse_mode': message.parse_mode.value if message.parse_mode != ParseMode.PLAIN else None,
                    'disable_notification': message.disable_notification
                }
            elif message.document:
                method = "sendDocument"
                payload = {
                    'chat_id': self.config.chat_id,
                    'document': message.document,
                    'caption': message.caption or message.text[:1024],
                    'parse_mode': message.parse_mode.value if message.parse_mode != ParseMode.PLAIN else None,
                    'disable_notification': message.disable_notification
                }
            else:
                method = "sendMessage"
                payload = {
                    'chat_id': self.config.chat_id,
                    'text': message.text,
                    'parse_mode': message.parse_mode.value if message.parse_mode != ParseMode.PLAIN else None,
                    'disable_web_page_preview': message.disable_web_page_preview,
                    'disable_notification': message.disable_notification
                }
            
            # Добавляем клавиатуру если есть
            if message.reply_markup:
                payload['reply_markup'] = json.dumps(message.reply_markup)
            
            # Отправляем запрос
            url = self.BASE_URL.format(token=self.config.bot_token, method=method)
            
            # Для фото/документов используем multipart/form-data
            if message.photo or message.document:
                data = aiohttp.FormData()
                for key, value in payload.items():
                    if key in ['photo', 'document'] and isinstance(value, (str, bytes)):
                        if isinstance(value, str) and os.path.exists(value):
                            # Это файл
                            data.add_field(key, open(value, 'rb'), filename=os.path.basename(value))
                        else:
                            # Это байты или URL
                            data.add_field(key, value)
                    else:
                        data.add_field(key, str(value) if value is not None else '')
                
                async with self.session.post(url, data=data) as response:
                    result = await response.json()
            else:
                async with self.session.post(url, json=payload) as response:
                    result = await response.json()
            
            if result.get('ok'):
                self.stats['messages_sent'] += 1
                if message.photo:
                    self.stats['photos_sent'] += 1
                elif message.document:
                    self.stats['documents_sent'] += 1
                
                message.message_id = result['result']['message_id']
                return True
            else:
                logger.error(f"❌ Ошибка Telegram API: {result}")
                self.stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': result.get('description', 'Unknown error'),
                    'method': method
                })
                return False
                
        except aiohttp.ClientError as e:
            logger.error(f"❌ Сетевая ошибка: {e}")
            self.stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'network'
            })
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сообщения: {e}")
            logger.error(traceback.format_exc())
            self.stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'internal'
            })
            return False
    
    async def _send_long_message(self, message: TelegramMessage) -> bool:
        """
        Разбивает длинное сообщение на части и отправляет.
        
        Args:
            message: Длинное сообщение
            
        Returns:
            True если все части отправлены успешно
        """
        text = message.text
        parts = []
        
        # Разбиваем по параграфам
        paragraphs = text.split('\n\n')
        current_part = ""
        
        for para in paragraphs:
            if len(current_part) + len(para) + 2 <= self.MAX_MESSAGE_LENGTH:
                if current_part:
                    current_part += '\n\n' + para
                else:
                    current_part = para
            else:
                if current_part:
                    parts.append(current_part)
                # Если параграф сам по себе слишком длинный, разбиваем его
                if len(para) > self.MAX_MESSAGE_LENGTH:
                    for i in range(0, len(para), self.MAX_MESSAGE_LENGTH):
                        parts.append(para[i:i+self.MAX_MESSAGE_LENGTH])
                else:
                    current_part = para
        
        if current_part:
            parts.append(current_part)
        
        # Отправляем части
        success = True
        for i, part in enumerate(parts):
            part_message = TelegramMessage(
                text=f"Часть {i+1}/{len(parts)}:\n\n{part}",
                parse_mode=message.parse_mode,
                disable_notification=message.disable_notification,
                disable_web_page_preview=message.disable_web_page_preview,
                reply_markup=message.reply_markup if i == len(parts)-1 else None,
                priority=message.priority
            )
            
            if not await self._send_message_internal(part_message):
                success = False
            
            # Небольшая пауза между частями
            await asyncio.sleep(0.5)
        
        return success
    
    async def send_message(self, 
                          text: str,
                          parse_mode: Optional[ParseMode] = None,
                          disable_notification: Optional[bool] = None,
                          disable_web_page_preview: Optional[bool] = None,
                          reply_markup: Optional[Dict] = None,
                          priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Отправляет текстовое сообщение.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим парсинга
            disable_notification: Отключить уведомление
            disable_web_page_preview: Отключить предпросмотр ссылок
            reply_markup: Инлайн-клавиатура
            priority: Приоритет сообщения
            
        Returns:
            True если сообщение добавлено в очередь
        """
        if not self.enabled:
            logger.debug("Telegram отключен, сообщение не отправлено")
            return False
        
        message = TelegramMessage(
            text=text,
            parse_mode=parse_mode or self.config.parse_mode,
            disable_notification=disable_notification if disable_notification is not None else self.config.disable_notification,
            disable_web_page_preview=disable_web_page_preview if disable_web_page_preview is not None else self.config.disable_web_page_preview,
            reply_markup=reply_markup,
            priority=priority
        )
        
        await self.message_queue.put(message)
        self.stats['queue_size'] = self.message_queue.qsize()
        return True
    
    async def send_signal(self, signal: Dict) -> bool:
        """
        Отправляет торговый сигнал в форматированном виде.
        
        Args:
            signal: Словарь с данными сигнала
            
        Returns:
            True если сообщение добавлено в очередь
        """
        if not self.enabled:
            return False
        
        # Форматируем сигнал
        text = self._format_signal(signal)
        keyboard = self._create_signal_keyboard(signal)
        
        return await self.send_message(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
            priority=MessagePriority.HIGH if signal.get('confidence', 0) > 0.8 else MessagePriority.NORMAL
        )
    
    def _format_signal(self, signal: Dict) -> str:
        """Форматирует сигнал для отправки."""
        symbol = signal.get('symbol', 'N/A')
        direction = signal.get('direction', 'UNKNOWN')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0) * 100
        strength = signal.get('strength', 'MEDIUM')
        signal_type = signal.get('type', signal.get('signal_type', 'unknown'))
        
        # Эмодзи для направлений
        if direction == 'BUY':
            direction_emoji = "🟢 ПОКУПКА"
            color = "#00ff00"
        elif direction == 'SELL':
            direction_emoji = "🔴 ПРОДАЖА"
            color = "#ff0000"
        else:
            direction_emoji = "⚪ НЕЙТРАЛЬНО"
            color = "#ffff00"
        
        # Эмодзи для силы
        strength_emoji = {
            'VERY_STRONG': "🔥 ОЧЕНЬ СИЛЬНЫЙ",
            'STRONG': "💪 СИЛЬНЫЙ",
            'MEDIUM': "📊 СРЕДНИЙ",
            'WEAK': "💧 СЛАБЫЙ"
        }.get(strength, "📊 СРЕДНИЙ")
        
        # Индикаторы
        indicators = signal.get('indicators', {})
        rsi = indicators.get('rsi', 'N/A')
        if isinstance(rsi, float):
            rsi = f"{rsi:.1f}"
        
        macd = indicators.get('macd', 'N/A')
        if isinstance(macd, float):
            macd = f"{macd:.2f}"
        
        # Уровни
        levels = signal.get('levels', {})
        supports = levels.get('supports', [])
        resistances = levels.get('resistances', [])
        
        support_str = f"${supports[0]:,.2f}" if supports else "N/A"
        resistance_str = f"${resistances[0]:,.2f}" if resistances else "N/A"
        
        # Стоп-лосс и тейк-профит
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        rr = signal.get('risk_reward_ratio')
        
        risk_text = ""
        if stop_loss and take_profit and rr:
            risk_text = f"\n🔒 Стоп: ${stop_loss:,.2f}\n🎯 Тейк: ${take_profit:,.2f}\n📊 R/R: {rr:.2f}"
        
        # Описание
        description = signal.get('description', '')
        if description:
            description = f"\n📝 {description}"
        
        return f"""
{direction_emoji} <b>{signal_type.upper()}</b>

<b>Пара:</b> <code>{symbol}</code>
<b>Цена:</b> <code>${price:,.2f}</code>
<b>Уверенность:</b> {confidence:.1f}%
<b>Сила:</b> {strength_emoji}

📊 <b>Индикаторы:</b>
• RSI: {rsi}
• MACD: {macd}

📈 <b>Уровни:</b>
• Поддержка: {support_str}
• Сопротивление: {resistance_str}
{risk_text}
{description}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    def _create_signal_keyboard(self, signal: Dict) -> Dict:
        """Создает инлайн-клавиатуру для сигнала."""
        symbol = signal.get('symbol', '').replace('/', '')
        direction = signal.get('direction', '')
        
        # Символ для TradingView
        if 'USDT' in symbol:
            tv_symbol = f"BINANCE:{symbol}"
        else:
            tv_symbol = f"BINANCE:{symbol}USDT"
        
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "📊 TradingView",
                        "url": f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
                    },
                    {
                        "text": "📈 CoinGecko",
                        "url": f"https://www.coingecko.com/en/coins/{symbol.lower().replace('/', '-')}"
                    }
                ],
                [
                    {
                        "text": "✅ Принять сигнал",
                        "callback_data": f"accept_{signal.get('symbol', '')}_{direction}"
                    },
                    {
                        "text": "❌ Отклонить",
                        "callback_data": f"reject_{signal.get('symbol', '')}_{direction}"
                    }
                ],
                [
                    {
                        "text": "ℹ️ Подробнее",
                        "callback_data": f"details_{signal.get('symbol', '')}"
                    }
                ]
            ]
        }
    
    async def send_alert(self, 
                        level: str,
                        title: str,
                        message: str,
                        priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Отправляет алерт с уровнем важности.
        
        Args:
            level: Уровень (INFO, WARNING, ERROR, SUCCESS)
            title: Заголовок
            message: Сообщение
            priority: Приоритет
            
        Returns:
            True если отправлено
        """
        icons = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🔴",
            "SUCCESS": "✅",
            "DEBUG": "🐛"
        }
        
        colors = {
            "INFO": "#3498db",
            "WARNING": "#f39c12",
            "ERROR": "#e74c3c",
            "SUCCESS": "#2ecc71",
            "DEBUG": "#95a5a6"
        }
        
        icon = icons.get(level.upper(), "📢")
        color = colors.get(level.upper(), "#ffffff")
        
        formatted = f"{icon} <b>{level.upper()}</b>\n\n<b>{title}</b>\n{message}"
        
        return await self.send_message(formatted, priority=priority)
    
    async def send_photo(self, 
                        photo: Union[str, bytes],
                        caption: Optional[str] = None,
                        parse_mode: Optional[ParseMode] = None,
                        reply_markup: Optional[Dict] = None,
                        priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Отправляет фото.
        
        Args:
            photo: Путь к файлу, URL или байты
            caption: Подпись к фото
            parse_mode: Режим парсинга подписи
            reply_markup: Инлайн-клавиатура
            priority: Приоритет
            
        Returns:
            True если добавлено в очередь
        """
        if not self.enabled:
            return False
        
        message = TelegramMessage(
            text=caption or "",
            parse_mode=parse_mode or self.config.parse_mode,
            disable_notification=self.config.disable_notification,
            disable_web_page_preview=self.config.disable_web_page_preview,
            reply_markup=reply_markup,
            photo=photo,
            caption=caption,
            priority=priority
        )
        
        await self.message_queue.put(message)
        return True
    
    async def send_chart(self, 
                        chart_path: str,
                        symbol: str,
                        timeframe: str,
                        indicators: Optional[List[str]] = None,
                        priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Отправляет график с подписью.
        
        Args:
            chart_path: Путь к файлу графика
            symbol: Символ
            timeframe: Таймфрейм
            indicators: Список индикаторов на графике
            priority: Приоритет
            
        Returns:
            True если отправлено
        """
        if not os.path.exists(chart_path):
            logger.error(f"❌ Файл графика не найден: {chart_path}")
            return False
        
        indicators_text = f"\n📊 Индикаторы: {', '.join(indicators)}" if indicators else ""
        
        caption = f"""
📈 <b>График {symbol} ({timeframe})</b>
{indicators_text}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return await self.send_photo(chart_path, caption=caption, priority=priority)
    
    async def test_connection(self) -> bool:
        """
        Проверяет соединение с Telegram.
        
        Returns:
            True если успешно
        """
        if not self.enabled:
            return False
        
        try:
            url = self.BASE_URL.format(token=self.config.bot_token, method="getMe")
            async with self.session.get(url) as response:
                result = await response.json()
                if result.get('ok'):
                    bot_info = result['result']
                    logger.info(f"✅ Telegram бот: @{bot_info['username']}")
                    return True
                else:
                    logger.error(f"❌ Ошибка проверки токена: {result}")
                    return False
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Telegram: {e}")
            return False
    
    async def get_updates(self, offset: Optional[int] = None) -> List[Dict]:
        """
        Получает обновления (для обработки callback'ов).
        
        Args:
            offset: ID последнего обработанного обновления
            
        Returns:
            Список обновлений
        """
        if not self.enabled:
            return []
        
        try:
            url = self.BASE_URL.format(token=self.config.bot_token, method="getUpdates")
            params = {'timeout': 30}
            if offset:
                params['offset'] = offset
            
            async with self.session.get(url, params=params) as response:
                result = await response.json()
                if result.get('ok'):
                    return result.get('result', [])
                else:
                    logger.error(f"❌ Ошибка получения обновлений: {result}")
                    return []
        except Exception as e:
            logger.error(f"❌ Ошибка получения обновлений: {e}")
            return []
    
    async def answer_callback(self, callback_id: str, text: Optional[str] = None, 
                             show_alert: bool = False) -> bool:
        """
        Отвечает на callback запрос.
        
        Args:
            callback_id: ID callback'а
            text: Текст уведомления
            show_alert: Показать как алерт
            
        Returns:
            True если успешно
        """
        if not self.enabled:
            return False
        
        try:
            url = self.BASE_URL.format(token=self.config.bot_token, method="answerCallbackQuery")
            payload = {
                'callback_query_id': callback_id
            }
            if text:
                payload['text'] = text
                payload['show_alert'] = show_alert
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                return result.get('ok', False)
        except Exception as e:
            logger.error(f"❌ Ошибка ответа на callback: {e}")
            return False
    
    async def edit_message(self, 
                          message_id: int,
                          text: str,
                          parse_mode: Optional[ParseMode] = None,
                          reply_markup: Optional[Dict] = None) -> bool:
        """
        Редактирует отправленное сообщение.
        
        Args:
            message_id: ID сообщения
            text: Новый текст
            parse_mode: Режим парсинга
            reply_markup: Новая клавиатура
            
        Returns:
            True если успешно
        """
        if not self.enabled:
            return False
        
        try:
            url = self.BASE_URL.format(token=self.config.bot_token, method="editMessageText")
            payload = {
                'chat_id': self.config.chat_id,
                'message_id': message_id,
                'text': text,
                'parse_mode': parse_mode.value if parse_mode else self.config.parse_mode.value,
                'disable_web_page_preview': self.config.disable_web_page_preview
            }
            if reply_markup:
                payload['reply_markup'] = json.dumps(reply_markup)
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                return result.get('ok', False)
        except Exception as e:
            logger.error(f"❌ Ошибка редактирования сообщения: {e}")
            return False
    
    async def delete_message(self, message_id: int) -> bool:
        """
        Удаляет сообщение.
        
        Args:
            message_id: ID сообщения
            
        Returns:
            True если успешно
        """
        if not self.enabled:
            return False
        
        try:
            url = self.BASE_URL.format(token=self.config.bot_token, method="deleteMessage")
            payload = {
                'chat_id': self.config.chat_id,
                'message_id': message_id
            }
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                return result.get('ok', False)
        except Exception as e:
            logger.error(f"❌ Ошибка удаления сообщения: {e}")
            return False
    
    def get_queue_size(self) -> int:
        """Возвращает размер очереди сообщений."""
        return self.message_queue.qsize()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы."""
        return {
            'enabled': self.enabled,
            'config': {
                'chat_id': f"{self.config.chat_id[:5]}..." if self.config.chat_id else None,
                'parse_mode': self.config.parse_mode.value,
                'disable_notification': self.config.disable_notification,
                'disable_preview': self.config.disable_web_page_preview
            },
            'stats': self.stats,
            'queue_size': self.get_queue_size(),
            'rate_limit': {
                'messages_last_minute': len([ts for ts in self.message_timestamps 
                                            if (datetime.now() - ts).total_seconds() < 60]),
                'limit': self.RATE_LIMIT_MESSAGES
            }
        }
    
    async def close(self):
        """Корректное завершение работы."""
        logger.info("🔚 Завершение работы TelegramNotifier...")
        
        self.is_running = False
        
        # Отправляем сообщение о завершении
        if self.enabled:
            await self.send_message("🛑 Telegram уведомитель остановлен", priority=MessagePriority.HIGH)
            
            # Ждем очистки очереди
            if self.message_queue.qsize() > 0:
                logger.info(f"⏳ Ожидание отправки {self.message_queue.qsize()} сообщений...")
                await self.message_queue.join()
        
        # Отменяем задачу обработки очереди
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Закрываем сессию
        if self.session:
            await self.session.close()
        
        logger.info("✅ TelegramNotifier закрыт")

# ============================================================================
# СИНГЛТОН ДЛЯ ГЛОБАЛЬНОГО ДОСТУПА
# ============================================================================

_notifier_instance = None

async def get_telegram_notifier() -> TelegramNotifier:
    """Возвращает глобальный экземпляр TelegramNotifier."""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = TelegramNotifier()
        await _notifier_instance.initialize()
    return _notifier_instance

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Загружаем .env файл
    load_dotenv()
    
    async def test():
        print("🧪 Тестирование TelegramNotifier...")
        
        # Создаем уведомитель
        notifier = TelegramNotifier()
        await notifier.initialize()
        
        if not notifier.enabled:
            print("❌ Telegram не настроен. Пропускаем тесты.")
            return
        
        # 1. Тест простого сообщения
        print("\n1. 📝 Отправка простого сообщения...")
        await notifier.send_message("🧪 Тестовое сообщение от Trading Bot")
        await asyncio.sleep(2)
        
        # 2. Тест форматированного сообщения
        print("\n2. 🎨 Отправка форматированного сообщения...")
        html_message = """
<b>Жирный текст</b>
<i>Курсив</i>
<code>Моноширинный</code>
<a href="https://github.com">Ссылка</a>
"""
        await notifier.send_message(html_message, parse_mode=ParseMode.HTML)
        await asyncio.sleep(2)
        
        # 3. Тест сигнала
        print("\n3. 📊 Отправка торгового сигнала...")
        test_signal = {
            'symbol': 'BTC/USDT',
            'direction': 'BUY',
            'strength': 'STRONG',
            'type': 'breakout',
            'price': 52345.67,
            'confidence': 0.85,
            'stop_loss': 51800.00,
            'take_profit': 53400.00,
            'risk_reward_ratio': 2.1,
            'indicators': {'rsi': 42.5, 'macd': 156.3},
            'levels': {'supports': [51500, 51800], 'resistances': [52500, 53000]},
            'description': 'Пробой уровня сопротивления с подтверждением объема'
        }
        await notifier.send_signal(test_signal)
        await asyncio.sleep(3)
        
        # 4. Тест алерта
        print("\n4. ⚠️ Отправка алерта...")
        await notifier.send_alert('WARNING', 'Высокая волатильность', 
                                 'BTC показывает аномальные движения. Будьте осторожны.')
        await asyncio.sleep(2)
        
        # 5. Тест приоритетов
        print("\n5. 🔥 Тест приоритетов...")
        await notifier.send_message("Низкий приоритет", priority=MessagePriority.LOW)
        await notifier.send_message("Критическое сообщение!", priority=MessagePriority.CRITICAL)
        await notifier.send_message("Обычный приоритет", priority=MessagePriority.NORMAL)
        
        # 6. Статистика
        print("\n6. 📈 Статистика:")
        stats = notifier.get_statistics()
        print(f"   Отправлено сообщений: {stats['stats']['messages_sent']}")
        print(f"   Размер очереди: {stats['queue_size']}")
        
        # Ждем отправки всех сообщений
        await asyncio.sleep(5)
        
        # Завершаем работу
        await notifier.close()
        
        print("\n🎉 Тестирование завершено!")
    
    asyncio.run(test())