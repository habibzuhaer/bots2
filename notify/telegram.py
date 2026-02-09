# bots2/notify/telegram.py
import os
import asyncio
from telegram import Bot
from telegram.constants import ParseMode

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = Bot(token=self.token) if self.token else None
    
    async def send_signal(self, symbol, direction, price, strength):
        if not self.bot:
            return
            
        message = f"""
🚨 **ТОРГОВЫЙ СИГНАЛ**
📊 Пара: {symbol}
📈 Направление: {direction}
💰 Цена: ${price:.2f}
⚡ Сила: {strength}
🕐 Время: {asyncio.get_event_loop().time()}
        """
        
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN
        )