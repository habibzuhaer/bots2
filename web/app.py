# bots2/web/app.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from datetime import datetime, timedelta
import json

# Импорты из нашего проекта
from storage.database import get_database
from notify.telegram import get_notifier, send_alert
from engine_runner import EngineRunner

# Модели Pydantic для валидации запросов/ответов
class SignalCreate(BaseModel):
    symbol: str = Field(..., example="BTC/USDT")
    direction: str = Field(..., example="BUY")
    price: float = Field(..., example=50000.50)
    strength: str = Field("MEDIUM", example="STRONG")
    confidence: Optional[float] = Field(0.8, ge=0, le=1)

class BacktestRequest(BaseModel):
    symbol: str
    start_date: str = Field(..., example="2024-01-01")
    end_date: str = Field(..., example="2024-01-31")
    timeframe: str = "1h"
    strategy: str = "levels_confluence"

# Инициализация FastAPI приложения
app = FastAPI(
    title="Trading Bot API",
    description="REST API для управления торговым ботом и получения сигналов",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы (для будущего веб-интерфейса)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Инициализация шаблонов Jinja2
templates = Jinja2Templates(directory="web/templates")

# Глобальные переменные для управления состоянием
_active_bots: Dict[str, EngineRunner] = {}
_background_tasks = set()

# ===== API ЭНДПОИНТЫ =====

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Главная страница дашборда."""
    # Здесь можно вернуть HTML с графиками
    return """
    <html>
        <head>
            <title>Trading Bot Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>🚀 Trading Bot v2.0</h1>
            <div id="dashboard">
                <p>API доступно по адресу: <a href="/api/docs">/api/docs</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Проверка работоспособности API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "active_bots": len(_active_bots)
    }

@app.get("/api/signals")
async def get_signals(
    symbol: Optional[str] = Query(None, description="Фильтр по символу"),
    limit: int = Query(50, ge=1, le=1000),
    hours: int = Query(24, ge=1, le=720)
):
    """Получение последних торговых сигналов."""
    db = get_database()
    signals = await db.get_recent_signals(symbol, limit, hours)
    
    # Добавляем статистику
    stats = await db.get_signal_statistics(days=7, symbol=symbol)
    
    return {
        "count": len(signals),
        "signals": signals,
        "statistics": stats,
        "request": {
            "symbol": symbol,
            "hours": hours,
            "limit": limit
        }
    }

@app.post("/api/signals")
async def create_signal(signal: SignalCreate, background_tasks: BackgroundTasks):
    """Создание нового торгового сигнала (ручной ввод)."""
    db = get_database()
    
    signal_data = {
        "symbol": signal.symbol,
        "direction": signal.direction,
        "price": signal.price,
        "strength": signal.strength,
        "confidence": signal.confidence,
        "timestamp": datetime.now().isoformat(),
        "levels": {"support": signal.price * 0.98, "resistance": signal.price * 1.02}
    }
    
    # Сохраняем в БД
    signal_id = await db.save_signal(signal_data)
    
    # Отправляем уведомление в Telegram
    background_tasks.add_task(
        send_alert,
        f"Ручной сигнал: {signal.direction} {signal.symbol} по ${signal.price:,.2f}",
        "INFO"
    )
    
    return {
        "id": signal_id,
        "message": "Сигнал сохранен и отправлен",
        "signal": signal_data
    }

@app.post("/api/bot/start")
async def start_bot(
    symbol: str = Query("BTC/USDT", description="Торговая пара"),
    timeframes: str = Query("1h,4h", description="Таймфреймы через запятую")
):
    """Запуск торгового бота для конкретной пары."""
    if symbol in _active_bots:
        raise HTTPException(400, f"Бот для {symbol} уже запущен")
    
    tf_list = [tf.strip() for tf in timeframes.split(",")]
    
    bot = EngineRunner(symbols=[symbol], timeframes=tf_list)
    
    # Запускаем в фоновой задаче
    task = asyncio.create_task(bot.run_continuous(interval_seconds=300))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    
    _active_bots[symbol] = bot
    
    # Уведомление
    await send_alert(f"🤖 Бот запущен для {symbol} | Таймфреймы: {timeframes}", "SUCCESS")
    
    return {
        "status": "started",
        "symbol": symbol,
        "timeframes": tf_list,
        "started_at": datetime.now().isoformat()
    }

@app.post("/api/bot/stop/{symbol}")
async def stop_bot(symbol: str):
    """Остановка бота для конкретной пары."""
    if symbol not in _active_bots:
        raise HTTPException(404, f"Бот для {symbol} не найден")
    
    # Здесь нужно добавить корректную остановку бота
    # Пока просто удаляем из списка
    _active_bots.pop(symbol, None)
    
    await send_alert(f"⏹️ Бот остановлен для {symbol}", "WARNING")
    
    return {"status": "stopped", "symbol": symbol}

@app.get("/api/bot/status")
async def bot_status():
    """Получение статуса всех активных ботов."""
    status = []
    for symbol, bot in _active_bots.items():
        # Здесь можно добавить реальную статистику из бота
        status.append({
            "symbol": symbol,
            "status": "running",
            "started_at": "2024-01-01T00:00:00",  # Заменить на реальное время
            "cycles_completed": 0,
            "last_signal": None
        })
    
    return {
        "active_count": len(_active_bots),
        "bots": status
    }

@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Запуск бэктеста стратегии."""
    # В реальной реализации здесь будет вызов модуля бэктеста
    # Пока возвращаем заглушку
    
    background_tasks.add_task(
        _run_background_backtest,
        request.symbol,
        request.start_date,
        request.end_date,
        request.timeframe,
        request.strategy
    )
    
    return {
        "status": "started",
        "backtest_id": "bt_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "message": "Бэктест запущен в фоне. Результаты будут отправлены в Telegram.",
        "parameters": request.dict()
    }

async def _run_background_backtest(symbol: str, start: str, end: str, tf: str, strategy: str):
    """Фоновая задача выполнения бэктеста."""
    await asyncio.sleep(2)  # Имитация работы
    
    # Здесь будет реальный вызов backtest.runner.py
    result = {
        "symbol": symbol,
        "period": f"{start} - {end}",
        "total_trades": 42,
        "win_rate": 0.67,
        "profit_factor": 1.85,
        "max_drawdown": -0.12,
        "sharpe_ratio": 1.34
    }
    
    # Отправляем результат
    message = f"""
📊 <b>Результаты бэктеста</b>

<b>Пара:</b> {symbol}
<b>Период:</b> {start} - {end}
<b>Стратегия:</b> {strategy}

<b>Итоги:</b>
• Сделок: {result['total_trades']}
• Win Rate: {result['win_rate']*100:.1f}%
• Профит-фактор: {result['profit_factor']:.2f}
• Макс. просадка: {result['max_drawdown']*100:.1f}%
• Шарп: {result['sharpe_ratio']:.2f}

<i>Бэктест завершен</i>
"""
    
    await send_alert(message, "INFO")

@app.get("/api/levels/{symbol}")
async def get_levels(
    symbol: str,
    timeframe: str = "1h",
    active_only: bool = True
):
    """Получение уровней поддержки/сопротивления для символа."""
    db = get_database()
    
    if active_only:
        levels = await db.get_active_levels(symbol, timeframe)
    else:
        # Здесь можно добавить получение всех исторических уровней
        levels = {"supports": [], "resistances": []}
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "levels": levels,
        "current_price": 0,  # Здесь нужно получить актуальную цену
        "updated_at": datetime.now().isoformat()
    }

@app.get("/api/statistics/daily")
async def daily_statistics():
    """Ежедневная статистика работы системы."""
    db = get_database()
    
    # Сигналы за сегодня
    today = datetime.now().date()
    signals_today = await db.get_recent_signals(hours=24)
    
    # Активные боты
    active_bots = list(_active_bots.keys())
    
    # Статистика сигналов
    stats = await db.get_signal_statistics(days=7)
    
    return {
        "date": today.isoformat(),
        "signals_today": len(signals_today),
        "active_bots": active_bots,
        "weekly_stats": stats,
        "system": {
            "status": "operational",
            "uptime": "24h",  # Здесь можно добавить реальный аптайм
            "memory_usage": "45%",
            "cpu_usage": "12%"
        }
    }

# ===== СОБЫТИЯ ЖИЗНЕННОГО ЦИКЛА =====

@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения."""
    print(f"🚀 Trading Bot API запущен: {datetime.now()}")
    
    # Инициализируем базу данных
    db = get_database()
    
    # Запускаем Telegram уведомитель
    try:
        notifier = await get_notifier()
        await send_alert("✅ Веб-интерфейс Trading Bot запущен", "SUCCESS")
    except Exception as e:
        print(f"⚠️ Telegram не настроен: {e}")
    
    # Здесь можно запустить демон-ботов по умолчанию
    # Например, для BTC/USDT

@app.on_event("shutdown")
async def shutdown_event():
    """Действия при остановке приложения."""
    print(f"🛑 Trading Bot API останавливается: {datetime.now()}")
    
    # Останавливаем всех активных ботов
    for symbol, bot in _active_bots.items():
        print(f"Останавливаем бота для {symbol}")
        # Здесь нужна корректная остановка бота
    
    # Отправляем уведомление
    try:
        await send_alert("🛑 Веб-интерфейс Trading Bot остановлен", "WARNING")
    except:
        pass

# ===== ЗАПУСК СЕРВЕРА =====

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Автоперезагрузка при изменениях кода
        log_level="info"
    )