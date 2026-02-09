# web/app.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
import json

from web.routes.api import router as api_router
from web.routes.trading import router as trading_router
from web.routes.backtest import router as backtest_router
from storage.database import get_user_strategies

app = FastAPI(title="Trading Bot v2", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(api_router, prefix="/api/v1")
app.include_router(trading_router, prefix="/api/v1/trading")
app.include_router(backtest_router, prefix="/api/v1/backtest")

# Статические файлы
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
async def root():
    return {"message": "Trading Bot v2 API", "status": "online"}

@app.get("/api/v1/user/{user_id}/strategies")
async def get_user_strategies_endpoint(user_id: str):
    """Получение стратегий пользователя"""
    strategies = await get_user_strategies(user_id)
    return {"user_id": user_id, "strategies": strategies}