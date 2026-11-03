# -*- coding: utf-8 -*-
"""Глобальная конфигурация пайплайна.

Все константы, пути, издержки и правила - здесь. Меняешь параметры - меняешь тут,
остальной код читает из этого модуля. Никаких магических чисел в других файлах.
"""

from pathlib import Path

# ---------------- Пути ----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Данные ----------------
TICKER = "AAPL"
# Предпочитаемый интервал. Если истории мало - fallback на 15m.
PREFERRED_INTERVAL = "5m"
FALLBACK_INTERVAL = "15m"
# Таймфрейм по умолчанию (v1.3): поддерживаемые см. TIMEFRAMES.
TIMEFRAME = "1h"
TIMEFRAMES = ["5m", "15m", "1h", "1d"]
# Период загрузки у yfinance для интрадей (макс ~60 дней для 5m/15m).
DOWNLOAD_PERIOD = "60d"
# Часовой пояс для интрадей-фичей.
TZ = "America/New_York"

# ---------------- Цель ----------------
# y_t = 1 если close_{t+1} > close_t
TARGET_COL = "y"
HORIZON = 1

# ---------------- Валидация / CV ----------------
# Количество временных фолдов (rolling walk-forward) на train+val.
N_FOLDS = 5
# Режим сплита: "rolling" (скользящее окно) или "expanding".
CV_MODE = "rolling"
# Размер валидационного блока в фолде (доля от train+val).
VAL_FRACTION = 0.2
# Минимальный размер обучающей части в фолде (доля).
MIN_TRAIN_FRACTION = 0.3

# ---------------- Multi-holdout ----------------
# Сколько untouched hold-out периодов прогнать и усреднить.
N_HOLDOUTS = 8
# Длина одного hold-out в днях (календарных).
HOLDOUT_DAYS = 14
# Анализ по режимам рынка на hold-out (см. также REGIME_ANALYSIS в секции "Стратегия").

# ---------------- Издержки бота ----------------
# Доля с оборота на одну сделку (вход+выход считается отдельно).
# Консервативно: ~0.01-0.015%.
COST_PER_SIDE = 0.000125  # 0.0125% за сторону
SLIPPAGE_PER_SIDE = 0.00005  # 0.005% за сторону
TOTAL_COST_PER_SIDE = COST_PER_SIDE + SLIPPAGE_PER_SIDE

# ---------------- Бот ----------------
START_CAPITAL = 1000.0
LONG_ONLY = True

# ---------------- Модели / тюнинг ----------------
RANDOM_STATE = 42
N_TUNING_ITER = 12  # число случайных конфигов на модель (умеренно)
# Метрика тюнинга: "neg_log_loss" | "roc_auc" | "bot_pnl"
TUNE_METRIC = "neg_log_loss"

# ---------------- Optuna ----------------
USE_OPTUNA = True
OPTUNA_N_TRIALS = 20
OPTUNA_TIMEOUT = None
OPTUNA_PRUNER = "median"
# ---------------- Калибровка ----------------
CALIBRATION_METHODS = ("sigmoid", "isotonic", "temperature", "beta")

# ---------------- Порог ----------------
# Пороги перебираем на валидационных OOF (только train+val).
THRESHOLD_GRID = [i / 1000 for i in range(400, 601)]  # 0.40..0.60

# ---------------- Стратегия (v1.3) ----------------
# Агрессивные фильтры отключены по умолчанию.
THR_HI = None
THR_LO = None
MIN_HOLD = 0
COOLDOWN = 0
TRAIL_MULT = None
USE_TREND_MASK = True
TIME_FILTERS = None
# Консенсус базовых моделей при входе (опционально).
USE_CONSENSUS = False
# Порог уверенности: покупаем только при p >= CONFIDENCE_THRESHOLD.
CONFIDENCE_THRESHOLD = 0.55
# Порог уверенности для шорта: short при p <= 1 - CONFIDENCE_THRESHOLD.
# Порог edge-фильтра (0.0 = без дополнительного порога).
EDGE_THRESHOLD = 0.0
SHORT_CONFIDENCE_THRESHOLD = 0.45
# Разрешить шорт (по умолчанию выключено).
USE_SHORT = True
# Включить regime-анализ в пайплайне.
REGIME_ANALYSIS = True
# ---------------- Воспроизводимость ----------------
N_JOBS = -1

# ---------------- Cross-asset ----------------
CROSS_ASSETS = ['SPY', 'QQQ', '^VIX']
USE_CROSS_ASSETS = True

# ---------------- Feature selection (v1.3) ----------------
FEATURE_SELECTION = True
FEATURE_TOP_K = 40
""" 
"""
