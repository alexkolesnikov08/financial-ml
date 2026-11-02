# Бэклог v1.4 (после v1.3)

## Выполнено в v1.3
- ✅ Переход на 1h (730d → ~1063 дней, 5085 баров).
- ✅ 4 hold-out вместо 2.
- ✅ Regime-анализ (trend/breakout/sideways) в pipeline.
- ✅ Accuracy/precision/recall/F1 в метриках.
- ✅ Feature selection (95 → 40).
- ✅ Отключение агрессивных фильтров по умолчанию.
- ✅ Confidence-фильтр (0.55).
- ✅ Edge-фильтр подключён.
- ✅ Short-стратегия добавлена (USE_SHORT=False).
- ✅ Cross-asset per-asset.
- ✅ Векторизация autocorr_ret_1_10.
- ✅ Багфиксы (dropna, select_features, exposure, short-accounting, min_hold/trailing).

## Осталось / новые задачи

### 🔴 Критическое
1. **Regime-анализ не сохраняется.** Сейчас вычисляется в pipeline, но main.py сохраняет только aggregate. Добавить в results.json и отчёт.
2. **6–8 hold-out на 1h.** Сейчас 4. Увеличить для надёжности (данных хватает).
3. **Оптимизация стратегических параметров.** Гистерезис, min_hold, trail_mult, cooldown — подбирать через Optuna на OOF (multi-objective).

### 🟠 Высокий
4. **1d (дневные) для проверки.** Ещё более длинная история, другой масштаб.
5. **Short-стратегия для сравнения.** Включить USE_SHORT=True, сравнить с long-only.
6. **Калибровка порогов на OOF.** Сейчас confidence=0.55 фиксированный. Подбирать на OOF.
7. **Формальные тесты (pytest).** features.build, simulate, make_time_mask, calibrate.

### 🟡 Средний
8. **Ужесточить типы** (basedpyright 149 ошибок, большинство pandas).
9. **Удалить мёртвый код** (ensure_min_history, make_calibrated, diversity_prune).
10. **Документировать удаление sma_*/dist_sma_*** как breaking change.

### 🟢 Низкий
11. **Векторизация rolling stats** (уже частично).
12. **Переключение ТФ через CLI** (сейчас через config.TIMEFRAME).
