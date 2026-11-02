# trading_ml_classic

Классический ML-пайплайн для предсказания направления следующего бара AAPL (5m, fallback 15m)
и симуляции long-only бота со стартовым капиталом $1000.

## Структура

- `config.py` — константы, пути, издержки, параметры CV.
- `data.py` — загрузка yfinance, чистка, кэш parquet.
- `features.py` — признаки строго из прошлого + цель `y_t = 1 if close_{t+1} > close_t`.
- `split.py` — временные фолды (rolling/expanding) и multi-holdout.
- `models.py` — реестр базовых моделей и мета-моделей.
- `tune.py` — тюнинг гиперпараметров по walk-forward.
- `calibrate.py` — калибровка вероятностей на отдельном временном куске.
- `cv_oof.py` — генерация OOF-предсказаний.
- `ensemble.py` — голосование (soft/weighted/geom/rank/hard) и стекинг.
- `backtest.py` — long-only бот-симулятор (signal_t -> enter_t+1, all-in, издержки).
- `eval.py` — метрики и агрегация по hold-out.
- `pipeline.py` — оркестрация + multi-holdout усреднение.
- `main.py` — точка входа.

## Запуск

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python main.py
```

## Принципы

- Никакого shuffle, только временные сплиты.
- OOF строго out-of-fold, одинаковые фолды для всех моделей.
- Порог подбирается только на валидационных OOF, на hold-out не меняется.
- Бот long-only: если прогноз "вверх" — покупаем на следующем баре, all-in, с издержками.
- Итог усредняется по нескольким untouched hold-out периодам.