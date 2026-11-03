# Financial ML

Classic + Deep Learning pipeline for AAPL next-bar direction prediction.

## Quick start (Colab / GPU)

```bash
# 1. Clone
git clone https://github.com/alexkolesnikov08/financial-ml.git
cd financial-ml

# 2. Install deps
pip install -r requirements.txt

# 3. Collect data
python collect_data.py --cross

# 4. Run pipeline
python main.py
```

## Structure

- `config.py` — all settings (timeframe, hold-outs, costs, model params)
- `data.py` / `collect_data.py` — data loading & collection
- `features.py` — feature engineering
- `models.py` — model zoo (classic ML + MLP variants)
- `pipeline.py` — walk-forward CV, OOF, thresholds, hold-out simulation
- `main.py` — entry point
- `backtest.py` — bot simulator (long/short, costs, trailing)
- `tune.py` — Optuna tuning
- `eval.py` / `viz.py` — metrics & plots

## Data

Data is NOT in git (see .gitignore). Run `python collect_data.py` to download from yfinance.

## GPU

MLP models use MPS/CUDA automatically. Set `OMP_NUM_THREADS=1` on macOS to avoid segfaults.

## Models

Classic: logreg, ridge, sgd, histgb, rf, et, lgbm, xgb, catboost, gaussian_nb, bernoulli_nb, lda, ada_boost, knn, dummy.

MLP variants (PyTorch): relu/gelu/silu/tanh/leaky_relu/elu, depth 1-4, width 32-256, dropout 0-0.5, batch norm.

## License

Apache-2.0
