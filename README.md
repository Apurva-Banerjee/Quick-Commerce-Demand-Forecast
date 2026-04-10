# Demand Forecasting for Quick Commerce

Portfolio-ready end-to-end Machine Learning project for hourly demand forecasting in a Blinkit/Zepto/Swiggy-like quick-commerce setting.

## Problem Statement

Predict hourly demand per SKU (for example: milk, bread, fruits) for a 10-30 minute delivery platform.

## Project Structure

```text
data science with cursor/
├── data/                     # Raw/simulated dataset
├── models/                   # Trained model artifacts (.pkl / .h5)
├── notebooks/                # Optional notebooks
├── outputs/
│   ├── plots/                # EDA and actual-vs-predicted charts
│   ├── metrics.json          # RMSE, MAE, MAPE per model
│   ├── predictions.csv       # SKU-hour level predictions
│   └── sample_forecasts.txt  # Business-readable forecast lines
├── src/
│   ├── config.py
│   ├── data_simulation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models_xgb.py
│   ├── models_lstm.py
│   ├── evaluation.py
│   └── pipeline.py
├── run_pipeline.py
├── streamlit_app.py
└── requirements.txt
```

## Features Implemented

- Simulated realistic multi-SKU hourly demand data.
- Time-based features:
  - hour, day_of_week, weekend, month, festival_flag
  - weather features: temperature, rainfall (simulated)
- Time-series feature engineering:
  - lag features: t-1, t-24, t-168
  - rolling stats: mean/std
  - cyclic encoding: sin/cos(hour), sin/cos(day_of_week)
  - demand spike indicator
- Models:
  - XGBoost regressor
  - LSTM (TensorFlow/Keras)
- Evaluation:
  - RMSE, MAE, MAPE
  - Actual vs Predicted plots
- Optional dashboard:
  - Streamlit product/hour selection + trend graph

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run End-to-End Pipeline

```bash
python run_pipeline.py
```

This command will:
- Generate dataset at `data/retail_demand_simulated.csv`
- Train and save models at:
  - `models/xgb_model.pkl`
  - `models/lstm_model.h5`
  - `models/lstm_meta.pkl`
- Save outputs:
  - `outputs/predictions.csv`
  - `outputs/metrics.json`
  - `outputs/sample_forecasts.txt`
  - `outputs/plots/*.png`

## Run Dashboard

```bash
streamlit run streamlit_app.py
```

## Notes

- Uses time-based split to prevent leakage.
- Built to be reusable and production-style modular.
- To use Kaggle data, replace data generation in `src/pipeline.py` with CSV loading while keeping the same schema:
  - `product_id`, `datetime`, `sales`
