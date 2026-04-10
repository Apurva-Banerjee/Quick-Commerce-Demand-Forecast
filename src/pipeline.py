"""End-to-end pipeline for quick-commerce demand forecasting."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, END_DATE, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, START_DATE
from .data_simulation import generate_synthetic_data
from .evaluation import plot_actual_vs_predicted, regression_metrics, run_eda_plots
from .features import finalize_features
from .models_lstm import LSTMArtifacts, create_lstm_sequences, save_lstm_model, scale_lstm_features, train_lstm
from .models_xgb import XGBArtifacts, prepare_xgb_features, save_xgb_artifacts, train_xgb
from .preprocessing import preprocess_data, time_split


def ensure_dirs() -> None:
    """Ensure output directories exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _example_forecast_line(df: pd.DataFrame, pred_col: str) -> str:
    """Return one human-readable demand forecast example."""
    row = df.sort_values("datetime").iloc[-1]
    hour_str = pd.to_datetime(row["datetime"]).strftime("%I %p")
    return f"Expected demand of {row['product_id']} at {hour_str} = {int(round(row[pred_col]))} units"


def run_pipeline() -> dict:
    """Run complete forecasting workflow and save artifacts."""
    ensure_dirs()

    # 1) Data loading (simulated; can be swapped with Kaggle input).
    raw = generate_synthetic_data(START_DATE, END_DATE)
    raw.to_csv(DATA_DIR / "retail_demand_simulated.csv", index=False)

    # 2) Preprocessing + feature engineering.
    clean = preprocess_data(raw)
    feats = finalize_features(clean)

    # 3) EDA plots.
    run_eda_plots(feats, PLOTS_DIR)

    # 4) Time-based split.
    train_df, test_df = time_split(feats, train_ratio=0.8)

    # =========================
    # XGBoost branch
    # =========================
    xgb_train, xgb_test, xgb_feature_cols, category_map = prepare_xgb_features(train_df, test_df)
    xgb_model = train_xgb(xgb_train, xgb_feature_cols)
    xgb_pred = xgb_model.predict(xgb_test[xgb_feature_cols])
    xgb_metrics = regression_metrics(xgb_test["sales"].to_numpy(), xgb_pred)

    xgb_artifacts = XGBArtifacts(model=xgb_model, feature_cols=xgb_feature_cols, category_map=category_map)
    save_xgb_artifacts(xgb_artifacts, MODELS_DIR / "xgb_model.pkl")

    xgb_pred_df = xgb_test[["product_id", "datetime", "sales"]].copy()
    xgb_pred_df["xgb_pred"] = xgb_pred
    plot_actual_vs_predicted(
        xgb_pred_df["datetime"],
        xgb_pred_df["sales"].to_numpy(),
        xgb_pred_df["xgb_pred"].to_numpy(),
        PLOTS_DIR / "xgb_actual_vs_pred.png",
        "XGBoost: Actual vs Predicted",
    )

    # =========================
    # LSTM branch (optional if TensorFlow exists)
    # =========================
    lstm_pred_df = xgb_pred_df[["product_id", "datetime", "sales"]].copy()
    lstm_pred_df["lstm_pred"] = pd.NA
    lstm_metrics = {"rmse": None, "mae": None, "mape": None, "note": "TensorFlow not available; LSTM skipped."}

    try:
        lstm_feature_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "weekend",
            "month",
            "festival_flag",
            "temperature",
            "rainfall",
            "lag_1",
            "lag_24",
            "lag_168",
            "rolling_mean_24",
            "rolling_std_24",
            "rolling_mean_168",
            "demand_spike_flag",
        ]
        lstm_train_scaled, lstm_test_scaled, lstm_scaler = scale_lstm_features(train_df, test_df, lstm_feature_cols)
        seq_len = 24

        X_train_all, y_train_all, _ = create_lstm_sequences(lstm_train_scaled, lstm_feature_cols, seq_len=seq_len)
        X_test, y_test, test_meta = create_lstm_sequences(lstm_test_scaled, lstm_feature_cols, seq_len=seq_len)

        X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.15, random_state=42)

        lstm_model = train_lstm(X_train, y_train, X_val, y_val, epochs=8, batch_size=64)
        lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
        lstm_metrics = regression_metrics(y_test, lstm_pred)

        save_lstm_model(lstm_model, MODELS_DIR / "lstm_model.h5")
        joblib.dump(
            LSTMArtifacts(scaler=lstm_scaler, seq_len=seq_len, feature_cols=lstm_feature_cols),
            MODELS_DIR / "lstm_meta.pkl",
        )

        lstm_pred_df = test_meta.copy()
        lstm_pred_df["sales"] = y_test
        lstm_pred_df["lstm_pred"] = lstm_pred
        plot_actual_vs_predicted(
            lstm_pred_df["datetime"],
            lstm_pred_df["sales"].to_numpy(),
            lstm_pred_df["lstm_pred"].to_numpy(),
            PLOTS_DIR / "lstm_actual_vs_pred.png",
            "LSTM: Actual vs Predicted",
        )
    except ModuleNotFoundError:
        pass

    # 5) Combined prediction file (aligned on keys).
    final = xgb_pred_df.merge(lstm_pred_df, on=["product_id", "datetime"], how="left")
    final.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    # 6) Metrics + business-like text output.
    metrics = {"xgboost": xgb_metrics, "lstm": lstm_metrics}
    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUTS_DIR / "sample_forecasts.txt", "w", encoding="utf-8") as f:
        for sku in final["product_id"].dropna().unique():
            sku_rows = final[final["product_id"] == sku]
            if not sku_rows.empty:
                f.write(_example_forecast_line(sku_rows, "xgb_pred") + "\n")

    return metrics
