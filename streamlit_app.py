"""Streamlit dashboard for quick-commerce demand forecasting outputs."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
PRED_PATH = PROJECT_ROOT / "outputs" / "predictions.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"

st.set_page_config(page_title="Quick Commerce Demand Forecast", layout="wide")
st.title("Quick Commerce Demand Forecast Dashboard")

if not PRED_PATH.exists():
    st.warning("Run `python run_pipeline.py` first to generate predictions.")
    st.stop()

df = pd.read_csv(PRED_PATH, parse_dates=["datetime"])

# Support both `sales` and merged variants like `sales_x`/`sales_y`.
if "sales" not in df.columns:
    if "sales_x" in df.columns:
        df["sales"] = df["sales_x"]
    elif "sales_y" in df.columns:
        df["sales"] = df["sales_y"]

# Ensure prediction columns exist even when a model is skipped.
if "lstm_pred" not in df.columns:
    df["lstm_pred"] = pd.NA
if "xgb_pred" not in df.columns:
    st.error("`xgb_pred` column is missing in predictions output. Run `python run_pipeline.py` again.")
    st.stop()

products = sorted(df["product_id"].dropna().unique())

col1, col2 = st.columns(2)
selected_product = col1.selectbox("Select Product (SKU)", options=products)
selected_hour = col2.selectbox(
    "Select Hour",
    options=sorted(df["datetime"].dt.hour.unique()),
    index=20 if 20 in df["datetime"].dt.hour.unique() else 0,
)

filtered = df[(df["product_id"] == selected_product) & (df["datetime"].dt.hour == selected_hour)]
filtered = filtered.sort_values("datetime")

if not filtered.empty:
    latest = filtered.iloc[-1]
    st.metric(
        label=f"Predicted Demand for {selected_product} at hour {selected_hour:02d}:00",
        value=f"{int(round(latest['xgb_pred']))} units",
    )
    st.caption(
        f"Expected demand of {selected_product} at {latest['datetime'].strftime('%I %p')} = "
        f"{int(round(latest['xgb_pred']))} units"
    )

st.subheader("Demand Trend (Actual vs Predicted)")
chart_df = df[df["product_id"] == selected_product].sort_values("datetime").set_index("datetime")
plot_cols = [c for c in ["sales", "xgb_pred", "lstm_pred"] if c in chart_df.columns]
if not plot_cols:
    st.warning("No plottable demand columns found in prediction output.")
else:
    st.line_chart(chart_df[plot_cols].ffill())

if METRICS_PATH.exists():
    st.subheader("Model Metrics")
    metrics = pd.read_json(METRICS_PATH)
    st.dataframe(metrics.T)
