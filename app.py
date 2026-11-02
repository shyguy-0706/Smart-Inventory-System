from pathlib import Path
from datetime import date, timedelta
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# SmartStock — user-entered stock + multi-day festival forecasting.
# Uses the existing saved Random Forest model. Does not read old prediction CSVs
# and does not retrain the model.

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ML" / "Dataset"
MODEL_PATH = ROOT / "ML" / "models" / "random_forest_model.pkl"

st.set_page_config(page_title="SmartStock | Demand Forecast", page_icon="🪔", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f4f7fb}
[data-testid="stSidebar"]{background:#14243c}
[data-testid="stSidebar"] *{color:#f2f6fb!important}
.block-container{max-width:1250px;padding-top:1.4rem}
.hero{background:linear-gradient(115deg,#14243c,#1c5369,#267b78);padding:27px 30px;border-radius:18px;color:white;margin-bottom:22px}
.hero h1{color:white!important;font-family:Manrope;font-size:2.1rem;margin:0}
.hero p{color:#deedf2;margin:8px 0 0}
.note{background:#eaf3f8;border-left:4px solid #287d8e;padding:12px 15px;border-radius:8px}
.kpi{background:white;border:1px solid #e2e9f0;border-radius:14px;padding:17px}
.kpi small{color:#748197;font-weight:700}
.kpi h2{font-family:Manrope;color:#17314a;margin:5px 0}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_resource
def load_model(path):
    saved = joblib.load(path)
    if isinstance(saved, dict):
        for key in ("pipeline", "model", "best_model", "estimator", "random_forest"):
            if key in saved and hasattr(saved[key], "predict"):
                return saved[key]
    if hasattr(saved, "predict"):
        return saved
    raise TypeError("Saved model object has no predict() method.")

products = load_csv(DATA / "product_master.csv")
sales = load_csv(DATA / "daily_sales_history.csv")
calendar = load_csv(DATA / "festival_calendar.csv")
mapping = load_csv(DATA / "festival_product_mapping.csv")

st.markdown("""
<div class="hero">
<h1>SmartStock</h1>
<p>Your store. Your inventory. Demand predictions for the dates you choose.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🪔 SmartStock")
st.sidebar.caption("GENERAL STORE • INDIA")
st.sidebar.success("Existing Random Forest model")
st.sidebar.caption("Multi-day demand forecasting")
st.sidebar.caption("Academic prototype • simulated training data")

if products.empty or sales.empty:
    st.error("Missing ML/Dataset/product_master.csv or daily_sales_history.csv. Keep this app.py beside the ML folder.")
    st.stop()
if not MODEL_PATH.exists():
    st.error("Trained model missing: ML/models/random_forest_model.pkl")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load trained model: {e}")
    st.stop()

# These names match the existing Random Forest training feature set.
FEATURES = [
    "product_id", "category", "month", "day_of_week_num", "day_of_month",
    "is_weekend", "selling_price_inr", "discount_pct", "sales_lag_1d",
    "sales_lag_7d", "sales_lag_14d", "sales_lag_28d", "rolling_mean_7d",
    "rolling_mean_28d", "is_festival_day", "is_festival_window",
    "days_to_next_festival", "supplier_lead_time_days", "day_of_week",
    "festival_day_name", "festival_name", "upcoming_festival_name"
]

sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
if not calendar.empty and "event_date" in calendar.columns:
    calendar["event_date"] = pd.to_datetime(calendar["event_date"], errors="coerce")

catalog = products.drop_duplicates("product_id").copy()
# Hide Diyas because the current trained model consistently predicts zero for this item.
catalog = catalog[
    ~catalog["product_name"].astype(str).str.strip().str.casefold().eq("diyas (set of 4)".casefold())
].copy()
catalog["product_id"] = catalog["product_id"].astype(str)
catalog = catalog.set_index("product_id", drop=False)

# The user chooses only a forecast start and end date.
st.header("1. Choose your forecast period")
date_col1, date_col2 = st.columns(2)
with date_col1:
    forecast_start = st.date_input("Forecast starts", value=date.today() + timedelta(days=1))
with date_col2:
    forecast_end = st.date_input("Forecast ends", value=date.today() + timedelta(days=7))

if forecast_end < forecast_start:
    st.error("Forecast end must be on or after forecast start.")
    st.stop()

st.markdown(
    f'<div class="note">The model will predict demand for each day from '
    f'<b>{forecast_start:%d %b %Y}</b> to <b>{forecast_end:%d %b %Y}</b>, '
    'then add the daily predictions to estimate total demand for this period.</div>',
    unsafe_allow_html=True
)

st.header("2. Enter your store's stock and sales")

st.write("Tick the products you sell. Enter your real current stock and recent sales. The initial sales figures are examples loaded from the synthetic dataset; edit them to match your shop.")

rows = []
for _, p in catalog.iterrows():
    pid = str(p["product_id"])
    hist = sales[sales["product_id"].astype(str) == pid].sort_values("date")
    last = hist.iloc[-1] if not hist.empty else pd.Series(dtype=object)

    def hv(col, fallback=0.0):
        v = last.get(col, fallback)
        return float(fallback if pd.isna(v) else v)

    lag1 = hv("sales_lag_1d", hv("quantity_sold", 0))
    lag7 = hv("sales_lag_7d", lag1)
    lag14 = hv("sales_lag_14d", lag7)
    lag28 = hv("sales_lag_28d", lag14)
    avg7 = hv("rolling_mean_7d", lag7)
    avg28 = hv("rolling_mean_28d", lag28)
    rows.append({
        "Use": False,
        "Product ID": pid,
        "Product": str(p["product_name"]),
        "Current stock": 0,
        "Selling price ₹": float(p.get("selling_price_inr", 0) or 0),
        "Discount %": 0.0,
        "Lead time days": int(p.get("supplier_lead_time_days", 3) or 3),
        "Sales 1d ago": lag1,
        "Sales 7d ago": lag7,
        "Sales 14d ago": lag14,
        "Sales 28d ago": lag28,
        "7d average": avg7,
        "28d average": avg28
    })

with st.form("festival_forecast_form"):
    edited = st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["Product ID", "Product"],
        column_config={
            "Use": st.column_config.CheckboxColumn("Use", help="Tick to include this product."),
            "Current stock": st.column_config.NumberColumn(min_value=0, step=1),
            "Selling price ₹": st.column_config.NumberColumn(min_value=0.0, step=1.0),
            "Discount %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
            "Lead time days": st.column_config.NumberColumn(min_value=0, max_value=90, step=1),
            "Sales 1d ago": st.column_config.NumberColumn(min_value=0.0, step=1.0),
            "Sales 7d ago": st.column_config.NumberColumn(min_value=0.0, step=1.0),
            "Sales 14d ago": st.column_config.NumberColumn(min_value=0.0, step=1.0),
            "Sales 28d ago": st.column_config.NumberColumn(min_value=0.0, step=1.0),
            "7d average": st.column_config.NumberColumn(min_value=0.0, step=0.5),
            "28d average": st.column_config.NumberColumn(min_value=0.0, step=0.5)
        },
        key="festival_user_inventory"
    )
    go = st.form_submit_button("Forecast demand for the full period", type="primary", use_container_width=True)

def get_calendar_context(day):
    """Use saved calendar dates automatically; user does not select an event."""
    if calendar.empty or not {"event_date", "event_name"}.issubset(calendar.columns):
        return 0, 0, 365, "None", "None", "None"
    events = calendar.dropna(subset=["event_date"]).copy()
    events["delta"] = (events["event_date"] - day).dt.days
    future = events[events["delta"] >= 0].sort_values("event_date")
    upcoming_name, days_to_next = ("None", 365)
    if not future.empty:
        upcoming_name = str(future.iloc[0]["event_name"])
        days_to_next = int(future.iloc[0]["delta"])
    same_day = events[events["event_date"] == day]
    if not same_day.empty:
        name = str(same_day.iloc[0]["event_name"])
        return 1, 1, 0, name, name, name
    nearby = events[(events["delta"] >= 0) & (events["delta"] <= 7)]
    if not nearby.empty:
        name = str(nearby.sort_values("event_date").iloc[0]["event_name"])
        return 0, 1, days_to_next, name, "None", name
    return 0, 0, days_to_next, "None", "None", upcoming_name

if go:
    chosen = edited[edited["Use"] == True].copy()
    if chosen.empty:
        st.warning("Tick at least one product in the Use column.")
    elif forecast_end < forecast_start:
        st.error("Please correct the forecast dates.")
    else:
        days = pd.date_range(forecast_start, forecast_end, freq="D")
        daily_rows = []
        metadata = []
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        for _, r in chosen.iterrows():
            pid = str(r["Product ID"])
            product = catalog.loc[pid]
            for day in days:
                dow = day.dayofweek
                is_day, is_window, days_to, f_name, f_day_name, upcoming = get_calendar_context(day)
                daily_rows.append({
                    "product_id": pid,
                    "category": str(product["category"]),
                    "month": day.month,
                    "day_of_week_num": dow,
                    "day_of_month": day.day,
                    "is_weekend": int(dow >= 5),
                    "selling_price_inr": float(r["Selling price ₹"]),
                    "discount_pct": float(r["Discount %"]),
                    "sales_lag_1d": float(r["Sales 1d ago"]),
                    "sales_lag_7d": float(r["Sales 7d ago"]),
                    "sales_lag_14d": float(r["Sales 14d ago"]),
                    "sales_lag_28d": float(r["Sales 28d ago"]),
                    "rolling_mean_7d": float(r["7d average"]),
                    "rolling_mean_28d": float(r["28d average"]),
                    "is_festival_day": int(is_day),
                    "is_festival_window": int(is_window),
                    "days_to_next_festival": int(days_to),
                    "supplier_lead_time_days": int(r["Lead time days"]),
                    "day_of_week": weekday_names[dow],
                    "festival_day_name": f_day_name,
                    "festival_name": f_name,
                    "upcoming_festival_name": upcoming
                })
                metadata.append({
                    "Product ID": pid,
                    "Product": str(r["Product"]),
                    "Date": day.date().isoformat(),
                    "Current stock": int(r["Current stock"]),
                    "Lead time days": int(r["Lead time days"])
                })

        X = pd.DataFrame(daily_rows, columns=FEATURES)
        try:
            # Fresh model inference for every selected product and every day.
            preds = np.asarray(model.predict(X)).reshape(-1)
            daily = pd.DataFrame(metadata)
            daily["Predicted demand"] = np.maximum(0, np.rint(preds)).astype(int)

            summary = daily.groupby(["Product ID", "Product"], as_index=False).agg(
                **{
                    "Current stock": ("Current stock", "first"),
                    "Lead time days": ("Lead time days", "first"),
                    "Total forecast demand": ("Predicted demand", "sum")
                }
            )
            summary["Stock shortfall"] = np.maximum(0, summary["Total forecast demand"] - summary["Current stock"])
            summary["Suggested order quantity"] = summary["Stock shortfall"]
            summary["Order-by date"] = [
                (pd.Timestamp(forecast_start) - pd.Timedelta(days=int(x))).date().isoformat()
                for x in summary["Lead time days"]
            ]
            summary["Order status"] = np.where(
                summary["Stock shortfall"] > 0, "ORDER REQUIRED", "CURRENT STOCK COVERS FORECAST"
            )

            st.divider()
            st.header("3. Your demand forecast")
            st.caption(f"Daily model predictions from {forecast_start:%d %b %Y} to {forecast_end:%d %b %Y}. The total is the sum of those daily predictions.")
            a, b, c = st.columns(3)
            with a:
                st.markdown(f'<div class="kpi"><small>PRODUCTS INCLUDED</small><h2>{len(summary)}</h2></div>', unsafe_allow_html=True)
            with b:
                st.markdown(f'<div class="kpi"><small>TOTAL FORECAST DEMAND</small><h2>{int(summary["Total forecast demand"].sum())} units</h2></div>', unsafe_allow_html=True)
            with c:
                n_order = int((summary["Stock shortfall"] > 0).sum())
                st.markdown(f'<div class="kpi"><small>PRODUCTS NEEDING ORDER</small><h2>{n_order}</h2></div>', unsafe_allow_html=True)

            st.subheader("Inventory and order recommendations")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.subheader("Daily demand breakdown")
            st.dataframe(daily[["Product", "Date", "Predicted demand"]], use_container_width=True, hide_index=True)

            # Product selection chart helps the owner inspect a seasonal product such as diya sets.
            chart_product = st.selectbox("View daily forecast for a product", summary["Product"].tolist())
            chart = daily[daily["Product"] == chart_product].set_index("Date")[["Predicted demand"]]
            st.line_chart(chart)

            colx, coly = st.columns(2)
            with colx:
                st.download_button(
                    "Download forecast summary (CSV)",
                    summary.to_csv(index=False).encode("utf-8"),
                    f"smartstock_forecast_summary_{forecast_start.isoformat()}_{forecast_end.isoformat()}.csv",
                    "text/csv"
                )
            with coly:
                st.download_button(
                    "Download daily predictions (CSV)",
                    daily.to_csv(index=False).encode("utf-8"),
                    f"smartstock_daily_forecast_{forecast_start.isoformat()}_{forecast_end.isoformat()}.csv",
                    "text/csv"
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("If the model expects a different feature schema, share the feature/preprocessing and joblib save section from ML/train.py so the app can use the exact saved pipeline.")

st.divider()
st.caption("SmartStock academic prototype • Existing Random Forest model • User-entered inventory • Multi-day demand forecast")
