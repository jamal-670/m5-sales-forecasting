import streamlit as st
import pandas as pd

st.set_page_config(page_title="M5 Forecasting Demo", page_icon="📈", layout="wide")

st.title("M5 Sales Forecasting")
st.write("Explore how different business factors affect sales predictions for the Walmart M5 forecasting challenge.")

st.markdown("### What this demo does")
st.write("- Uses a simple retail-style forecasting interface")
st.write("- Adjusts predictions for store, department, month, weekend, promotion, price, and recent sales")
st.write("- Compares a few model-style views side by side")

with st.sidebar:
    st.header("Forecast inputs")
    store_id = st.selectbox("Store ID", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
    dept_id = st.selectbox("Department ID", [1, 2, 3, 4, 5], index=1)
    month = st.slider("Month", 1, 12, 8)
    week_of_year = st.slider("Week of year", 1, 53, 20)
    weekend = st.checkbox("Weekend", value=True)
    promo = st.checkbox("Promotion active", value=False)
    price = st.number_input("Unit price", min_value=1.0, max_value=20.0, value=3.5, step=0.1)
    previous_sales = st.number_input("Previous week's sales", min_value=0, max_value=5000, value=180, step=10)
    holiday = st.checkbox("Holiday / special event", value=False)


def seasonal_factor(month_value: int) -> float:
    seasonal_map = {
        1: 0.88,
        2: 0.90,
        3: 0.95,
        4: 1.02,
        5: 1.08,
        6: 1.11,
        7: 1.14,
        8: 1.10,
        9: 1.05,
        10: 1.01,
        11: 0.98,
        12: 1.16,
    }
    return seasonal_map.get(month_value, 1.0)


def predict_sales(store: int, dept: int, month_value: int, week_value: int, weekend_flag: bool, promo_flag: bool, price_value: float, prev_sales: int, holiday_flag: bool) -> pd.DataFrame:
    base_demand = 75 + (store * 8) + (dept * 12)
    seasonality = seasonal_factor(month_value)
    weekend_adjustment = 1.12 if weekend_flag else 0.95
    promo_adjustment = 1.18 if promo_flag else 1.0
    holiday_adjustment = 1.10 if holiday_flag else 1.0
    price_adjustment = max(0.7, 1.0 - (price_value - 2.5) * 0.04)
    trend_adjustment = 1.0 + min(0.6, prev_sales / 1000)
    week_adjustment = 1.0 + (week_value % 10) * 0.01

    baseline = base_demand * seasonality * weekend_adjustment * holiday_adjustment * week_adjustment
    promo_model = baseline * promo_adjustment * price_adjustment
    trend_model = baseline * trend_adjustment * price_adjustment
    ensemble = (baseline + promo_model + trend_model) / 3

    return pd.DataFrame(
        {
            "Model": ["Baseline", "Promo-aware", "Trend-aware", "Ensemble"],
            "Predicted sales": [round(baseline, 2), round(promo_model, 2), round(trend_model, 2), round(ensemble, 2)],
        }
    )


if st.button("Predict sales", type="primary"):
    results = predict_sales(
        store_id,
        dept_id,
        month,
        week_of_year,
        weekend,
        promo,
        price,
        previous_sales,
        holiday,
    )

    top_prediction = results.loc[results["Predicted sales"].idxmax()]

    st.subheader("Prediction summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended forecast", f"{top_prediction['Predicted sales']:.2f}")
    col2.metric("Model type", top_prediction["Model"])
    col3.metric("Price effect", f"{price:.2f} USD")

    st.dataframe(results.set_index("Model"), use_container_width=True)

    st.markdown("### Why this changes")
    st.write(
        f"The forecast rises when the month is more seasonal, promotion is active, the day is a weekend, and recent sales are stronger."
    )
else:
    st.info("Adjust the factors on the left and click Predict sales to see the forecast.")
