import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Turkey Inflation Forecast", layout="wide")

st.title("Turkey Inflation Forecast Model 🇹🇷")
st.markdown("### Machine Learning Based Inflation Projection")
st.write("This app uses a simple linear regression model to forecast Turkey's inflation trend based on historical World Bank data.")

forecast_horizon = st.slider(
    "Select forecast horizon (years)",
    min_value=1,
    max_value=10,
    value=5
)

@st.cache_data
def load_inflation_data():
    data = pd.read_csv("../economic-dashboard/inflation_data.csv", skiprows=4)
    turkey = data[data["Country Name"] == "Turkiye"]

    years = []
    values = []

    for year in range(2000, 2024):
        value = pd.to_numeric(turkey[str(year)].values[0], errors="coerce")
        if pd.notna(value):
            years.append(year)
            values.append(float(value))

    return years, values

years, values = load_inflation_data()

X = np.array(years).reshape(-1, 1)
y = np.array(values)

model = LinearRegression()
model.fit(X, y)

future_years = np.array(range(2024, 2024 + forecast_horizon)).reshape(-1, 1)
predictions = model.predict(future_years)

forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Inflation": predictions
})

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Years", f"{min(years)}-{max(years)}")

with col2:
    st.metric("Latest Actual Inflation", f"{values[-1]:.2f}%")

with col3:
    st.metric("Forecast Horizon", f"{forecast_horizon} years")

st.subheader("Forecast Table")
st.dataframe(forecast_df, width="stretch")

st.subheader("Inflation Forecast Chart")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    years,
    values,
    marker="o",
    linewidth=2,
    label="Actual Data"
)

ax.plot(
    future_years.flatten(),
    predictions,
    linestyle="--",
    marker="o",
    linewidth=2,
    label="Forecast"
)

ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlabel("Year")
ax.set_ylabel("Inflation (%)")
ax.set_title("Turkey Inflation Forecast")
ax.legend()

st.pyplot(fig)

st.caption("Developed by İpek Pala")