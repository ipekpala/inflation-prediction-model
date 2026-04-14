import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load World Bank inflation data
data = pd.read_csv("../economic-dashboard/inflation_data.csv", skiprows=4)
turkey = data[data["Country Name"] == "Turkiye"]

years = []
values = []

for year in range(2000, 2024):
    value = pd.to_numeric(turkey[str(year)].values[0], errors="coerce")
    if pd.notna(value):
        years.append(year)
        values.append(float(value))

# Prepare training data
X = np.array(years).reshape(-1, 1)
y = np.array(values)

# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast future years
future_years = np.array(range(2024, 2030)).reshape(-1, 1)
predictions = model.predict(future_years)

# Create prediction table
forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Inflation": predictions
})

print("\nTurkey Inflation Forecast")
print(forecast_df.to_string(index=False))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(years, values, marker="o", linewidth=2, label="Real Data")
plt.plot(
    future_years.flatten(),
    predictions,
    linestyle="--",
    marker="o",
    linewidth=2,
    label="Forecast"
)

plt.xlabel("Year")
plt.ylabel("Inflation (%)")
plt.title("Turkey Inflation Forecast Model")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()