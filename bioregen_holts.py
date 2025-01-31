# Import necessary libraries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings


warnings.filterwarnings('ignore')

# Define the input data directly
data = pd.DataFrame({
    "Sample": ["Control", "BioRegen + 1g CaCO3", "BioRegen + 2g CaCO3", "BioRegen + 3g CaCO3", "BioRegen + 4g CaCO3", "BioRegen + 5g CaCO3",
               "BioRegen + 1g TiO2", "BioRegen + 2g TiO2", "BioRegen + 3g TiO2", "BioRegen + 4g TiO2", "BioRegen + 5g TiO2",
               "BioRegen + 1g ZnO", "BioRegen + 2g ZnO", "BioRegen + 3g ZnO", "BioRegen + 4g ZnO", "BioRegen + 5g ZnO"],
    "Day_10": [11.56, 11.11, 9.87, 8.44, 7.89, 5.67, 10.99, 8.87, 8.89, 7.59, 5.56, 11.34, 9.56, 8.76, 7.77, 6.67],
    "Day_20": [21.88, 20.91, 18.78, 16.67, 15.71, 13.43, 21.22, 17.55, 15.67, 15.72, 12.43, 21.56, 17.76, 15.34, 15.77, 12.98],
    "Day_30": [45.66, 41.67, 38.65, 30.95, 25.43, 21.77, 40.55, 37.22, 29.98, 24.35, 22.78, 40.98, 38.77, 31.09, 24.91, 22.33],
    "Day_40": [57.45, 50.88, 45.87, 51.12, 45.32, 31.87, 51.41, 46.78, 50.88, 44.37, 30.99, 51.34, 45.54, 50.78, 44.78, 30.87],
    "Day_50": [74.15, 70.01, 64.55, 60.01, 56.76, 40.11, 70.01, 64.55, 60.01, 56.76, 44.34, 70.01, 64.55, 60.01, 56.76, 42.36]
})

# Function to estimate days to reach 100% degradation using Holt's linear trend model
def estimate_days_to_100_holt(data):
    estimates_holt = {}
    for index, row in data.iterrows():
        ratio = row['Sample']
        series = np.array(row[1:].astype(float))
        days = 50  # Starting from 50 days, as the last observed interval
        
        # Applying Holt's linear trend model
        model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
        model_fit = model.fit()
        
        # Forecast until reaching 100%
        total_degradation = series[-1]
        while total_degradation < 100:
            forecast = model_fit.forecast(1)[0]
            total_degradation = forecast  # Assuming the forecast is the latest degradation percentage
            days += 10  # Assuming each interval represents an additional 10 days
            # Update the series with the new forecast for re-fitting in the next iteration if necessary
            series = np.append(series, forecast)
            model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
            model_fit = model.fit()
        
        estimates_holt[ratio] = days
    
    return estimates_holt

if __name__ == "__main__":
    estimates_holt = estimate_days_to_100_holt(data)
    print("\n-------------------------------------------\n")
    print("Holt's Linear Trend Model Prediction")
    print("\n-------------------------------------------\n")
    for ratio, days in estimates_holt.items():
        print(f"{ratio}: {days} days")
    print("\n-------------------------------------------\n")


