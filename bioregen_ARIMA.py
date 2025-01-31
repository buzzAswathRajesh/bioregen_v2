import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to ensure cleaner output

# Function to estimate days until 100% degradation for each series in the DataFrame
def estimate_days_for_100_percent_degradation(data, n_iterations=50, threshold=100):
    # Dictionary to hold the final days estimate for each ratio
    final_days_to_100_percent = {}

    # Iterate over each row in the DataFrame, treating each as a separate time series
    for ratio, series in data.iterrows():
        series = series.astype(float)  # Ensure data is in float format
        initial_sum = series.sum()  # Initial sum of degradation

        # List to collect the estimated days to reach 100% for each bootstrap iteration
        estimated_days_list = []

        # Perform bootstrap iterations to simulate various paths to 100% degradation
        for iteration in range(n_iterations):
            cumulative_degradation = initial_sum
            days_count = len(series) * 10  # Start counting from the observed days (assuming 10 days per interval)

            # Continue forecasting until cumulative degradation reaches the threshold (100%)
            while cumulative_degradation < threshold:
                # Resample the series with replacement to simulate uncertainty
                resampled_series = series.sample(frac=1, replace=True, random_state=iteration)
                model = ARIMA(resampled_series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)  # Forecast the next step

                forecasted_degradation = forecast.iloc[0]
                cumulative_degradation += forecasted_degradation

                days_count += 10  # Assume each forecast step represents an additional 10 days

                # If cumulative degradation exceeds the threshold, adjust the final day count
                if cumulative_degradation >= threshold:
                    excess_degradation = cumulative_degradation - threshold
                    exact_days_adjustment = 10 * (excess_degradation / forecasted_degradation)
                    days_count -= exact_days_adjustment  # Adjust days based on the excess degradation

            estimated_days_list.append(days_count)  # Collect the estimated days for this iteration

        # Calculate the mean of the estimated days from all iterations for this ratio
        mean_estimated_days = np.mean(estimated_days_list)
        final_days_to_100_percent[ratio] = mean_estimated_days  # Store the mean estimate

    return final_days_to_100_percent  # Return the final estimates for all ratios


if __name__ == "__main__":

    data = pd.DataFrame({
        'Ratio': [
            'Control', 'BioRegen Polymer + 1 g CaCO3', 'BioRegen Polymer + 2 g CaCO3',
            'BioRegen Polymer + 3 g CaCO3', 'BioRegen Polymer + 4 g CaCO3', 
            'BioRegen Polymer + 5 g CaCO3', 'BioRegen Polymer + 1 g TiO2',
            'BioRegen Polymer + 2 g TiO2', 'BioRegen Polymer + 3 g TiO2', 
            'BioRegen Polymer + 4 g TiO2', 'BioRegen Polymer + 5 g TiO2', 
            'BioRegen Polymer + 1 g ZnO', 'BioRegen Polymer + 2 g ZnO', 
            'BioRegen Polymer + 3 g ZnO', 'BioRegen Polymer + 4 g ZnO',
            'BioRegen Polymer + 5 g ZnO'
        ],
        '0-10 days': [11.09, 11.94, 9.71, 9.26, 7.72, 5.39, 11.38, 9.46, 9.64, 8.42, 6.07, 11.93, 9.40, 9.77, 8.60, 7.36],
        '10-20 days': [12.11, 9.98, 10.08, 8.06, 7.82, 8.87, 9.56, 8.92, 6.80, 8.06, 6.95, 10.46, 9.37, 6.07, 7.76, 6.63],
        '20-30 days': [22.07, 20.33, 19.69, 13.46, 9.99, 8.17, 20.36, 19.29, 14.36, 7.71, 10.40, 19.28, 20.50, 15.08, 9.38, 9.35],
        '30-40 days': [14.13, 9.46, 6.23, 20.17, 20.37, 9.28, 10.86, 9.87, 19.91, 20.94, 8.40, 10.68, 5.89, 20.68, 18.88, 8.54],
        '40-50 days': [14.47, 19.31, 19.67, 9.56, 10.69, 8.24, 18.44, 17.52, 9.13, 12.39, 13.35, 18.35, 20.40, 9.41, 12.79, 11.49]
    }).set_index('Ratio')  # Set 'Ratio' as the index for the DataFrame

    # Call the estimation function
    degradation_days = estimate_days_for_100_percent_degradation(data)
    # Print the estimated days for each ratio
    print("\n-------------------------------------------\n")
    print("Estimated Exact Days to Reach 100% Degradation")
    print("\n-------------------------------------------\n")
    for ratio, days in degradation_days.items():
        print(f"{ratio}: {days:.2f} days")
    print("\n-------------------------------------------\n")



