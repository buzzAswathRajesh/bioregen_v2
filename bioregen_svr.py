import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Dataset without Concentration and Additive
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

# Define features and target
X = data[["Day_10", "Day_20", "Day_30", "Day_40", "Day_50"]]
y = 100 - data["Day_50"]  # Remaining percentage to degrade to 100%

# Scale the features for SVR
scaler = StandardScaler()
print(scaler)
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVR Model with Linear Kernel
svr_model = SVR(kernel='linear', C=10.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

# Add predictions to the dataset
data["Predicted Remaining Days to Degrade"] = svr_model.predict(scaler.transform(X))

# Add 50 to the predictions
data["Predicted Days"] = data["Predicted Remaining Days to Degrade"] + 50

# Print the result
print(data[["Sample",  "Predicted Days"]])
