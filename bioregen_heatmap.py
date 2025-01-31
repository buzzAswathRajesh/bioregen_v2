# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define the index (samples) and attributes (measurements) for the DataFrame
index = [
    "Control", "CaCO3_1g", "TiO2_1g", "ZnO_1g",
    "CaCO3_2g", "TiO2_2g", "ZnO_2g",
    "CaCO3_3g", "TiO2_3g", "ZnO_3g",
    "CaCO3_4g", "TiO2_4g", "ZnO_4g",
    "CaCO3_5g", "TiO2_5g", "ZnO_5g"
]
attributes = ["Tensile Strength", "Flexibility", "Water Resistance", "Phytotoxicity", "Biodegradability"]

# Data extracted from the uploaded file
data = {
    "Tensile Strength": [
        26, 29.32, 28.18, 26.12, 
        30.58, 29.27, 31.57, 
        32.44, 33.13, 29.88, 
        35.35, 36.06, 33.21, 
        38.59, 38, 39.13
    ],
    "Flexibility": [
        3.6, 3.8, 3.4, 3.8, 
        3.6, 3.1, 3.3, 
        3.4, 3.2, 3.3, 
        3.2, 2.6, 3.2, 
        3.8, 2.6, 2.9
    ],
    "Water Resistance": [
        82, 91, 91, 91, 
        93, 92, 92, 
        92, 93, 93, 
        94, 95, 94, 
        97, 96, 95
    ],
    "Phytotoxicity": [
        19.3, 19.8, 19.5, 19.8, 
        19.2, 19.5, 19.5, 
        19.5, 19.3, 19.3, 
        19.8, 19.2, 19.2, 
        19.5, 19.2, 19.2
    ],
    "Biodegradability": [
        73.87, 71.02, 70.6, 70.7, 
        65.38, 65.05, 65.56, 
        60.51, 59.85, 61.02, 
        56.6, 57.51, 57.14, 
        39.95, 45.17, 43.37
    ]
}

# Creating a DataFrame from the dictionary, setting the index to the sample names
df = pd.DataFrame(data, index=index)

# Standardizing the data by subtracting the mean and dividing by the standard deviation for each attribute
data_standardized = (df - df.mean()) / df.std()

# Define a custom colormap using the given colors
colors = ["#8fc03f", "#76b215", "#1fc2ba", "#0a958e", "#3a80b6", "#125c96"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Begin plotting with adjusted figure size
plt.figure(figsize=(20, 10))  # Adjusted figure size for better label fit

# Create the heatmap using standardized data, specifying the custom colormap and aspect ratio
heatmap = plt.imshow(data_standardized, cmap=custom_cmap, aspect='auto')

# Set the background color of the figure to white
plt.gcf().set_facecolor('white')

# Add a color bar to the figure to indicate the scale of standardized values, with customization
cbar = plt.colorbar(heatmap, label='Standardized Values', orientation='horizontal', pad=0.15)

# Set the x-axis and y-axis ticks to the column and row names respectively, making labels bold
plt.xticks(np.arange(data_standardized.shape[1]), df.columns, rotation=0, fontsize=14, fontweight='bold')
plt.yticks(np.arange(data_standardized.shape[0]), df.index, fontsize=14, fontweight='bold')

# Loop through the data to add text annotations in each cell
for i in range(data_standardized.shape[0]):  # Iterate over rows (samples)
    for j in range(data_standardized.shape[1]):  # Iterate over columns (attributes)
        # Choose text color for better visibility based on cell's background color
        text_color = 'white' if abs(data_standardized.iloc[i, j]) > 0.5 else 'black'
        # Add the text annotation, showing the original value with two decimal places
        plt.text(j, i, f'{df.iloc[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=12)

# Add a title to the heatmap, with customization for font size and weight
plt.title('BIOREGEN POLYMER HEATMAP ANALYSIS', fontsize=18, fontweight='bold', pad=30)

# Adjust the layout to ensure all elements are clearly visible without overlap
plt.tight_layout()

# Save the figure before showing it
plt.savefig('bioregen_polymer_heatmap_updated.jpeg', dpi=95, format='jpeg')

# Display the plot
plt.show()

