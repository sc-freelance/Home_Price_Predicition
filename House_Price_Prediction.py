# Import Necessary Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Dataset
data = {
    "HouseSize": [
        793, 2477, 1263, 1291, 603, 1655, 1071, 1877, 1610, 3058,
        1609, 1630, 1211, 1219, 1700, 1629, 740, 740, 1637, 1637,
        1291, 1211, 400, 800, 700, 1711, 847, 1230, 1291, 2300,
        886, 1700, 389, 793, 978, 582, 793, 1327, 791, 793,
        1630, 740, 1244, 1183, 1327, 1183, 1643, 2500, 1085, 740,
        1291, 1291, 740, 1230, 1219, 1275, 793, 1259, 1395
    ],
    "HousePrice": [
        1300000, 3700000, 1480000, 2380000, 955000, 2130000, 1300000, 2700000, 2650000, 2850000,
        2250000, 2950000, 1900000, 1820000, 2100000, 2480000, 1270000, 1260000, 2520000, 3450000,
        2430000, 2250000, 1, 925000, 950000, 2630000, 975000, 1570000, 2010000, 3630000,
        1120000, 2020000, 695974, 1330000, 1460000, 685000, 1340000, 2440000, 925000, 1400000,
        2980000, 1200000, 1650000, 1880000, 2440000, 1880000, 2790000, 3470000, 1160000, 1230000,
        2180000, 2060000, 1270000, 1660000, 1870000, 1800000, 1330000, 1650000, 1950000
    ]
}

#Create dataframe
df = pd.DataFrame(data)

# Features and target
X = df[['HouseSize']]  # Keep as DataFrame for sklearn
y = df['HousePrice']

# Visualize data
plt.scatter(X, y, color='blue', edgecolor='k')
plt.title('House Prices vs. House Size')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation and training
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
predictions = model.predict(x_test)

# Visualize predictions
plt.scatter(x_test, y_test, color='blue', edgecolor='k', label='Actual Prices')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted Line')
plt.title('House Price Prediction (Linear Regression)')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()

# Model details
print(f"Model slope (coefficient): {model.coef_[0]:,.2f}")
print(f"Model intercept: {model.intercept_:,.2f}")
