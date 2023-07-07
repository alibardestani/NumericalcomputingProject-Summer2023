import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('Real estate.csv')

# Display the first few rows of the dataset
print(data.head())

# Create line charts to show the relationships between each feature and the target variable
features = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
            'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
target = 'Y house price of unit area'

plt.figure(figsize=(12, 8))  # Set the size of the figure

for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)  # Create subplots
    plt.plot(data[feature], data[target], 'o', alpha=0.5)  # Plot the data points
    plt.xlabel(feature)  # Set the x-axis label
    plt.ylabel(target)  # Set the y-axis label

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()  # Display the plots

# Prepare the data for modeling
X = data.drop(['No', 'X1 transaction date', 'Y house price of unit area'], axis=1)
y = data['Y house price of unit area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Calculate the root mean squared error
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Calculate the R-squared value
r_squared = model.score(X_test, y_test)
print('R-squared:', r_squared)
