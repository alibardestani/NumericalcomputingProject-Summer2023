import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Real estate.csv')

# Scatter plot of house age vs house price
plt.scatter(data['X2 house age'], data['Y house price of unit area'])
plt.xlabel('House Age')
plt.ylabel('House Price')
plt.title('House Age vs House Price')
plt.show()
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
