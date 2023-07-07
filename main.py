import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Real estate.csv')

# Scatter plot of house age vs house price
plt.scatter(data['X2 house age'], data['Y house price of unit area'])
plt.xlabel('House Age')
plt.ylabel('House Price')
plt.title('House Age vs House Price')
plt.show()
