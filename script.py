import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Load the data
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Task 1: Check out the data
print(df.head())

# Task 2: Group by year and calculate mean total production
prod_per_year = df.groupby('year')['totalprod'].mean().reset_index()

# Task 3: Create X (years)
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Task 4: Create y (total production)
y = prod_per_year['totalprod']

# Task 5: Create scatter plot
plt.scatter(X, y)
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.title('Honey Production Over Time')
plt.show()

# Task 6: Create linear regression model
regr = linear_model.LinearRegression()

# Task 7: Fit the model to the data
regr.fit(X, y)

# Task 8: Print slope and intercept
print(f"Slope: {regr.coef_[0]}")
print(f"Intercept: {regr.intercept_}")

# Task 9: Make predictions
y_predict = regr.predict(X)

# Task 10: Plot the regression line
plt.scatter(X, y)
plt.plot(X, y_predict, color='red', linewidth=3)
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.title('Honey Production Over Time with Regression Line')
plt.show()

# Task 11: Creating X_future
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)


#final Step:
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_future, future_predict, color='red', label='Prediction')
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.title('Honey Production Prediction up to 2050')
plt.legend()
plt.grid(True)

# Highlight the prediction for 2050
production_2050 = future_predict[-1]
plt.scatter(2050, production_2050, color='green', s=100, zorder=5)
plt.annotate(f'2050: {production_2050:.2f}', (2050, production_2050), 
             xytext=(10, 10), textcoords='offset points')

plt.show()

print(f"Predicted honey production in 2050: {production_2050:.2f}")

