import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

# Load the dataset
data = pd.read_csv('Q2.csv')

# Assuming the last column is the target variable y and the rest are features X
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Display the coefficients
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Predict y for a randomly chosen x value
random_index = random.randint(0, len(X) - 1)
x_random = X[random_index].reshape(1, -1)
x_random_poly = poly.transform(x_random)
y_pred = model.predict(x_random_poly)
print(f"Predicted y value for x={x_random}: {y_pred}")

# Compute and display the RMSE
y_pred_all = model.predict(X_poly)
rmse = np.sqrt(mean_squared_error(y, y_pred_all))
print("Root Mean Squared Error (RMSE):", rmse)

# Explain the accuracy of the function
if rmse < 10:  # This threshold is arbitrary and should be adjusted based on the specific problem
    print("The function is acceptable in terms of accuracy of prediction.")
else:
    print("The function is not acceptable in terms of accuracy of prediction.")