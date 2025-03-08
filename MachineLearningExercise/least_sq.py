import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('Q2.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Add a column of ones to X for the intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Compute the coefficients using the least squares formula
X_transpose = X.T
X_transpose_X = np.dot(X_transpose, X)
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
X_transpose_y = np.dot(X_transpose, y)
w = np.dot(X_transpose_X_inv, X_transpose_y)

# Print the polynomial equation
print("Polynomial equation obtained from the least square method:")
equation = "y = "
for i in range(len(w)):
    if i == 0:
        equation += f"{w[i]}"
    else:
        equation += f" + ({w[i]} * x{i})"
print(equation)

# Compute the RMSE
y_pred = np.dot(X, w)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
print(f"RMSE: {rmse}")

# Compare with the function obtained in (b) and explain which one is more accurate
# Note: You need to provide the function obtained in (b) and its RMSE for comparison
rmse_b = 120.0  # Example RMSE value for the function obtained in (b)
if rmse < rmse_b:
    print("The least square method is more accurate.")
else:
    print("The function obtained in (b) is more accurate.")