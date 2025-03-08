import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to perform linear regression and plot the results
def linear_regression_and_plot(X, Y, x_predict):
    # Reshape the data
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    # Create and train the model
    model = LinearRegression()
    model.fit(X, Y)

    # Predict the y value for the given x_predict
    y_predict = model.predict(np.array([[x_predict]]))

    # Display the polynomial equation
    w0 = model.intercept_
    w1 = model.coef_[0]
    print(f"The polynomial equation is: y = {w0:.2f} + {w1:.2f}x")

    # Display the predicted y value
    print(f"The predicted y value when x = {x_predict} is: {y_predict[0]:.2f}")

    # Plot the dataset and the regression line
    plt.scatter(X, Y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.title('Scatter plot with regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return model

# Function to compute error metrics
def compute_error_metrics(Y, Y_pred):
    total_error = np.sum(Y - Y_pred)
    squared_error = np.sum((Y - Y_pred) ** 2)
    mean_squared_error_value = mean_squared_error(Y, Y_pred)
    rooted_mean_squared_error = np.sqrt(mean_squared_error_value)

    print(f"Total Error: {total_error:.2f}")
    print(f"Squared Error: {squared_error:.2f}")
    print(f"Mean Squared Error: {mean_squared_error_value:.2f}")
    print(f"Rooted Mean Squared Error: {rooted_mean_squared_error:.2f}")

# First dataset
X1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y1 = [2.5, 4.1, 5.6, 7.2, 8.8, 10.3, 11.9, 13.5, 15.0, 16.8]
model1 = linear_regression_and_plot(X1, Y1, 100)
Y1_pred = model1.predict(np.array(X1).reshape(-1, 1))
compute_error_metrics(Y1, Y1_pred)

# Second dataset
X2 = [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]
Y2 = [17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4]
model2 = linear_regression_and_plot(X2, Y2, 0.5)
Y2_pred = model2.predict(np.array(X2).reshape(-1, 1))
compute_error_metrics(Y2, Y2_pred)

# Display the coefficients of the second model
print(f"The coefficients of the second model are: {model2.intercept_:.2f}, {model2.coef_[0]:.2f}")