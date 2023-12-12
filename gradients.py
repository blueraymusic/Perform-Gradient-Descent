import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to the input features
X_b = np.c_[np.ones((100, 1)), X]

# Set up initial parameters
theta = np.random.randn(2, 1)

# Set hyperparameters
learning_rate = 0.01
n_iterations = 1000

# Define the number of data points
m = len(X)

# Perform gradient descent
for iteration in range(n_iterations):
    gradients = 1/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

# Print the final parameters
print("Final Parameters:", theta)

# Plot the original data and the best-fit line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.show()
