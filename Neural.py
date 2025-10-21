# Dataset: MNIST (you can use sklearn.datasets.load_digits)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()
X, y = digits.data / 16.0, digits.target
Y = np.eye(10)[y]  # One-hot

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Define your own forward/backward pass and train with gradient descent
