
import numpy as np

X = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
test_size = 0.2

def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    print(X[indices], y[indices])
    return X[indices], y[indices]


def train_test_split(X, y, test_size):
    X, y = shuffle_data(X, y)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
print(X_train, X_test, y_train, y_test)


