import numpy as np
import pandas as pd

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
X = data.drop('species', axis=1).values  # Features
labels = data['species'].values  # True labels

# Convert labels to numeric format manually
unique_labels, y_true = np.unique(labels, return_inverse=True)
num_classes = len(unique_labels)

# Neural Network Classes
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        samples = len(dvalues)
        if samples > 1:
            # Calculate Jacobian matrix for softmax
            self.dinputs = np.empty_like(dvalues)
            for i in range(samples):
                jacobian_matrix = np.diag(self.output[i]) - np.outer(self.output[i], self.output[i])
                self.dinputs[i] = np.dot(dvalues[i], jacobian_matrix)
        else:
            jacobian_matrix = np.diag(self.output) - np.outer(self.output, self.output)
            self.dinputs = np.dot(dvalues, jacobian_matrix)
        return self.dinputs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Entropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clip[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clip * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Convert labels to one-hot encoding for loss calculation
def to_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# Initialize layers
layer1 = Layer_Dense(X.shape[1], 5)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(5, num_classes)
activation2 = Activation_Softmax()

# Training parameters
learning_rate = 0.01
epochs = 100

# Convert y_true to one-hot encoding
y_true_one_hot = to_one_hot(y_true, num_classes)

for epoch in range(epochs):
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Calculate loss
    loss_function = Loss_Entropy()
    loss = loss_function.calculate(activation2.output, y_true)
    print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Backward pass
    dvalues = activation2.output - y_true_one_hot
    dvalues = activation2.backward(dvalues)
    dvalues = layer2.backward(dvalues)
    dvalues = activation1.backward(dvalues)
    dvalues = layer1.backward(dvalues)

    # Update weights and biases
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

