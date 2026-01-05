import numpy as np
import pandas as pd

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
X = data.drop('species', axis=1).values  
labels = data['species'].values  


unique_labels, y_true = np.unique(labels, return_inverse=True)
num_classes = len(unique_labels)


def to_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# Shuffle
def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

# Split data into training and testing
def train_test_split(X, y, test_size):
    X, y = shuffle_data(X, y)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

# Parameters
test_size = 0.2  
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size)

# Convert labels
y_train_one_hot = to_one_hot(y_train, num_classes)
y_test_one_hot = to_one_hot(y_test, num_classes)

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

# Initialize layers
layer1 = Layer_Dense(X_train.shape[1], 5)   # Input layer with 4 features, hidden layer with 5 neurons
activation1 = Activation_ReLU()      # ReLU activation function for hidden layer
layer2 = Layer_Dense(5, num_classes) # Second layer with 5 neurons, output layer with 3 classes
activation2 = Activation_Softmax()   # Softmax activation function for output layer

# Training parameters
learning_rate = 0.001
epochs = 400

# Training 
for epoch in range(epochs):
    # Forward pass
    layer1.forward(X_train)               
    activation1.forward(layer1.output)  
    layer2.forward(activation1.output)   
    activation2.forward(layer2.output)   

     # Calculate loss
    loss_function = Loss_Entropy()
    loss = loss_function.calculate(activation2.output, y_train)
    print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Backward pass
    dvalues = activation2.output - y_train_one_hot
    dvalues = activation2.backward(dvalues)
    dvalues = layer2.backward(dvalues)
    dvalues = activation1.backward(dvalues)
    dvalues = layer1.backward(dvalues)

    
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

# Testing phase
layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)


y_pred = np.argmax(activation2.output, axis=1)

# Calculate loss and accuracy
loss_function = Loss_Entropy()
test_loss = loss_function.calculate(activation2.output, y_test_one_hot)
accuracy = np.mean(y_pred == y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

