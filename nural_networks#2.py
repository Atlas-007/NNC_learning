import numpy as np
import pandas
import math


X = np.load('iris_features.npy')
np.random.seed(3)



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        pass
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss():
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_entropy(Loss):
    def forward(self, y_pred,y_true):
        sample = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clip[range(sample),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clip*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

#num_samples = X.shape[0]
#y_true = np.random.randint(0, 3, size=num_samples)
y_true = np.array([2, 2, 2, 2, 2])
print(y_true)

layer1 = Layer_Dense(4,5)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(5, 3)
activation2 = Activation_softmax()


layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)

loss_function = Loss_entropy()
loss = loss_function.calculate(activation2.output, y_true)

print(f"loss: {loss}")
