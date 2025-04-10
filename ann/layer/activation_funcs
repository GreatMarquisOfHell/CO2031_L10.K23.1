import numpy as np
from ann.layer.ILayer import ILayer
from ann.layer.activation import Activation


class Softmax(ILayer):
    def forward(self, input):
        self.input = input
        shifted_input = input - np.max(input) 
        exp_values = np.exp(shifted_input)
        self.output = exp_values / np.sum(exp_values)
        return self.output

    def backward(self, output_gradient, learning_rate):
        s = self.output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
        return np.dot(jacobian_matrix, output_gradient)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def sigmoid_derivative(x):
            sig = sigmoid(x)
            return sig * (1.0 - sig)

        super().__init__(sigmoid, sigmoid_derivative)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_derivative(x):
            return np.where(x > 0, 1.0, 0.0)

        super().__init__(relu, relu_derivative)


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1.0 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_derivative)
