import numpy as np
from ann.layer.ILayer import ILayer


class Activation(ILayer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self. activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.input))
