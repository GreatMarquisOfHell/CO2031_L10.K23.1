import numpy as np
from ann.layer.ILayer import ILayer


class FCLayer(ILayer):
    def __init__(self, X_size, Y_size):
        super().__init__()
        rng = np.random.default_rng()
        self.Weight = rng.standard_normal((Y_size, X_size)) * 0.001
        self.Bias = rng.random((Y_size, 1))

    def forward(self, X):
        self.input = X
        self.output = np.dot(self.Weight, X) + self.Bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        self.Weight -= learning_rate * np.dot(output_gradient, self.input.T)
        self.Bias -= learning_rate * output_gradient
        return np.dot(self.Weight.T, output_gradient)
