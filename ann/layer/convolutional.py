import numpy as np
from scipy import signal
from ann.layer.ILayer import ILayer


class Convolutional(ILayer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.filter = np.random.randn(depth, input_shape[0], kernel_size, kernel_size)
        self.bias = np.random.randn(depth, 1, 1)

    def forward(self, input):
        self.input = input
        c, h, w = input.shape
        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1
        output = np.zeros((self.depth, out_h, out_w))

        for i in range(self.depth):
            for j in range(c):
                output[i] += signal.correlate(input[j], self.filter[i, j], mode='valid')
            output[i] += self.bias[i]

        return output

    def backward(self, output_gradient, learning_rate):
        self.bias -= learning_rate * np.sum(output_gradient, axis=(1, 2), keepdims=True)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_shape[0]):
                self.filter[i, j] -= learning_rate * signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += signal.correlate2d(output_gradient[i], self.filter[i, j], mode='full')

        return input_gradient
