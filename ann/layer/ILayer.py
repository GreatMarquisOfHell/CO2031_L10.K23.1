from abc import ABC, abstractmethod


class ILayer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass
