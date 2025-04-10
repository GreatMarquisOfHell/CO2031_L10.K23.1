import numpy as np

def train(network, loss, loss_prime, x_train, y_train, epoch=1000, learning_rate=0.01, verbose=True):
    for e in range(epoch):
        error = 0
        for x, y in zip(x_train, y_train):
            output = x

            for layer in network:
                output = layer.forward(output)

            error += loss(y, output)

            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        if verbose and (e + 1) % 100 == 0:
            print(f"Epoch {e+1}/{epoch}, Error = {error / len(x_train):.6f}")



def predict(network, input_data):
    results = []
    for x in input_data:
        output = x
        for layer in network:
            output = layer.forward(output)
        results.append(output)
    return np.array(results)

