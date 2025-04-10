from ann.layer.FCLayer import FCLayer
from ann.layer.convolutional import Convolutional
from ann.layer.reshape import Reshape
from ann.layer.loss_funcs import CCE, CCE_derivative
from ann.layer.activation_funcs import Sigmoid, Softmax
from ann.model.network import train
from preprocessing.image2matrix import img2matr
import os
import numpy as np
import random

Label_map = {
    "Horses": [[1.], [0.], [0.], [0.]],
    "Dogs": [[0.], [1.], [0.], [0.]],
    "Cats": [[0.], [0.], [1.], [0.]],
    "Chickens": [[0.], [0.], [0.], [1.]]
}

def load_n_preprocess_data(data_dir, img_size=(64, 64), num_sample_each_class=100):
    samples = []
    for label, one_hot in Label_map.items():
        class_path = os.path.join(data_dir, label)
        if not os.path.exists(class_path):
            break
        files = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".png"))][:num_sample_each_class]
        for file in files:
            path = os.path.join(class_path, file)
            img = img2matr(path, img_size)
            samples.append((img, one_hot))
    return samples

data = load_n_preprocess_data("data_images")
random.shuffle(data)
x_train = [x for x, _ in data]
y_train = [y for _, y in data]

network = [
    Convolutional((3, 64, 64), 3, 8),
    Sigmoid(),
    Reshape((8, 62, 62), (8 * 62 * 62, 1)),
    FCLayer(30752, 128),
    Sigmoid(),
    FCLayer(128, 4),
    Softmax()
]

train(
    network,
    CCE,
    CCE_derivative,
    x_train,
    y_train,
    500,
    0.1
)
