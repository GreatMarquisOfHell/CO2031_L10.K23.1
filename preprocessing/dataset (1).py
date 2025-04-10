import numpy as np
from abc import ABC, abstractmethod


# e.g. The image's size  is 28x28 (a 2D matrix). So the data variable will be a 2D matrix.
# If each batch contains 10 images, then the data variable in it is 10x28x28 (a 3D matrix)

# The label variable in the DataLabel class is a 1D matrix of size (1x4) (horse, cat, dog, chicken)
# The element at the corresponding position to the animal will be 1 and the rest will be 0

class DataLabel:  # This class will store a pair of 1 image and its label
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    # def __repr__(self):
    #     return f"DataLabel(data_shape = {self.data.shape}, label = {self.label})"


class Batch:  # This class will store multiple images
    def __init__(self, data_list, label_list):
        self.data_list = np.array(data_list)
        self.label_list = np.array(label_list)
    
    # def __repr__(self):
    #     return f"Batch(data_list_shape = {self.data_list.shape}, label_list = {self.label_list})"    
        
class Dataset(ABC):
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def getimage(self, index):  # This is the getitem method in the major assignment 1
        pass


# This class and its data variable will hold the whole dataset
# All the images will be load into this class in the implementation step
# e.g.
# images = np.array([image0, image1,image2])
# labels = np.array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1])
# TensorDataset data_images(images, labels)
class TensorDataset(Dataset):
    def __init__(self, dataset, labels):
        # TODO: need to initialize the attributes:
        #   * 1. data, label;
        #   * 2. data_shape, label_shape
        assert len(dataset) == len(labels)
        self.dataset = dataset
        self.labels = labels
        
        self.data_shape = self.dataset.shape
        self.labels_shape = self.labels.shape
       

    def length(self):
        # TODO: return the size of the first dimension (dimension 0)
        return len(self.data_shape[0])
    
    def getimage(self, index) -> DataLabel:  # This is the getitem method in the major assignment 1
        # TODO: return the data item (of type: DataLabel) that is specified by index
        return DataLabel(self.dataset[index], self.labels[index])
