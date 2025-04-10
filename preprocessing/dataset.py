import numpy as np
from abc import ABC, abstractmethod

class DataLabel:
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
     def __repr__(self):
         return f"DataLabel(data_shape = {self.data.shape}, label = {self.label})"


class Batch:  # This class will store multiple images
    def __init__(self, data_list, label_list):
        self.data_list = np.array(data_list)
        self.label_list = np.array(label_list)
    
    def __repr__(self):
         return f"Batch(data_list_shape = {self.data_list.shape}, label_list = {self.label_list})"    
        
class Dataset(ABC):
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def getimage(self, index):  
        pass



class TensorDataset(Dataset):
    def __init__(self, dataset, labels):
        
        assert len(dataset) == len(labels)
        self.dataset = dataset
        self.labels = labels
        
        self.data_shape = self.dataset.shape
        self.labels_shape = self.labels.shape
       

    def length(self):
        return len(self.data_shape[0])
    
    def getimage(self, index) -> DataLabel:  
        return DataLabel(self.dataset[index], self.labels[index])
