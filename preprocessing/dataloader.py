import numpy as np
from .dataset import *

class DataLoader:
    class Iterator:
        def __init__(self, dataloader, index = 0):
            self.dataloader = dataloader
            self.index = index 
        
        def __init__(self):
            return self    

        def __ne__(self, other):
            return self.index != other.index
            

        def __iter__(self):
            return self

        
        def __next__(self):
            if (self.index >= len(self.dataloader.batches)):
                raise StopIteration
            batch = self.dataloader.batches[self.index]
            self.index+=1
            return batch

        
        def __getitem__(self, index):
            return self.dataloader.batches[index]
    
    def __init__(self, dataset, batch_size, shuffle = True, drop_last = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = np.arange(len(dataset))
        if (shuffle == True):
            np.random.shuffle(self.indices)
        self.batches = self.creat_batch()
        
    def creat_batch(self):
        batches = []
        num_batches = len(self.dataset) // self.batch_size 
        remainder = len(self.dataset) % self.batch_size
        
        for i in range(num_batches):
            batches_indices = self.indices[i * self.batch_size : (i+1) * self.batch_size -1]
            batches.append([self.dataset[idx] for idx in batches_indices])
            
        if remainder > 0 and not self.drop_last:
            batches_indices = self.indices[num_batches * self.batch_size :]
            batches.append([self.dataset[idx] for idx in batches_indices])
        
        return batches

    
    def __iter__(self):
        return DataLoader.Iterator(self)
    
    def __getitem__(self, index):
        return self.batches[index]

    def num_samples_each_batch(self):
        return self.batch_size

    def num_samples(self):
        return len(self.dataset)
    
    def num_batches(self):
        return len(self.dataset) // self.dataset 

    

