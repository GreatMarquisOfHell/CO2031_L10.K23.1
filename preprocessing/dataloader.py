import numpy as np
from .dataset import *
#from preprocessing.dataset import *

class DataLoader:
    class Iterator:
        # constructor of class Iterator
        def __init__(self, dataloader, index = 0):
            self.dataloader = dataloader
            self.index = index 
        
        def __init__(self):
            return self    

        # operator != 
        def __ne__(self, other):
            return self.index != other.index
            

        # obtain an iterator from an iterable object
        def __iter__(self):
            return self

        # sequential traversal of the elements in the object
        
        def __next__(self):
            if (self.index >= len(self.dataloader.batches)):
                raise StopIteration
            batch = self.dataloader.batches[self.index]
            self.index+=1
            return batch

        # access the elements of an object by index
        
        def __getitem__(self, index):
            return self.dataloader.batches[index]

    # constructor of class DataLoader    
        # iterate through each batch.
        
            # at the last batch
        
                # if drop_last is true, discard the remainder
                
                # else include the remainder in the last batch
                
            # sort samples include data and label (shuffled if necessary) into each batch.
            
            # array contains batches of samples
    
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
        num_batches = len(self.dataset) // self.batch_size #get floor num batches
        remainder = len(self.dataset) % self.batch_size
        
        for i in range(num_batches):
            batches_indices = self.indices[i * self.batch_size : (i+1) * self.batch_size -1]
            batches.append([self.dataset[idx] for idx in batches_indices])
            
        if remainder > 0 and not self.drop_last:
            batches_indices = self.indices[num_batches * self.batch_size :]
            batches.append([self.dataset[idx] for idx in batches_indices])
        
        return batches

    
    # return an Iterator object to iterate over the elements in the batch
    def __iter__(self):
        return DataLoader.Iterator(self)
    
    # return the element at the index in the batch
    def __getitem__(self, index):
        return self.batches[index]

    # get number of samples in each batch  
    def num_samples_each_batch(self):
        return self.batch_size

    # get number of samples
    def num_samples(self):
        return len(self.dataset)
    
    # get number of batches
    def num_batches(self):
        return len(self.dataset) // self.dataset 

    

