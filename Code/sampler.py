import torch
import torchvision
import torch.utils.data
import random


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset = {}
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
    
        self.keys = list(self.dataset.keys())
        self.currentkey = 0

    def __iter__(self):
        while len(self.dataset[self.keys[self.currentkey]]) > 0:
            yield self.dataset[self.keys[self.currentkey]].pop()
            self.currentkey = (self.currentkey + 1) % len(self.keys)

    
    def _get_label(self, dataset, idx):
        return str(dataset.data[idx][3])

    def __len__(self):
        return self.balanced_max*len(self.keys)