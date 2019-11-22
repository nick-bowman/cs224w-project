import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import snap

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SupervisedWikiGraphsDataset(Dataset):
    def __init__(self, lang, current_year, future_year, sample_factor=0.1):
        self.data = []

    
    
    def __len__(self):
        return 100
    
    def __getitem__(self, i):
        data_val, target = self.data[i]
        return data_val, target
    
if __name__ == "__main__":
    test = SupervisedWikiGraphsDataset("es", 2002, 2003)
    
    

    