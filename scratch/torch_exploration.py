import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import snap

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset[0])
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
print(len(dataset))

