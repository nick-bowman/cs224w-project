import os
import torch
import numpy as np
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import InMemoryDataset, Data
import csv
import snap
from utils.util import project_base, generate_file_name, generate_diff_file_name

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def create_masks(num_positive, num_negative, train, test, val):
    num_positive_train = int(train * num_positive)
    num_positive_test = int(test * num_positive)
    num_positive_val = num_positive - num_positive_train - num_positive_test
        
    num_negative_train = int(train * num_negative)
    num_negative_test = int(test * num_negative)
    num_negative_val = num_negative - num_negative_train - num_negative_test
        
#     print(num_positive_train, num_positive_test, num_positive_val)
#     print(num_negative_train, num_negative_test, num_negative_val)
        
    mask_positive = torch.zeros(num_positive)
    mask_positive[:num_positive_train] = 1
    mask_positive[num_positive_train:num_positive_train + num_positive_test] = 2
    mask_positive[num_positive_train + num_positive_test:] = 3
    mask_positive = mask_positive[torch.randperm(num_positive)]
#     print(mask_positive)
        
    train_mask_positive = torch.zeros(num_positive)
    train_mask_positive[mask_positive == 1] = 1
        
    test_mask_positive = torch.zeros(num_positive)
    test_mask_positive[mask_positive == 2] = 1
        
    val_mask_positive = torch.zeros(num_positive)
    val_mask_positive[mask_positive == 3] = 1
        
#     print(train_mask_positive)
#     print(test_mask_positive)
#     print(val_mask_positive)
    
    mask_negative = torch.zeros(num_negative)
    mask_negative[:num_negative_train] = 1
    mask_negative[num_negative_train:num_negative_train + num_negative_test] = 2
    mask_negative[num_negative_train + num_negative_test:] = 3
    mask_negative = mask_negative[torch.randperm(num_negative)]
#     print(mask_negative)
        
    train_mask_negative = torch.zeros(num_negative)
    train_mask_negative[mask_negative == 1] = 1
        
    test_mask_negative = torch.zeros(num_negative)
    test_mask_negative[mask_negative == 2] = 1
        
    val_mask_negative = torch.zeros(num_negative)
    val_mask_negative[mask_negative == 3] = 1
        
#     print(train_mask_negative)
#     print(test_mask_negative)
#     print(val_mask_negative)

    train_mask = torch.tensor(torch.cat((train_mask_positive, train_mask_negative)), dtype=torch.bool)
    test_mask = torch.tensor(torch.cat((test_mask_positive, test_mask_negative)), dtype=torch.bool)
    val_mask = torch.tensor(torch.cat((val_mask_positive, val_mask_negative)), dtype=torch.bool)
    
    return train_mask, test_mask, val_mask

class WikiGraphsInMemoryDataset(InMemoryDataset):
    def __init__(self, lang, current_year, future_year, sample_factor=0.1, transform=None, pre_transform=None, num_node_features=100, train=0.8, test=0.1, val=0.1):
        self.lang = lang
        self.current_year = current_year
        self.future_year = future_year
        self.sample_factor = sample_factor
        self.feature_dim = num_node_features
        self.train = train
        self.test = test
        self.val = val
        super(WikiGraphsInMemoryDataset, self).__init__(os.path.join(project_base, "datasets"), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        
                                                    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f"{self.lang}-{self.current_year}-{self.future_year}-data.pt"]
    
    @property
    def num_node_features(self):
        return self.feature_dim

    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def process(self):
        data_list = []
        graph_file = generate_file_name(self.lang, self.current_year)
        diff_file = generate_diff_file_name(self.lang, self.current_year, self.future_year)
        
        g = snap.LoadEdgeList(snap.PNGraph, graph_file, 0, 2, "\t")
        edge_index = torch.tensor([[0] * g.GetEdges(), [0] * g.GetEdges()])
        i = 0
        for e in g.Edges():
            edge_index[0][i] = e.GetSrcNId()
            edge_index[1][i] = e.GetDstNId()
            i += 1
        
        # TODO: replace with more complex node feature generation
        x = torch.ones(size=(g.GetNodes(), self.feature_dim))
        
        diff_g = snap.LoadEdgeList(snap.PNGraph, diff_file, 0, 1, "\t")
        num_positive = diff_g.GetEdges()
        num_edges_in_dataset = int(num_positive / self.sample_factor)
        num_negative = num_edges_in_dataset - num_positive
        edge_list = torch.tensor([[0] * num_edges_in_dataset, [0] * num_edges_in_dataset])
        y = torch.zeros(num_edges_in_dataset)
        i = 0
        for e in diff_g.Edges():
            edge_list[0][i] = e.GetSrcNId()
            edge_list[1][i] = e.GetDstNId()
            i += 1
        
        y[:num_positive] = 1
                
        while i < num_edges_in_dataset:
            src = g.GetRndNId()
            dst = g.GetRndNId()
            if not g.IsEdge(src, dst) and not diff_g.IsEdge(src, dst):
                edge_list[0][i] = src
                edge_list[1][i] = dst
                i += 1
        
        train_mask, test_mask, val_mask = create_masks(num_positive, num_negative, self.train, self.test, self.val)
                
        data = Data(x=x, edge_index=edge_index, y=y)
        data.eval_edges = edge_list
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask
        data_list.append(data)     
    
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
if __name__ == "__main__":
    test = WikiGraphsInMemoryDataset("fr", 2002, 2003)
    print(test[0])
    
    
    

    