import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from datasets import WikiGraphsInMemoryDataset
from models.gnn import GNNStack

import torch.optim as optim

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def train(dataset, args, dev):
    loader = DataLoader(dataset, shuffle=True)#, batch_size=args.batch_size, shuffle=True)

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args, dev).to(dev)
    scheduler, opt = build_optimizer(args, model.parameters())
        
    train_accs = []
    test_accs = []
    val_accs = []
    # train
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print(total_loss)

        if epoch % 10 == 0:
            train_acc = test(loader, model)
            train_accs.append(train_acc)
            test_acc = test(loader, model, is_test=True)
            print(test_acc,   '  test', epoch)
            test_accs.append(test_acc)
            val_acc = test(loader, model, is_validation=True)
            val_accs.append(val_acc)
    
    return train_accs, test_accs, val_accs

def test(loader, model, is_test=False, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y
        
        mask = data.train_mask
        if is_test:
            mask = data.test_mask
        elif is_validation:
            mask = data.val_mask
            
        pred = pred[mask]
        label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    total = 0
    for data in loader.dataset:
        mask = data.train_mask
        if is_test:
            mask = data.test_mask
        elif is_validation:
            mask = data.val_mask
        total += torch.sum(mask).item()
    
    return correct / total
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WikiGraphsInMemoryDataset("es", 2004, 2005, dev)
#     dataset = Planetoid(root='/tmp/Cora', name='Cora')
    args_list = [
#         {
#             'model_type': 'GCN',
#             'num_layers': 4, 
#             'batch_size': 32,
#             'hidden_dim': 64,
#             'dropout': 0.25, 
#             'epochs': 500,
#             'opt': 'adam',
#             'opt_scheduler': 'none',
#             'opt_restart': 0,
#             'weight_decay': 5e-3,
#             'lr': 0.01
#         },
        {
            'model_type': 'GraphSage',
            'num_layers': 4, 
            'batch_size': 32,
            'hidden_dim': 64,
            'dropout': 0.50, 
            'epochs': 500,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'weight_decay': 5e-3,
            'lr': 0.01
        }  
    ]
    for args in args_list:
        args = objectview(args)
        train_accs, test_accs, val_accs = train(dataset, args, dev)
        xs = list(range(0,args.epochs, 10))
        plt.plot(xs, train_accs,label="Train")
        plt.plot(xs, test_accs,label="Test")
        plt.plot(xs, val_accs,label="Validation")
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Epochs for Spanish 2002-2003 Dataset')
        plt.legend()
        plt.savefig('plot.png')
        # TODO: plot train/val loss data here

if __name__ == "__main__":
    main()
  
  