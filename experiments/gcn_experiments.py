import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import traceback

from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt

import os
import sys
sys.path.append("..")

from datetime import datetime
from tqdm import tqdm

from datasets import WikiGraphsInMemoryDataset
from models.gnn import GNNStack
from utils.util import load_rolx_features, load_temporal_features

import torch.optim as optim

from sklearn.metrics import precision_recall_fscore_support


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
    print("Starting experiment with args", args)
    loader = DataLoader(dataset, shuffle=True)#, batch_size=args.batch_size, shuffle=True)

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args, dev).to(dev)
    scheduler, opt = build_optimizer(args, model.parameters())
        
    train_plot_data = []
    val_plot_data = []
    loss_data = []
    # train
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0
        val_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            with torch.no_grad():
                val_loss += F.nll_loss(pred[batch.val_mask], label[batch.val_mask], weight=torch.Tensor([1,3]).to(dev))
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch.num_graphs
            
        train_loss /= len(loader.dataset)
        val_loss /= len(loader.dataset)

        if epoch % 10 == 0:
            loss_data.append((train_loss, val_loss))
            train_acc, train_prec, train_recall, train_f1 = test(loader, model)
            train_plot_data.append((train_acc, train_prec, train_recall, train_f1))
            val_acc, val_prec, val_recall, val_f1 = test(loader, model, is_validation=True)
            val_plot_data.append((val_acc, val_prec, val_recall, val_f1))
    
    return train_plot_data, val_plot_data, loss_data, model, loader

def test(loader, model, is_test=False, is_validation=False):
    model.eval()

    correct = 0
    metrics = []
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
        metrics.append(precision_recall_fscore_support(label.cpu(), pred.cpu(), average="binary"))
            
        correct += pred.eq(label).sum().item()
    
    total = 0
    for data in loader.dataset:
        mask = data.train_mask
        if is_test:
            mask = data.test_mask
        elif is_validation:
            mask = data.val_mask
        total += torch.sum(mask).item()
    
    return correct / total, sum([t[0] for t in metrics]) / len(metrics), sum([t[1] for t in metrics]) / len(metrics), sum([t[2] for t in metrics]) / len(metrics)
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
    
    def __str__(self):
        return str(self.__dict__)
                
def generate_single_plot(train_data, val_data, epochs, ylabel, title, filename):
    xs = list(range(0, epochs, 10))
    fig, ax = plt.subplots()
    ax.plot(xs, train_data, label="Train")
    ax.plot(xs, val_data, label="Validation")
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(filename)
    
def generate_title(data, lang, curr, nxt):
    return f"{data} Plot for {lang}-wiki Data From {curr} To {nxt}"

def generate_file_name(dname, data, lang, curr, nxt, args):
    return os.path.join(dname, f"{lang}-{curr}-{nxt}-{data}-{args.model_type}-{args.num_layers}-{args.hidden_dim}-{args.dropout}-{args.epochs}.png")

def plot_results(train_data, val_data, loss_data, lang, curr, nxt, args):
    now = datetime.now()
    time = now.strftime("%m-%d-%Y-%H:%M:%S")
    dname = os.path.join("plots", f"{lang}-{curr}-{nxt}-{args.model_type}-{time}")
    os.mkdir(dname)
    
    train_accs = [t[0] for t in train_data]
    val_accs = [t[0] for t in val_data]
    generate_single_plot(train_accs, val_accs, args.epochs, "Accuracy", generate_title("Accuracy", lang, curr, nxt), generate_file_name(dname, "accuracy", lang, curr, nxt, args))
    
    train_precs = [t[1] for t in train_data]
    val_precs = [t[1] for t in val_data]
    generate_single_plot(train_precs, val_precs, args.epochs, "Precision", generate_title("Precision", lang, curr, nxt), generate_file_name(dname, "precision", lang, curr, nxt, args))
    
    train_recalls = [t[2] for t in train_data]
    val_recalls = [t[2] for t in val_data]
    generate_single_plot(train_recalls, val_recalls, args.epochs, "Recall", generate_title("Recall", lang, curr, nxt), generate_file_name(dname, "recall", lang, curr, nxt, args))
    
    train_f1s = [t[3] for t in train_data]
    val_f1s = [t[3] for t in val_data]
    generate_single_plot(train_f1s, val_f1s, args.epochs, "F1", generate_title("F1", lang, curr, nxt), generate_file_name(dname, "f1", lang, curr, nxt, args))

    train_loss = [t[0] for t in loss_data]
    val_loss = [t[1] for t in loss_data]
    generate_single_plot(train_loss, val_loss, args.epochs, "Loss", generate_title("Loss", lang, curr, nxt), generate_file_name(dname, "loss", lang, curr, nxt, args))
    
    return dname
    
def save_test_prediction_results(dataset, model, loader, dname, lang, curr, nxt, args):
    model.eval()

    for data in loader:
        print(data)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y
        
        mask = data.test_mask
            
        pred = pred[mask]
        label = data.y[mask]
        fname = os.path.join(dname, f"{lang}-{curr}-{nxt}-predictions.txt")
        with open(fname, "w") as f:
            for i in tqdm(range(pred.shape[0])):
                edge = data.eval_edges[:,i]
                f.write(f"{data.reverse_edge_map[edge[0]]}\t{data.reverse_edge_map[edge[1]]}\t{pred[i].item()}\t{label[i].item()}\n")
        
    
def run_experiment(lang, curr_year, future_year, args_list, feature_dir=None, num_node_features=None):
    try:
        print(f"Dataset: {lang}-{curr_year}-{future_year}")
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if feature_dir:
            # f, m = load_rolx_features(feature_dir)
            f, m = load_temporal_features(feature_dir)
            dataset = WikiGraphsInMemoryDataset(lang, curr_year, future_year, dev, node_features=f, node_feature_mapping=m, num_node_features=num_node_features)
        else:
            dataset = WikiGraphsInMemoryDataset(lang, curr_year, future_year, dev)
        for args in args_list:
            args = objectview(args)
            train_plot_data, val_plot_data, loss_data, model, loader = train(dataset, args, dev)
            dname = plot_results(train_plot_data, val_plot_data, loss_data, lang, curr_year, future_year, args)
            save_test_prediction_results(dataset, model, loader, dname, lang, curr_year, future_year, args)
    except Exception:
        print(traceback.format_exc())
        print("Experiment failed due to out of memory error")
  