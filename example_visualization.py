import networkx as nx

import matplotlib.pyplot as plt

from utils.util import generate_file_name

from tqdm import tqdm
from datasets import WikiGraphsInMemoryDataset
import collections
import numpy as np

def load_networkx_graph(filename):
    G = nx.DiGraph()
    
    with open(filename, "r") as f:
        lines = f.readlines()
    
    edges = [(int(t[0]), int(t[2])) for t in [l.split("\t") for l in lines[1:]]]
    G.add_edges_from(edges)
    
    return G

def visualize_subgraph(nodes_of_interest, lang, curr, nxt, filename):
    curr_filename = generate_file_name(lang, curr)
    
    G = load_networkx_graph(filename)
    
    no_existing_edges = []
    
    with open("./experiments/plots/es-2005-2006-GCN-12-06-2019-19:32:18/es-2005-2006-predictions.txt", "r") as f:
        for line in tqdm(f.readlines()):
            l = [int(x) for x in line.split()]
            if l[3] == 1 and not G.has_edge(l[1], l[0]) and not G.has_edge(l[0], l[1]):
                no_existing_edges.append((l[0], l[1]))
    
    print(no_existing_edges[:20])
    
    
    shared_neighbors = set(G.neighbors(nodes_of_interest[0]))
    for node in nodes_of_interest:
        shared_neighbors = shared_neighbors.intersection(set(G.neighbors(node)))
    nodelist_to_plot = sorted(list(set(nodes_of_interest + list(shared_neighbors))))
    subgraph = G.subgraph(nodelist_to_plot)
    plt.figure(figsize=(10,10))
    node_colors = ['r'] * len(nodelist_to_plot)
    for node in nodes_of_interest:
        node_colors[nodelist_to_plot.index(node)] = 'g'
    shells = [nodes_of_interest, [node for node in nodelist_to_plot if node not in nodes_of_interest]]
    pos = nx.shell_layout(subgraph, shells)
    sign = 1
    for node in nodes_of_interest:
        print(pos[node])
        pos[node][1] += sign * 0.1
        sign = -1 if sign == 1 else 1
    nx.draw_networkx(subgraph, pos=pos, font_size=8, width=0.5, node_size=400, nodelist=nodelist_to_plot, node_color=node_colors)
    plt.savefig(filename)

    
def plot_common_neighbor_distribution(lang, curr, nxt):
    G = load_networkx_graph(generate_file_name(lang, curr)).to_undirected()
    
    data_true = collections.defaultdict(int)
    data_predicted = collections.defaultdict(int)
    
    with open("./experiments/plots/es-2005-2006-GCN-12-06-2019-19:32:18/es-2005-2006-predictions.txt", "r") as f:
        for line in tqdm(f.readlines()):
            src, dst, pred, label = [int(x) for x in line.split()]
            if pred == 1:
                common_neighbors = len(list(nx.common_neighbors(G, src, dst)))
                data_predicted[common_neighbors] += 1
            if label == 1: 
                common_neighbors = len(list(nx.common_neighbors(G, src, dst)))
                data_true[common_neighbors] += 1
    
    plt.figure(figsize=(10,10))
    plt.bar(list(data_true.keys()), data_true.values(), color='r', label="True", alpha=0.4)
    plt.bar(list(data_predicted.keys()), data_predicted.values(), color='b', label="Predicted", alpha=0.4)
    plt.title("Histogram of Common Neighbor Distribution of Positively Classified Nodes")
    plt.xlabel("Number of Common Neighbors")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.xlim(0, 100)
    plt.ylim(0, 4000)
    plt.legend()
    plt.savefig("paper_plots/common_neighbor_dist.png")

            
            
    
def main():
#     nodes_of_interest = [10, 719]
#     visualize_subgraph(nodes_of_interest, "es", 2005, 2006, "paper_plots/true_positive.png")
#     nodes_of_interest = [10, 3346]
#     visualize_subgraph(nodes_of_interest, "es", 2005, 2006, "paper_plots/false_negative.png")
#     nodes_of_interest = [4121, 10]
#     visualize_subgraph(nodes_of_interest, "es", 2005, 2006, "paper_plots/true_negative.png")
#     nodes_of_interest = [23757, 4256]
#     visualize_subgraph(nodes_of_interest, "es", 2005, 2006, "paper_plots/false_positive.png")
    plot_common_neighbor_distribution("es", 2005, 2006)


if __name__ == "__main__":
    main()