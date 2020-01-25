from sklearn.metrics import precision_recall_fscore_support

import networkx as nx
import numpy as np

import cugraph
import cudf

import time 

import sys
sys.path.append("..")

from time import sleep
from tqdm import tqdm

from utils.util import load_cugraph, generate_test_file_name, generate_training_file_name

def calculate_f1_score(top_n, sources, destinations, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in tqdm(range(len(sources))):
        if i in top_n.index:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == 0:
                tn += 1
            else:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def jaccard_baseline_test(G, sources, destinations, labels, num_positive):
    results = cugraph.jaccard(G, first=sources, second=destinations).to_pandas().dropna()
    top_n = results.nlargest(num_positive, "jaccard_coeff")
    print("Jaccard", calculate_f1_score(top_n, sources, destinations, labels))
    
def overlap_baseline_test(G, sources, destinations, labels, num_positive):        
    results = cugraph.overlap(G, first=sources, second=destinations).to_pandas().dropna()
    top_n = results.nlargest(num_positive, "overlap_coeff")
    print("Overlap", calculate_f1_score(top_n, sources, destinations, labels))

def init_experiments(lang, year, k=1):
    G, renumbering_map = load_cugraph(lang, year)
        
    test_file = generate_test_file_name(lang, year, k)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    
    edges_df = cudf.read_csv(test_file, sep="\t")
    
    sources = cudf.Series(edges_df["page_id_from"], dtype=np.int32)
    destinations = cudf.Series(edges_df["page_id_to"], dtype=np.int32)
    labels = cudf.Series(edges_df["label"])
    
    num_positive = labels.sum()

    keys = list(renumbering_map)
    vals = list(range(len(keys)))
    sources = sources.replace(keys, vals)
    destinations = destinations.replace(keys, vals)
    
    return G, sources, destinations, labels, num_positive
    
    
if __name__ == "__main__":
    G, sources, destinations, labels, num_positive = init_experiments("en", 2017)
    jaccard_baseline_test(G, sources, destinations, labels, num_positive)
    overlap_baseline_test(G, sources, destinations, labels, num_positive)