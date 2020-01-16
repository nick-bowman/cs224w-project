from sklearn.metrics import precision_recall_fscore_support

import networkx as nx
import numpy as np

import cugraph
import cudf

import time 

import sys
sys.path.append("..")

from utils.util import load_cugraph, generate_test_file_name, generate_training_file_name

def jaccard_baseline_test(lang, year, k=1):
    G, renumbering_map = load_cugraph(lang, year)
    test_file = generate_test_file_name(lang, year, k)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    
    edges_df = cudf.read_csv(test_file, sep="\t")
    print(edges_df)
    
    sources = cudf.Series(edges_df["page_id_from"])
    destinations = cudf.Series(edges_df["page_id_to"])
    labels = cudf.Series(edges_df["label"])
    
    num_positive = labels.sum()

    keys = list(renumbering_map)
    vals = list(range(len(keys)))
    sources = sources.replace(keys, vals)
    destinations = destinations.replace(keys, vals)
    
    print(sources, destinations)
    
    pairs = cudf.DataFrame()
    pairs.add_column("sources", sources)
    pairs.add_column("destinations", destinations)
    print(pairs)
    
    results = cugraph.jaccard(G, pairs)
#     results = cugraph.jaccard(G)
    print(results)
    
    
    
if __name__ == "__main__":
    jaccard_baseline_test("en", 2001)