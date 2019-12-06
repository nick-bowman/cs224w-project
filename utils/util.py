import os
import numpy as np

home = os.path.expanduser("~")
base = os.path.join(home, "WikiLinksGraph/WikiLinksGraph")
project_base = os.path.join(home, "cs224w-project")
rolx_base = os.path.join(home, "RolXEmbeddings")

def generate_file_name(language, year):
    file_name = "%swiki.wikilink_graph.%s-03-01.csv" % (language, str(year))
    return os.path.join(base, file_name)

def generate_diff_file_name(lang, curr, future):
    file_name = f"{lang}wiki-{curr}-{future}-diff.csv"
    return os.path.join(project_base, "data_diffs", file_name)

def load_diff(language, curr_year, future_year):
    all_edges = set()
    with open(generate_diff_file_name(language, curr_year, future_year), "r") as f:
        for line in f: 
            src, dst = line.split()
            all_edges.add((int(src),int(dst)))
    return all_edges

def evaluate_predicted_edges(lang, curr_year, future_year, predicted_edges, filter_nodes=None):
    true_added_edges = load_diff(lang, curr_year, future_year)
    
    to_remove = set()
    if filter_nodes:
        for e in true_added_edges:
            if e[0] not in filter_nodes or e[1] not in filter_nodes:
                to_remove.add(e)
    true_added_edges -= to_remove

    false_positive = len(predicted_edges - true_added_edges)
    false_negative = len(true_added_edges - predicted_edges)
    true_positive = len(true_added_edges.intersection(predicted_edges))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)

def load_rolx_features(dname):
    data_dir = os.path.join(rolx_base, dname)
    feature_file = os.path.join(data_dir, "v.txt")
    mappings_file = os.path.join(data_dir, "mappings.txt")
    embeds = np.loadtxt(feature_file)
    mappings = {}
    with open(mappings_file, "r") as f:
        for line in f:
            if "#" in line: continue
            ids = line.split()
            mappings[int(ids[1])] = int(ids[0])
    return embeds, mappings

    
    
