import os
from datetime import datetime
import time
from csv import DictReader
import numpy as np
from tqdm import tqdm
import cugraph
import cudf
import mmap

home = os.path.expanduser("~")
base = os.path.join(home, "WikiLinksGraph/WikiLinksGraph")
project_base = os.path.join(home, "cs224w-project")
rolx_base = os.path.join(home, "RolXEmbeddings")
GOLD_DIR = "gold_edges"

def generate_file_name(language, year):
    file_name = "%swiki.wikilink_graph.%s-03-01.csv" % (language, str(year))
    return os.path.join(base, file_name)

def generate_gold_file_name(lang, curr, future):
    file_name = f"{lang}wiki-{curr}-{future}-diff.csv"
    return os.path.join(project_base, "gold_edges", file_name)

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

def timestamp_to_int(timestamp):
    """
    Takes a string timestamp in the form
    
        year-month-day hour-minute-second timezone
        
    and converts it into an integer.
    
    Parameters
    ----------
    timestamp : str
        Timestamp in the form specified above.
        
    Returns
    -------
    int_timestamp : int
        Timestamp in integer form.
    """
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S %Z")
    return int(time.mktime(dt.timetuple()))

def write_temporal_edge_list_networkx(temporal_graph_csv):
    """
    Creates a networkx-readable edge list with timestamps for each edge.
    
    Parameters
    ----------
    temporal_graph_csv : str
        Absolute path to the .csv file containing the temporal graph data.
    """
    graph_file = open(temporal_graph_csv, 'r')
    filename = os.path.splitext(temporal_graph_csv)[0]
    edge_list_file = open(filename + "_edgelist_networkx.txt", 'w')
    reader = DictReader(graph_file)
    for row in reader:
        timestamp = row["timestamp"]
        int_timestamp = timestamp_to_int(timestamp)
        start_id = str(row["start_id"])
        end_id = str(row["end_id"])
        data_dict = {"timestamp" : int_timestamp}
        line = ' '.join((start_id, end_id, str(data_dict)))
        edge_list_file.write(line + '\n')

    graph_file.close()
    edge_list_file.close()
    
def write_temporal_edge_list_snap(temporal_graph_csv):
    """
    Creates a Snap-readable edge list with timestamps for each edge.
    
    Parameters
    ----------
    temporal_graph_csv : str
        Absolute path to the .csv file containing the temporal graph data.
    """
    graph_file = open(temporal_graph_csv, 'r')
    filename = os.path.splitext(temporal_graph_csv)[0]
    
    reader = DictReader(graph_file)
    seen_nodes = set()
    edge_count = 0
    with open(temporal_graph_csv, 'r') as graph_file:
        reader = DictReader(graph_file)
        for row in reader:
            start_id = str(row["start_id"])
            end_id = str(row["end_id"])
            seen_nodes.add(start_id)
            seen_nodes.add(end_id)
            edge_count += 1
    node_count = len(seen_nodes)
    
    edge_list_filename = filename + "_edgelist_snap.txt"
    edge_list_file = open(edge_list_filename, 'w')
    edge_list_file.write("# Directed network: {} \n".format(edge_list_filename))
    edge_list_file.write("# Temporal edge list.\n")
    edge_list_file.write("# Nodes: {} Edges: {}\n".format(node_count, edge_count))
    edge_list_file.write("#NODES\tNId\n")
    for node in seen_nodes:
        edge_list_file.write(str(node) + '\n')
    edge_list_file.write("#END\n")
    edge_list_file.write("#EDGES\tSrcNId\tDstNId\tFlt:Timestamp\n")
    with open(temporal_graph_csv, 'r') as graph_file:
        reader = DictReader(graph_file)
        for row in reader:
            timestamp = str(timestamp_to_int(row["timestamp"]))
            start_id = str(row["start_id"])
            end_id = str(row["end_id"])
            line = '\t'.join((start_id, end_id, timestamp))
            edge_list_file.write(line + '\n')
    edge_list_file.write("#END\n")
    edge_list_file.close()

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

def load_temporal_features(filename):
    data_file = os.path.join(home, filename)
    with open(data_file, "r") as f:
        lines = f.readlines()[1:]
    embeds = np.loadtxt(lines)
    mappings = {}
    for i in tqdm(range(embeds.shape[0])):
        node_id = int(embeds[i][0])
        mappings[node_id] = i
    embeds = embeds[:,1:]
    return embeds, mappings

def load_cugraph(lang, year):
    G = cugraph.Graph()
    
    file_name = generate_file_name(lang, year)

    gdf = cudf.read_csv(file_name, usecols=[0,2], dtype=["int32", "str", "int32", "str"], sep="\t")

    sources = cudf.Series(gdf["page_id_from"])
    destinations = cudf.Series(gdf["page_id_to"])
    source_col, dest_col, renumbering_map = cugraph.renumber(sources, destinations)

    G.add_edge_list(source_col, dest_col, None)
    
    return G, renumbering_map

def get_num_lines(file_path):
    """
    Calculates and returns the number of lines in a file. Helpful for 
    tqdm progress bar for file reading. 
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines