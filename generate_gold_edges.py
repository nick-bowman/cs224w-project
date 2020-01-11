import snap
import os
from tqdm import tqdm
from utils.util import generate_file_name
from utils.util import GOLD_DIR, load_networkx_graph, get_num_lines


YEARS = list(range(2001, 2018))
# LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
LANGS = ["en"]

def generate_edge_difference_snap(lang, curr, future, k=1):
    g1 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, curr), 0, 2, "\t")
    g2 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, future), 0, 2, "\t")
    
    with open(f"{GOLD_DIR}/{lang}wiki-{curr}-{future}-{k}-gold.csv", "w") as f:
        for e in tqdm(g2.Edges(), total=g2.GetEdges(), desc="Edge Difference"):
            src = e.GetSrcNId()
            dst = e.GetDstNId()
            if g1.IsNode(src) and g1.IsNode(dst) and not g1.IsEdge(src,dst):
                f.write(f"{src}\t{dst}\n")

def generate_gold_edges_networkx(lang, curr, future, k=1):
    g1 = load_networkx_graph(generate_file_name(lang, curr))
    g2_filename = generate_file_name(lang, future)
    edges = g1.edges()
    nodes = g1.nodes()
    degrees = g1.degree
    first = True
    with open(g2_filename) as f:
        with open(f"{GOLD_DIR}/{lang}wiki-{curr}-{future}-{k}-gold.csv", "w") as outfile:
            for line in tqdm(f, total=get_num_lines(g2_filename)):
                if first: 
                    first = False
                    continue
                line = line.strip().split("\t")
                src, dst = int(line[0]), int(line[2])
                if src in nodes and dst in nodes and degrees[src] >= k and degrees[dst] >= k and (src, dst) not in edges: 
                    outfile.write(f"{src}\t{dst}\n")

def main():
    if not os.path.exists(GOLD_DIR):
        os.makedirs(GOLD_DIR)
    for lang in LANGS:
        for year in YEARS:
            if year != 2018: 
                print(lang, year, year+1)
                generate_gold_edges_networkx(lang, year, year+1, k=3)
                

if __name__ == "__main__":
    main()