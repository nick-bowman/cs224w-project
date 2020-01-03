import snap
import os
from tqdm import tqdm
from utils.util import generate_file_name
import cugraph
from utils.util import load_cugraph, GOLD_DIR, get_num_lines


YEARS = list(range(2014, 2019))
# LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
LANGS = ["en"]

def generate_edge_difference_snap(lang, curr, future):
    g1 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, curr), 0, 2, "\t")
    g2 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, future), 0, 2, "\t")
    
    with open(f"data_diffs/{lang}wiki-{curr}-{future}-diff.csv", "w") as f:
        for e in tqdm(g2.Edges(), total=g2.GetEdges(), desc="Edge Difference"):
            src = e.GetSrcNId()
            dst = e.GetDstNId()
            if g1.IsNode(src) and g1.IsNode(dst) and not g1.IsEdge(src,dst):
                # gold_edges.append((src,dst))
                f.write(f"{src}\t{dst}\n")

def generate_gold_edges_cugraph(lang, curr, future, k=1):
    g1, g1_renumbering = load_cugraph(lang, curr)
    # g2, g2_renumbering = load_cugraph(lang, future)
    g2_filename = generate_file_name(lang, future)
    print(g1.number_of_edges())
    print(g1.number_of_nodes())
#     print(g2.number_of_edges())
#     print(g2.number_of_nodes())
    with open(g2_filename) as f:
        for line in tqdm(f, total=get_num_lines(g2_filename)):
            line = line.strip().split("\t")
            src, dst = line[0], line[2]
    


def main():
    if not os.path.exists(GOLD_DIR):
        os.makedirs(GOLD_DIR)
    for lang in LANGS:
        for year in YEARS:
            if year != 2018: 
                print(lang, year, year+1)
                generate_gold_edges_cugraph(lang, year, year+1, k=3)
                

if __name__ == "__main__":
    main()