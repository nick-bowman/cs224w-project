import snap
import os
from tqdm import tqdm
from utils.util import generate_file_name

YEARS = list(range(2010, 2019))
# LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]
LANGS = ["en"]

def generate_edge_difference(lang, curr, future):
    g1 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, curr), 0, 2, "\t")
    g2 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, future), 0, 2, "\t")
    
    with open(f"data_diffs/{lang}wiki-{curr}-{future}-diff.csv", "w") as f:
        for e in tqdm(g2.Edges(), total=g2.GetEdges(), desc="Edge Difference"):
            src = e.GetSrcNId()
            dst = e.GetDstNId()
            if g1.IsNode(src) and g1.IsNode(dst) and not g1.IsEdge(src,dst):
                # gold_edges.append((src,dst))
                f.write(f"{src}\t{dst}\n")

def main():
    if not os.path.exists("data_diffs"):
        os.makedirs("data_diffs")
    for lang in reversed(LANGS):
        for year in YEARS:
            if year != 2018: 
                print(lang, year, year+1)
                generate_edge_difference(lang, year, year+1)

if __name__ == "__main__":
    main()