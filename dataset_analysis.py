import snap
import os
from tqdm import tqdm
from utils.util import generate_file_name

YEARS = list(range(2001, 2019))
LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "ru", "sv"]

def generate_edge_difference(lang, curr, future):
    g1 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, curr), 0, 2, "\t")
    g2 = snap.LoadEdgeList(snap.PNGraph, generate_file_name(lang, future), 0, 2, "\t")
    
    gold_edges = []
    for e in tqdm(g2.Edges(), total=g2.GetEdges(), desc="Edge Difference"):
        src = e.GetSrcNId()
        dst = e.GetDstNId()
        if g1.IsNode(src) and g1.IsNode(dst) and not g1.IsEdge(src,dst):
            gold_edges.append((src,dst))
#     node_set1 = set()
#     node_set2 = set()
    
#     for n in tqdm(g1.Nodes(), total=g1.GetNodes(), desc="Node Set 1"):
#         node_set1.add(n.GetId())
#     for n in tqdm(g2.Nodes(), total=g2.GetNodes(), desc="Node Set 2"):
#         node_set2.add(n.GetId())
    
#     node_intersection = node_set1.intersection(node_set2)
# #     print(node_intersection)
    
#     node_v = snap.TIntV()
#     for node in tqdm(node_intersection, total=len(node_intersection), desc="Node Intersection"):
#         node_v.Add(node)
    
#     sub_g1 = snap.GetSubGraph(g1, node_v)
#     sub_g2 = snap.GetSubGraph(g2, node_v)
    
#     gold_edges = []
#     for e in tqdm(sub_g2.Edges(), total=sub_g2.GetEdges(), desc="Edge Intersection"):
#         if not sub_g1.IsEdge(e.GetSrcNId(), e.GetDstNId()):
#             gold_edges.append((e.GetSrcNId(), e.GetDstNId()))
    
    with open(f"data_diffs/{lang}wiki-{curr}-{future}-diff.csv", "w") as f:
        for start, end in tqdm(gold_edges, total=len(gold_edges), desc="Writing to File"):
            f.write(f"{start}\t{end}\n")
#     print(gold_edges)

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