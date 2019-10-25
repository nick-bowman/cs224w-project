import snap

class HybridModel():
    def __init__(self, g, node_frac=1.0, l=0.9):
        self.g = g
        self.node_frac = node_frac
        self.l = l

    def generate_scores(self):
        scores = {}
        common_neighbor_scores = {}
        for e in self.g.Edges():
            # common_neighbor_scores[(e.GetSrcNId(), e.GetDstNId())] = snap.GetCmnNbrs(self.g, e.GetSrcNId(), e.GetDstNId())
            n1 = snap.TIntV()
            n2 = snap.TIntV()
            snap.GetNodesAtHop(self.g, e.GetSrcNId(), 1, n1, True)
            snap.GetNodesAtHop(self.g, e.GetDstNId(), 1, n2, True)
            common_neighbor_scores[(e.GetSrcNId(), e.GetDstNId())] = len(set(n1) & set(n2))
        Nodes = snap.TIntFltH()
        Edges = snap.TIntPrFltH()
        snap.GetBetweennessCentr(self.g, Nodes, Edges, self.node_frac, True)
        edge_betweenness_scores = {}
        for e in Edges:
            edge_betweenness_scores[(e.GetVal1(), e.GetVal2())] = Edges[e]
        max_cn = max(common_neighbor_scores.values())
        max_eb = max(edge_betweenness_scores.values())
        print(common_neighbor_scores)
        print(edge_betweenness_scores)
        for e in self.g.Edges():
            src = e.GetSrcNId()
            dst = e.GetDstNId()
            scores[(src, dst)] = self.l * common_neighbor_scores[(src,dst)] / max_cn + (1-self.l) * edge_betweenness_scores[(src,dst)] / max_eb
        return scores
