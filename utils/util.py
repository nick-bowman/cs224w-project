def countExistingAndRemovedEdges(graph1, graph2):
    existing = 0
    removed = 0
    for e in graph1.Edges():
        if graph2.IsEdge(e.GetSrcNId(), e.GetDstNId()):
            existing += 1
        else:
            removed +=1
    return (existing, removed)
