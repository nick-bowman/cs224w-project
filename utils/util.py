
# Graph2 is the more recent graph
def countExistingAndRemovedEdges(graph1, graph2):
    existing = 0
    removed = 0
    for e in graph1.Edges():
        if graph2.IsEdge(e.GetSrcNId(), e.GetDstNId()):
            existing += 1
        else:
            removed +=1
    return (existing, removed)

# Graph2 is the more recent graph
def countNewEdges(graph1, graph2):
    added = 0
    for e in graph2.Edges():
        if graph1.IsEdge(e.GetSrcNId(), e.GetDstNId()):
            added += 1
    return added
