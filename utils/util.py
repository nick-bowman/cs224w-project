
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


def findAddedEdgesInIntersection(graph1, graph2):
    edges = set()
    for edge in graph2.Edges():
        start = edge.GetSrcNId()
        end = edge.GetDstNId()
        if graph1.IsNode(start) and graph1.IsNode(end):
            if not graph1.IsEdge(start, end):
                edges.add((start, end))
    return edges

def computeMetric(graph1, graph2, predictedEdges):
    trueAddedEdges = findAddedEdgesInIntersection(graph1, graph2)

    falsePositive = len(predictedEdges - trueAddedEdges)
    falseNegative = len(trueAddedEdges - predictedEdges)
    truePositive = len(trueAddedEdges.intersection(predictedEdges))

    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)
