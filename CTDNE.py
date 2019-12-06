import numpy as np
import networkx as nx
import random


class Graph():
    def __init__(self, nx_G, is_directed, p = 1, q = 1):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def CTDNE_walk(self, start_edge, max_length):
        """
        Simulate a temporal random walk starting from the initial temporal
        edge.
        
        Parameters
        ----------
        start_edge : tuple(start_node, end_node, dict)
            The initial temporal edge to begin the random walk. The dict stores
            a key "timestamp" that maps to the timestamp of this edge.
        max_length : int
            Maximum length for the temporal random walk.
            
        Returns
        -------
        walk : list[int]
            List of nodes encountered on the random walk.
        """
        G = self.G
        
        """
        For now ignore this alias sampling stuff, but may need later if we want
        to sample non-uniformly.
        
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        """
        
        walk = [start_edge[0], start_edge[1]]
        curr_node = start_edge[1]
        curr_time = start_edge[2]["timestamp"]
        while len(walk) < max_length:
            out_edges = G.out_edges(curr_node, data = True)
            valid_edges = [edge for edge in out_edges if edge[2]["timestamp"] > curr_time]
            if len(valid_edges) == 0:
                break
            edge_index = np.random.choice(len(valid_edges)) # NOTE: sampling uniformly
            random_edge = valid_edges[edge_index]
            walk.append(random_edge[1])
            curr_node = random_edge[1]
            curr_time = random_edge[2]["timestamp"]
        return walk

    def simulate_walks(self, num_walks, max_length, min_length = 0):
        """
        Repeatedly simulate random walks from each node.
        
        Parameters
        ----------
        num_walks : int
            Number of random walks to simulate.
        max_length : int
            Maximum length for random walk.
        min_length : int, optional
            Minimum length for random walk, defaults to 0.

        Returns
        -------
        walks : list[list[int]]
            List of random walks, where each walk is a list of node IDs encountered
            on the walk.
        """
        G = self.G
        walks = []
        edges = list(G.edges(data = True))
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            edge_index = np.random.choice(len(edges)) # NOTE: sampling uniformly
            start_edge = edges[edge_index]
            random_walk = self.CTDNE_walk(start_edge, max_length)
            if len(random_walk) > min_length:
                walks.append(random_walk)
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]