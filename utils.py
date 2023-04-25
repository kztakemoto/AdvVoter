import numpy as np
import pandas as pd
import networkx as nx
import igraph
import random

def generate_correlated_net_Sneppen(g, correlation='disassort', seed=None):
    if seed is not None:
        random.seed(seed)

    g0 = g.copy()
    degree = g0.degree()
    edgelist = [e for e in g0.edges()]
    err = 0
    while len(edgelist) > 1 and err < 5000:
        idxs = random.sample(range(len(edgelist)), 2)
        e1 = edgelist[idxs[0]]
        e2 = edgelist[idxs[1]]
        nodepairs = np.array([e1[0], e1[1], e2[0], e2[1]])
        degree_pairs = [degree[v] for v in nodepairs]
        nodepairs_new = nodepairs[np.argsort(degree_pairs)]
        if correlation == 'assort':
            e1_new = (nodepairs_new[0], nodepairs_new[1])
            e2_new = (nodepairs_new[2], nodepairs_new[3])
        elif correlation == 'disassort':
            e1_new = (nodepairs_new[0], nodepairs_new[3])
            e2_new = (nodepairs_new[1], nodepairs_new[2])
        else:
            raise ValueError("invalid correlation type")

        if len(np.unique(nodepairs)) == 4 and e1_new not in g0.edges() and e2_new not in g0.edges():
            g0.remove_edge(*e1)
            g0.remove_edge(*e2)
            g0.add_edge(*e1_new)
            g0.add_edge(*e2_new)
            edgelist.remove(e1)
            edgelist.remove(e2)
        else:
            err = err + 1
            
    return g0

def GKK_model(nb_nodes, nb_edges, gamma):
    g = igraph.GraphBase.Static_Power_Law(nb_nodes, nb_edges, gamma, loops=False, multiple=False, finite_size_correction=False)
    node_idx = np.array(list(range(nb_nodes)))
    
    nb_isolated_nodes = nb_nodes
    while nb_isolated_nodes > 0:
        deg = np.array(g.degree())
        isolated_nodes = node_idx[deg==0].tolist()
        random.shuffle(isolated_nodes)
        nb_isolated_nodes = len(isolated_nodes)
        endpints_deleted_edges = random.sample(g.get_edgelist(), nb_isolated_nodes)
        g.delete_edges(endpints_deleted_edges)
        endpints_added_edges = []
        for i, pair in enumerate(endpints_deleted_edges):
            l = list(pair)
            endpints_added_edges.append(tuple((isolated_nodes[i], l[1])))
        g.add_edges(endpints_added_edges)

    # igraph to networkx
    edgelist = g.get_edgelist()
    g = nx.Graph(edgelist)
    
    return g

def load_network_data(network):
    # load edge list
    edgelist = pd.read_csv('./network_data/{}.txt'.format(network), delimiter=' ', header=None)
    edgelist = edgelist[[0,1]]
    edgelist = edgelist.rename(columns={0: 'source', 1: 'target'})
    edgelist = edgelist.drop_duplicates()
    
    # network object
    g = nx.from_pandas_edgelist(edgelist, source='source', target='target')

    # ectract the largest connected component
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(gcc[0])
    
    return g
