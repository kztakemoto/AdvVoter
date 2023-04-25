import argparse
import numpy as np
import scipy.sparse as sp
from scipy import stats
import networkx as nx
import pandas as pd
import igraph
import random

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

#### Parameters #############
parser = argparse.ArgumentParser(description='Run voter models on real-world networks')
parser.add_argument('--netid', type=str, default='facebook_combined', help='network identifier')
parser.add_argument('--tmax', type=int, default=1000, help='maximum number of iterations in voter models')
parser.add_argument('--ttrans', type=int, default=0, help='transition time (i.e., time when adversarial attacks start)')
parser.add_argument('--nb_xinit', type=int, default=300, help='number of initial states for a network')
parser.add_argument('--rhoinit', type=float, default=0.8, help='initial ratio of nodes with the positive opinion')
parser.add_argument('--dir', type=str, default='./results', help='path to the directory where results will be output')
args = parser.parse_args()

# voter model parameters
global initial_positive_opinion_ratio
initial_positive_opinion_ratio = args.rhoinit
global t_max
t_max = args.tmax
global nb_xinit
nb_xinit = args.nb_xinit
global transition_time
transition_time = args.ttrans

# network and its adjacency matrix
# load edge list
edgelist = pd.read_csv('./network_data/{}.txt'.format(args.netid), delimiter=' ', header=None)
edgelist = edgelist[[0,1]]
edgelist = edgelist.rename(columns={0: 'source', 1: 'target'})
edgelist = edgelist.drop_duplicates()
# network object
g = nx.from_pandas_edgelist(edgelist, source='source', target='target')
gcc = sorted(nx.connected_components(g), key=len, reverse=True)
global g_gcc
g_gcc = g.subgraph(gcc[0])

# network model parameters
global nb_nodes
nb_nodes = g_gcc.number_of_nodes()

# opnion initialization
np.random.seed(123)
global x_init
for n in range(nb_xinit):
    x = np.repeat(-1, nb_nodes)
    idx = np.random.choice(list(range(nb_nodes)), int(nb_nodes * initial_positive_opinion_ratio), replace=False)
    x[idx] = 1
    if n == 0:
        x_init = x
    else:
        x_init = np.vstack((x_init, x))

# adversarial attacks
global x_target
x_target = np.repeat(-1, nb_nodes) # target state

def clean_voter_model(seed=0):
    adj = nx.adjacency_matrix(g_gcc, dtype='float64')
    g = igraph.Graph.Adjacency(adj.todense(), mode='undirected')
    deg = g.degree()
    random.seed(seed)
    g = igraph.GraphBase.Degree_Sequence(deg, method='vl')
    adj = sp.csr_matrix(igraph.GraphBase.get_adjacency(g), dtype='float64')
    sp.csr_matrix.setdiag(adj, 1)
    weighted_degree = np.array(np.sum(adj, axis=1)).reshape(-1)
    # opnion initialization
    x = x_init[seed].copy()
    # opnion dynamics
    np.random.seed(seed)
    for t in range(t_max):
        prob = (1.0 - x * adj.dot(x) / weighted_degree) / 2.0
        flip_bool = np.random.rand(nb_nodes) < prob
        x[flip_bool] = -x[flip_bool]
    
    return len(x[x == 1]) / nb_nodes

def adversarial_voter_model(epsilon=0.01, seed=0):
    adj = nx.adjacency_matrix(g_gcc, dtype='float64')
    g = igraph.Graph.Adjacency(adj.todense(), mode='undirected')
    deg = g.degree()
    random.seed(seed)
    g = igraph.GraphBase.Degree_Sequence(deg, method='vl')
    adj = sp.csr_matrix(igraph.GraphBase.get_adjacency(g), dtype='float64')
    sp.csr_matrix.setdiag(adj, 1)
    weighted_degree = np.array(np.sum(adj, axis=1)).reshape(-1)
    adj_nonzero_rows, adj_nonzero_cols = adj.nonzero()
    # opnion initialization
    x = x_init[seed].copy()
    # opnion dynamics before adversarial attacks
    np.random.seed(seed)
    for t in range(transition_time):
        prob = (1.0 - x * adj.dot(x) / weighted_degree) / 2.0
        flip_bool = np.random.rand(nb_nodes) < prob
        x[flip_bool] = -x[flip_bool]

    # opnion dynamics after adversarial attacks
    for t in range(transition_time, t_max):
        # attack using the signed version of gradient
        gradient = sp.csc_matrix((epsilon * x_target[adj_nonzero_rows] * x[adj_nonzero_cols], (adj_nonzero_rows, adj_nonzero_cols)), dtype='float64')
        adj_adv = adj + gradient
        
        sp.csr_matrix.setdiag(adj_adv, 1)
        weighted_degree_mod = np.array(np.sum(adj_adv, axis=1)).reshape(-1)
        prob = (1.0 - x * adj_adv.dot(x) / weighted_degree_mod) / 2.0
        flip_bool = np.random.rand(nb_nodes) < prob
        x[flip_bool] = -x[flip_bool]
    
    return len(x[x == 1]) / nb_nodes

def random_attacked_voter_model(epsilon=0.01, seed=0):
    adj = nx.adjacency_matrix(g_gcc, dtype='float64')
    g = igraph.Graph.Adjacency(adj.todense(), mode='undirected')
    deg = g.degree()
    random.seed(seed)
    g = igraph.GraphBase.Degree_Sequence(deg, method='vl')
    adj = sp.csr_matrix(igraph.GraphBase.get_adjacency(g), dtype='float64')
    sp.csr_matrix.setdiag(adj, 1)
    weighted_degree = np.array(np.sum(adj, axis=1)).reshape(-1)
    adj_nonzero_rows, adj_nonzero_cols = adj.nonzero()
    # opnion initialization
    x = x_init[seed].copy()
    # opnion dynamics before random attacks
    np.random.seed(seed)
    for t in range(transition_time):
        prob = (1.0 - x * adj.dot(x) / weighted_degree) / 2.0
        flip_bool = np.random.rand(nb_nodes) < prob
        x[flip_bool] = -x[flip_bool]

    # opnion dynamics after random attacks
    for t in range(transition_time, t_max):
        # attack using the signed version of gradient
        gradient = sp.csc_matrix((np.random.choice([-epsilon, epsilon], len(adj_nonzero_rows)), (adj_nonzero_rows, adj_nonzero_cols)), dtype='float64')
        adj_adv = adj + gradient

        sp.csr_matrix.setdiag(adj_adv, 1)
        weighted_degree_mod = np.array(np.sum(adj_adv, axis=1)).reshape(-1)
        prob = (1.0 - x * adj_adv.dot(x) / weighted_degree_mod) / 2.0
        flip_bool = np.random.rand(nb_nodes) < prob
        x[flip_bool] = -x[flip_bool]
    
    return len(x[x == 1]) / nb_nodes

def wrap_adversarial_voter_model(args):
    return adversarial_voter_model(*args)

def wrap_random_attacked_voter_model(args):
    return random_attacked_voter_model(*args)

if __name__ == "__main__":
    seed_list = list(range(nb_xinit))
    p = Pool(processes=cpu_count())

    # for eps=0
    result_adversarial_nets = np.array(list(tqdm(p.imap(clean_voter_model, seed_list), total=len(seed_list))))
    result_random_nets = result_adversarial_nets.copy()

    epsilon_set = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]
    for eps in epsilon_set:
        job_args = [(eps, seed) for seed in seed_list]

        result_adversarial = np.array(list(tqdm(p.imap(wrap_adversarial_voter_model, job_args), total=len(seed_list))))
        result_random = np.array(list(tqdm(p.imap(wrap_random_attacked_voter_model, job_args), total=len(seed_list))))
        
        result_adversarial_nets = np.vstack((result_adversarial_nets, result_adversarial))
        result_random_nets = np.vstack((result_random_nets, result_random))

    output_filename_base = "{}/{}_randomized_tmax{}_rhoinit{}_ttrans{}_nbxinit{}".format(
        args.dir,
        args.netid,
        t_max,
        initial_positive_opinion_ratio,
        transition_time,
        nb_xinit,
    )

    filename = output_filename_base + "_adversarial.npy"
    np.save(filename, result_adversarial_nets)

    filename = output_filename_base + "_random.npy"
    np.save(filename, result_random_nets)
    
