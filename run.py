import argparse
import numpy as np
import scipy.sparse as sp
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

from voter_model_simulators import (
    clean_voter_model,
    adversarial_voter_model,
    random_attacked_voter_model,
)

#### Parameters #############
parser = argparse.ArgumentParser(description='Run the voter model dynamics in complex networks')
parser.add_argument('--network', type=str, default='ER', help='network types: Erdos-Renyi (ER), Watts-Strogatz (WS), Barabasi-Albert (BA) networks, and real-world networks')
parser.add_argument('--N', type=int, default=400, help='number of nodes in model networks')
parser.add_argument('--kave', type=float, default=6, help='average degree in model networks')
parser.add_argument('--tmax', type=int, default=400, help='maximum number of iterations in the voter model')
parser.add_argument('--ttrans', type=int, default=0, help='transition time (i.e., time when adversarial attacks start)')
parser.add_argument('--eps', type=float, default=0.01, help='epsilon or perturbation strength')
parser.add_argument('--nb_xinit', type=int, default=100, help='number of initial states per network')
parser.add_argument('--rhoinit', type=float, default=0.8, help='initial ratio of nodes with the opinion +1')
args = parser.parse_args()

# network model parameters
nb_nodes = args.N
average_degree = args.kave
nb_edges = int(average_degree * nb_nodes / 2)

# voter model parameters
t_max = args.tmax
nb_xinit = args.nb_xinit
transition_time = args.ttrans

# generate a network
if args.network == 'BA':
    # Barabasi-Albert model
    g = nx.barabasi_albert_graph(nb_nodes, int(average_degree / 2), seed=123)
    adj = nx.adjacency_matrix(g, dtype='float64')
elif args.network  == 'ER':
    # Erdos-Renyi model
    g = nx.gnm_random_graph(nb_nodes, nb_edges, directed=False, seed=123)
    adj = nx.adjacency_matrix(g, dtype='float64')
elif args.network == 'WS':
    # Watts-Strogatz model
    pws = 0.05
    g = nx.watts_strogatz_graph(nb_nodes, int(average_degree), pws, seed=123)
    adj = nx.adjacency_matrix(g, dtype='float64')
elif args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
    # real-world social network
    # load edge list
    edgelist = pd.read_csv('./network_data/{}.txt'.format(args.network), delimiter=' ', header=None)
    edgelist = edgelist[[0,1]]
    edgelist = edgelist.rename(columns={0: 'source', 1: 'target'})
    edgelist = edgelist.drop_duplicates()
    # network object
    g = nx.from_pandas_edgelist(edgelist, source='source', target='target')
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(gcc[0])
    adj = nx.adjacency_matrix(g, dtype='float64')
    nb_nodes = g.number_of_nodes() # update `nb_nodes`
else:
    raise ValueError("invalid network")

sp.csr_matrix.setdiag(adj, 1)
# adj = adj.T # for directed networks

# opnion initialization
np.random.seed(123)
for n in range(nb_xinit):
    x = np.repeat(-1, nb_nodes)
    idx = np.random.choice(list(range(nb_nodes)), int(nb_nodes * args.rhoinit), replace=False)
    x[idx] = 1
    if n == 0:
        x_init = x
    else:
        x_init = np.vstack((x_init, x))

# adversarial attacks
x_target = np.repeat(-1, nb_nodes) # target state
epsilon = args.eps # attack strength

# simulate the voter model dynamics
result_clean = clean_voter_model(adj, x_init, t_max, seed=123)
result_adversarial = adversarial_voter_model(adj, x_init, x_target, epsilon, t_max, transition_time, seed=123)
result_random = random_attacked_voter_model(adj, x_init, epsilon, t_max, transition_time, seed=123)

# Average rho values
print("Average rho values")
print("(clean):", np.mean(result_clean))
print("(adversarial):", np.mean(result_adversarial))
print("(random):", np.mean(result_random))

# KS distance
print("\nKS distance between rho distributions and its p-value")
ks = stats.ks_2samp(result_clean, result_adversarial)
print("clean vs adversarial:", ks[0], "p =",ks[1])
ks = stats.ks_2samp(result_clean, result_random)
print("clean vs random:", ks[0], "p =",ks[1])

# plot histgram
weights = np.ones_like(result_random) / len(result_random)

plt.ylim(0,1)
plt.hist(result_random, alpha = 0.5, weights=weights, label='random')
plt.hist(result_adversarial, alpha = 0.5, weights=weights, label='adversarial')
plt.hist(result_clean, alpha = 0.5, weights=weights, label='no perturbation')
plt.ylabel(r'$P(\rho)$')
plt.xlabel(r'$\rho$')
plt.legend(loc='upper center')
plt.show()
