import argparse
import numpy as np
import scipy.sparse as sp
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt

from voter_model_simulators import (
    clean_voter_model,
    adversarial_voter_model,
    random_attacked_voter_model,
    clean_voter_model_consensus,
    adversarial_voter_model_consensus,
    random_attacked_voter_model_consensus,
)

from utils import (
    GKK_model,
    generate_correlated_net_Sneppen,
    load_network_data,
)

#### Parameters #############
parser = argparse.ArgumentParser(description='Run the voter model dynamics in complex networks')
parser.add_argument('--network', type=str, default='ER', help='network types: Erdos-Renyi (ER), Watts-Strogatz (WS), Barabasi-Albert (BA) networks, Goh-Kahng-Kim (GKK) model, Holme-Kim (HK) model, and real-world networks')
parser.add_argument('--N', type=int, default=400, help='number of nodes in model networks')
parser.add_argument('--kave', type=float, default=6, help='average degree in model networks')
parser.add_argument('--gamma', type=float, default=2.2, help='degree exponent for GKK model')
parser.add_argument('--phk', type=float, default=0.1, help='Probability of adding a triangle after adding a random edge for HK model')
parser.add_argument('--correlation', type=str, default='uncorr', help='degree correlation. assort, disassort')
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
elif args.network  == 'ER':
    # Erdos-Renyi model
    g = nx.gnm_random_graph(nb_nodes, nb_edges, directed=False, seed=123)
elif args.network == 'WS':
    # Watts-Strogatz model
    pws = 0.05
    g = nx.watts_strogatz_graph(nb_nodes, int(average_degree), pws, seed=123)
elif args.network == 'GKK':
    # Goh-Kahng-Kim model
    g = GKK_model(nb_nodes, nb_edges, args.gamma)
elif args.network == 'HK':
    # Holme-Kim model
    g = nx.powerlaw_cluster_graph(nb_nodes, int(average_degree / 2), args.phk, seed=123)
elif args.network in ['facebook_combined', 'soc-advogato', 'soc-anybeat', 'soc-hamsterster']:
    # real-world social network
    g = load_network_data(args.network)
    nb_nodes = g.number_of_nodes() # update `nb_nodes`
else:
    raise ValueError("invalid network")

# generate correlated networks
if args.correlation != 'uncorr':
    g = generate_correlated_net_Sneppen(g, correlation=args.correlation, seed=123)

adj = nx.adjacency_matrix(g, dtype='float64')
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
if t_max >= 0:
    rho_clean = clean_voter_model(adj, x_init, t_max, seed=123)
    rho_adversarial = adversarial_voter_model(adj, x_init, x_target, epsilon, t_max, transition_time, seed=123)
    rho_random = random_attacked_voter_model(adj, x_init, epsilon, t_max, transition_time, seed=123)
else:
    rho_clean = clean_voter_model_consensus(adj, x_init, seed=123)
    rho_adversarial = adversarial_voter_model_consensus(adj, x_init, x_target, epsilon, transition_time, seed=123)
    rho_random = random_attacked_voter_model_consensus(adj, x_init, epsilon, transition_time, seed=123)

# Average rho values
print("Average rho values:")
print("(clean) {:.3f}".format(np.mean(rho_clean)))
print("(adversarial) {:.3f}".format(np.mean(rho_adversarial)))
print("(random) {:.3f}".format(np.mean(rho_random)))

# KS distance
print("\nKS distance between rho distributions and its p-value")
ks = stats.ks_2samp(rho_clean, rho_adversarial)
print("clean vs adversarial:", ks[0], "p =",ks[1])
ks = stats.ks_2samp(rho_clean, rho_random)
print("clean vs random:", ks[0], "p =",ks[1])

# plot histgram
weights = np.ones_like(rho_random) / len(rho_random)

plt.ylim(0,1)
plt.hist(rho_random, alpha = 0.5, weights=weights, label='random')
plt.hist(rho_adversarial, alpha = 0.5, weights=weights, label='adversarial')
plt.hist(rho_clean, alpha = 0.5, weights=weights, label='no perturbation')
plt.ylabel(r'$P(\rho)$')
plt.xlabel(r'$\rho$')
plt.legend(loc='upper center')
plt.savefig('rho_distribution.png')
