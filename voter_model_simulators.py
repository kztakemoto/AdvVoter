import numpy as np
import scipy.sparse as sp

def clean_voter_model(adj, x_init, t_max, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # get parameters
    nb_samples = len(x_init)
    nb_nodes = adj.shape[0]
    weighted_degree = np.array(np.sum(adj, axis=1)).reshape(-1)

    result = []
    for n in range(nb_samples):
        # opnion initialization
        x = x_init[n].copy()
        # opnion dynamics
        for t in range(t_max):
            prob = (1.0 - x * adj.dot(x) / weighted_degree) / 2.0
            flip_bool = np.random.rand(nb_nodes) < prob
            x[flip_bool] = -x[flip_bool]
            
        result.append(len(x[x == 1]) / nb_nodes)
    
    return result

def adversarial_voter_model(adj, x_init, x_target, epsilon, t_max, transition_time, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # get parameters
    nb_samples = len(x_init)
    nb_nodes = adj.shape[0]
    weighted_degree = np.array(np.sum(adj, axis=1)).reshape(-1)
    adj_nonzero_rows, adj_nonzero_cols = adj.nonzero()

    result = []
    for n in range(nb_samples):
        # opnion initialization
        x = x_init[n].copy()

        # opnion dynamics before adversarial attacks
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
            weighted_degree = np.array(np.sum(adj_adv, axis=1)).reshape(-1)
            prob = (1.0 - x * adj_adv.dot(x) / weighted_degree) / 2.0
            flip_bool = np.random.rand(nb_nodes) < prob
            x[flip_bool] = -x[flip_bool]
            
        result.append(len(x[x == 1]) / nb_nodes)

    return result

def random_attacked_voter_model(adj, x_init, epsilon, t_max, transition_time, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # get parameters
    nb_samples = len(x_init)
    nb_nodes = adj.shape[0]
    weighted_degree = np.array(np.sum(adj, axis=1)).reshape(-1)
    adj_nonzero_rows, adj_nonzero_cols = adj.nonzero()

    result = []
    for n in range(nb_samples):
        # opnion initialization
        x = x_init[n].copy()

        # opnion dynamics before random attacks
        for t in range(transition_time):
            prob = (1.0 - x * adj.dot(x) / weighted_degree) / 2.0
            flip_bool = np.random.rand(nb_nodes) < prob
            x[flip_bool] = -x[flip_bool]

        # opnion dynamics after random attacks
        for t in range(transition_time, t_max):
            # random attack
            gradient = sp.csc_matrix((np.random.choice([-epsilon, epsilon], len(adj_nonzero_rows)), (adj_nonzero_rows, adj_nonzero_cols)), dtype='float64')
            adj_adv = adj + gradient

            sp.csr_matrix.setdiag(adj_adv, 1)
            weighted_degree = np.array(np.sum(adj_adv, axis=1)).reshape(-1)
            prob = (1.0 - x * adj_adv.dot(x) / weighted_degree) / 2.0
            flip_bool = np.random.rand(nb_nodes) < prob
            x[flip_bool] = -x[flip_bool]
        
        result.append(len(x[x == 1]) / nb_nodes)
    
    return result

