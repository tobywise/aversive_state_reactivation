import networkx as nx
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
from sequenceness import plot_matrices
import matplotlib.pyplot as plt

def select_timepoints(X, idx=33, embedding=10):
    idx = int(idx)
    shifts = (int(0 - embedding / 2), int(embedding / 2 + 1))
    return X[..., idx + shifts[0]:idx + shifts[1]] 


def add_features(X):
    return X.reshape(X.shape[0], -1)

def select_path(transition_matrix, start_state, outcome_state):

    G = nx.DiGraph(transition_matrix)

    path = nx.shortest_path(G, start_state, outcome_state)

    m = transition_matrix.copy()

    for i in range(len(m)):
        if i not in path:
            m[i, :] = 0
            m[:, i] = 0

    return m

def path_to_matrix(path, nstates):

    matrix = np.zeros((nstates, nstates))

    for i in range(len(path) - 1):
        matrix[path[i], path[i+1]] = 1

    return matrix


