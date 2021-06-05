import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
    
def plot_matrices(matrices):
    
    n_rows = int(np.ceil(len(matrices) / 4))

    fig, ax = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))

    [axi.set_axis_off() for axi in ax.ravel()]

    if ax.ndim == 1:
        ax = np.expand_dims(ax, axis=0)

    pos = [(-4.5, 0), (-1.5, 0), (1.5, 0), (4.5, 0),
        (-4.5, -2), (-1.5, -2), (1.5, -2), (4.5, -2),
        (-4.5, -4), (-1.5, -4), (1.5, -4), (4.5, -4),
        (-3, -6), (3, -6)]

    for n, matrix in enumerate(matrices):
        G = nx.DiGraph(matrix)
        nx.draw(G, pos, ax=ax[np.floor(n / 4).astype(int), n % 4], node_color='white', with_labels=True, font_color='#5b5b5b', edge_color="#5b5b5b", width=2, edgecolors='#5b5b5b', 
                edgewidths=2)
        ax[np.floor(n / 4).astype(int), n % 4].set_title("Matrix {0}".format(n + 1), fontweight='light', color='#5b5b5b')
    plt.tight_layout()

class StateReactivation(object):

    def __init__(self, reactivation_array):
        
        self.reactivation_array = reactivation_array

    def get_sequenceness(self, max_lag, matrices):

        sequenceness = get_sequenceness(self.reactivation_array[..., :matrices[0].shape[0]], max_lag, matrices)

        return sequenceness

    def get_windowed_sequenceness(self, max_lag, matrices, alpha=True, constant=False, width=20,
                                    method='glm', set_zero=True, scale=True, covariates=()):

        windowed_seq = []
        
        for window in tqdm(range(self.reactivation_array.shape[1] - width)):

            sequenceness = get_sequenceness(self.reactivation_array[:, window:window+width, :], max_lag, matrices)
            
            windowed_seq.append(sequenceness)

        windowed_seq_dict = {}

        for k in sequenceness.keys():
            seq_list = []
            for window in windowed_seq:
                seq_list.append(window[k])
            windowed_seq_dict[k] = np.stack(seq_list)

        self.windowed_seq = windowed_seq

        return windowed_seq_dict
    

@njit
def get_sequenceness(reactivation_array:np.ndarray, max_lag:int, matrices:list):

    temp_reactivation_array = reactivation_array.copy()

    # Create empty arrays for output
    forwards, backwards, difference = [np.empty((temp_reactivation_array.shape[0], max_lag, len(matrices))) for i in range (3)]

    # Get sequenceness
    forwards, backwards, difference = run_cross_correlation(temp_reactivation_array, forwards, backwards, difference, np.stack(matrices), max_lag)

    # Put the results into a dictionary
    sequenceness = dict(forwards=forwards, backwards=backwards, difference=difference)

    return sequenceness


@njit
def run_cross_correlation(temp_reactivation_array, forwards, backwards, difference, matrices, max_lag):
    
    for trial in range(temp_reactivation_array.shape[0]):
        # Loop over matrices and trials
        for n, matrix in enumerate(matrices):
            forwards[trial, :, n], backwards[trial, :, n], \
            difference[trial, :, n], = cross_correlation(temp_reactivation_array[trial, :, :matrices[0].shape[0]], 
                                                         matrix, maxlag=max_lag)  

    return forwards, backwards, difference

@njit
def numba_roll(X, shift):
    # Rolls along 1st axis
    new_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        new_X[:, i] = np.roll(X[:, i], shift)
    return new_X
    

@njit
def cross_correlation(X_data, transition_matrix, maxlag=40, minlag=0):
    """
    Computes sequenceness by cross-correlation
    """
    X_dataf = X_data @ transition_matrix
    X_datar = X_data @ transition_matrix.T

    ff = np.zeros(maxlag - minlag)
    fb = np.zeros(maxlag - minlag)
    diffs = np.zeros(maxlag - minlag)

    for lag in range(minlag, maxlag):

        r = np.corrcoef(X_data[lag:, :].T, numba_roll(X_dataf, lag)[lag:, :].T)
        r = np.diag(r, k=transition_matrix.shape[0])
        forward_mean_corr = np.nanmean(r)

        r = np.corrcoef(X_data[lag:, :].T, numba_roll(X_datar, lag)[lag:, :].T)
        r = np.diag(r, k=transition_matrix.shape[0])
        backward_mean_corr = np.nanmean(r)

        diffs[lag - minlag] = forward_mean_corr - backward_mean_corr
        ff[lag - minlag] = forward_mean_corr
        fb[lag - minlag] = backward_mean_corr

    return ff, fb, diffs

