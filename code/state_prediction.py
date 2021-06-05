import numpy as np
from tqdm import tqdm

def predict_states(X, clf, n_stim=8, shifts=(-5, 6), remove=(14, )):

    """

    Args:
        X: MEG data
        clf: Classifier trained on localiser data
        n_stim: Number of states
        shifts: Number of adjacent states to use. Tuple of (previous states, subsequent states)

    Returns:
        Numpy array of state activation probabilities

    """

    n_tp = X.shape[2]  # Number of timepoints
    state_probabilities = np.zeros((X.shape[0], n_tp + shifts[0] - shifts[1] + 1, n_stim))
    for i in tqdm(range(X.shape[0])):  # predict on every trial
        trial_X = np.expand_dims(X[i, ...], 0)

        # exclude first and last few timepoints as we don't have any adjacent data to add as features
        timepoints = []
        for j in range(n_tp)[0 - shifts[0]:n_tp - shifts[1] + 1]:
            tp_X = trial_X[..., j + shifts[0]:j + shifts[1]]
            timepoints.append(tp_X)
        timepoints = np.stack(timepoints).squeeze()
        if timepoints.ndim < 3:
            timepoints = timepoints[..., np.newaxis]
        pred = clf.predict_proba(timepoints)
        state_probabilities[i, :, :] = pred[..., [i for i in range(pred.shape[1]) if i not in remove]]

    return state_probabilities[..., :n_stim]


