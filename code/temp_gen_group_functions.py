
from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import tqdm
from mne.stats.cluster_level import _pval_from_histogram, _reshape_clusters
import matplotlib.pyplot as plt

def load_and_filter(fname):
    return gaussian_filter(np.load(fname), 3)

def plot_clusters(array, clusters, cluster_p_values, threshold=0.05, cmap='plasma', total_duration=None, start=None, end=None, figsize=(3.5, 1.3)):

    times = np.arange(array.shape[1]) * 10
    yvals = np.arange(array.shape[0]) * 10

    T_obs_plot = np.nan * np.ones_like(array)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= threshold:
            T_obs_plot[c] = array[c]


    fig, (ax, cax_a, cax_b) = plt.subplots(ncols=3,figsize=figsize, gridspec_kw={"width_ratios":[0.9, 0.03, 0.03]}, dpi=100)
    plt.subplots_adjust(wspace=0.1)

    vmax = array.max()
    vmin = array.min()
    
    im_a = ax.imshow(array, cmap=plt.cm.gray,
               extent=[times[0], times[-1], yvals[0], yvals[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    im_b = ax.imshow(T_obs_plot, cmap=cmap,
               extent=[times[0], times[-1], yvals[0], yvals[-1]],
               aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
    cb_a = plt.colorbar(im_a, cax=cax_a)
    cax_a.set_yticks([])
    cax_a.set_yticklabels([])
    cb_b = plt.colorbar(im_b, label='t', cax=cax_b)
    
    ax.contour(np.kron(~np.isnan(T_obs_plot), np.ones((10, 10))), colors=["k"], linewidths=[.5], corner_mask=False,
                           antialiased=True, levels=[.5], extent=[times[0], times[-1], yvals[0], yvals[-1] + 1])
    ax.set_xlabel('Testing (ms)')
    ax.set_ylabel('Training (ms)')
    yticks = np.arange(0, ax.get_ylim()[1], 200)
    ax.set_yticks(yticks)
    ax.set_yticks(yticks)
    
    if total_duration is not None:
        if start is not None:
            ax.set_xlim(0 - start, total_duration - start)
            ax.axvline(0, linestyle='--', color='#3d3d3d')
            ax.set_xticklabels(list(np.array(ax.get_xticks()).astype(int) + start))
            ylim = ax.get_ylim()[1]
            ax.fill([-start, 0, 0, -start], [0, 0, ylim, ylim], fill=False, hatch=r'\\\\', linewidth=0.0, color='gray', alpha=0.5)
            ax.scatter([-start, 0], [ylim + 85, ylim + 85], clip_on=False, marker=9, s=15, color='#2b2b2b')
            ax.set_ylim(0, ylim)
            ax.text(0 + 300, ylim + 45, 'Rest', color='#2b2b2b', size=8, fontweight='regular')
            ax.text(-start + 300, ylim + 45, 'Outcome', color='#2b2b2b', size=8, fontweight='regular')
        elif end is not None:
            ax.set_xlim(0, total_duration)
            ax.axvline(end, linestyle='--', color='#3d3d3d')
            ylim = ax.get_ylim()[1]
            ax.fill([end, total_duration, total_duration, end], [0, 0, ylim, ylim], fill=False, hatch=r'\\\\', linewidth=0.0, color='gray', alpha=0.5)
            ax.scatter([end, 0], [ylim + 85, ylim + 85], clip_on=False, marker=9, s=15, color='#2b2b2b')
            ax.set_ylim(0, ylim)
            ax.text(0 + 300, ylim + 45, 'Outcome', color='#2b2b2b', size=8, fontweight='regular')
            ax.text(end + 300, ylim + 45, 'Next trial', color='#2b2b2b', size=8, fontweight='regular')
                
    fig.subplots_adjust(top=0.7)
    


def permutation_lm_test(x, y, plot=True, threshold=None, seed=0, regressor_names=('intercept', 'generalisation'), test_regressor='generalisation'):
    """
    This code is adapted from MNE's permutation code.

    """
    betas, stderr, t_val, p_val, mlog10_p_val = _fit_lm(y, x, regressor_names)
    betas = betas[test_regressor]

    from mne.stats.cluster_level import _find_clusters, _check_fun
    
    if threshold == None:
        stat_fun, threshold = _check_fun(t_val[test_regressor], stat_fun=None, threshold=None, tail=0)

    clusters, cluster_stats = _find_clusters(t_val[test_regressor], threshold)
    T_obs = t_val[test_regressor]
    clusters = _reshape_clusters(clusters, y.shape[1:])

    orig = abs(cluster_stats).max()
    
    if len(clusters):
        print("Found {0} clusters".format(len(clusters)))
        # perm_betas = np.zeros((1000, ) + betas.shape)
        max_cluster_sums = np.zeros(1000)

        randomstate = np.random.RandomState(seed=seed)

        for seed_idx in tqdm(range(1000)):
            x_perm = x.copy()
            randomstate.shuffle(x_perm)
            betas, stderr, t_val, p_val, mlog10_p_val = _fit_lm(y, x_perm, regressor_names)
            _, perm_clusters_sums  = _find_clusters(t_val[test_regressor], threshold)
            if len(perm_clusters_sums) > 0:
                max_cluster_sums[seed_idx] = np.max(perm_clusters_sums)

        H0 = list(max_cluster_sums)
        orig = abs(cluster_stats).max()
        H0.insert(0, [orig])

        H0 = np.hstack(H0)
        cluster_p_values = _pval_from_histogram(cluster_stats, H0, tail=0)

        print(cluster_p_values)
        if plot:
            plot_clusters(T_obs, clusters, cluster_p_values, cmap='viridis')
            
        return T_obs, clusters, cluster_p_values
    
    else:
        print("No clusters found")
        return None, None, None

def _fit_lm(data, design_matrix, names):
    """Aux function."""
    from scipy import stats
    from numpy import linalg
    n_samples = len(data)
    n_features = np.product(data.shape[1:])
    if design_matrix.ndim != 2:
        raise ValueError('Design matrix must be a 2d array')
    n_rows, n_predictors = design_matrix.shape

    if n_samples != n_rows:
        raise ValueError('Number of rows in design matrix must be equal '
                         'to number of observations')
    if n_predictors != len(names):
        raise ValueError('Number of regressor names must be equal to '
                         'number of column in design matrix')

    y = np.reshape(data, (n_samples, n_features))
    betas, resid_sum_squares, _, _ = linalg.lstsq(a=design_matrix, b=y)

    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))
    tiny = np.finfo(np.float64).tiny
    beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))
    for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
        beta[predictor] = x.reshape(data.shape[1:])
        stderr[predictor] = sqrt_noise_var * unscaled_stderr
        p_val[predictor] = np.empty_like(stderr[predictor])
        t_val[predictor] = np.empty_like(stderr[predictor])

        stderr_pos = (stderr[predictor] > 0)
        beta_pos = (beta[predictor] > 0)
        t_val[predictor][stderr_pos] = (beta[predictor][stderr_pos] /
                                        stderr[predictor][stderr_pos])
        cdf = stats.t.cdf(np.abs(t_val[predictor][stderr_pos]), df)
        p_val[predictor][stderr_pos] = np.clip((1. - cdf) * 2., tiny, 1.)
        # degenerate cases
        mask = (~stderr_pos & beta_pos)
        t_val[predictor][mask] = np.inf * np.sign(beta[predictor][mask])
        p_val[predictor][mask] = tiny
        # could do NaN here, but hopefully this is safe enough
        mask = (~stderr_pos & ~beta_pos)
        t_val[predictor][mask] = 0
        p_val[predictor][mask] = 1.
        mlog10_p_val[predictor] = -np.log10(p_val[predictor])

    return beta, stderr, t_val, p_val, mlog10_p_val
    
def get_cluster_means(array, clusters, cluster_p_values):
    return [array[:, i].mean(axis=1) for n, i in enumerate(clusters) if cluster_p_values[n] < .05]