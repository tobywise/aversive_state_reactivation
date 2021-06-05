import joblib
import mne
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import ttest_1samp
from tqdm import tqdm
from mne.stats import permutation_cluster_1samp_test
from mne import spatial_src_connectivity
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy import stats as stats
import numpy as np

if __name__ == "__main__":

    import os
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1" 

    parser = argparse.ArgumentParser()
    parser.add_argument("phase")  # Task phase (outcome or planning)
    parser.add_argument("var")  # Variable to use (replay or reactivation)
    parser.add_argument('band')  # Frequency band
    parser.add_argument('measure')  # Amplitude or phase
    args = parser.parse_args()

    if mne.__version__ != '0.21.0':
        raise ValueError('This code requires MNE version 0.21 to run, currently installed version = {0}'.format(mne.__version__))
    phase = args.phase
    band = args.band

    assert band in ['theta', 'low_gamma', 'high_gamma'], 'Band must be in one of theta, low_gamma, high_gamma'
    assert args.phase in ['outcome', 'planning'], "Invalid phase argument"
    assert args.measure in ['amplitude', 'phase'], 'Invalid measure argument'

    # Number of cores to use
    n_jobs = len(os.sched_getaffinity(0))

    print("Collecting results for {0} phase, frequency band = {1}, variable = {2}".format(args.phase, args.band, args.var))


    fs_dir = '/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/fsaverage'
    data_dir = '/central/scratch/tobywise/data/derivatives'
    subjects_dir = os.path.dirname(fs_dir)
    
    output_path = os.path.join(data_dir, 'beamforming_group_analysis/{0}/{1}/{2}'.format(args.var, phase, band))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get source stuff
    src = os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
    src = mne.read_source_spaces(src)
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    ras_mni_t = mne.transforms.read_ras_mni_t('fsaverage', subjects_dir)

    # Subject IDs
    subs = [str(i).zfill(3) for i in range(1, 29)]
    n_subs = len(subs)

    # Iterate over everything and get results
    extracted = {}

    extracted_df = {'phase': [], 
                    'band': [],
                    'Hemisphere': [],
                    'value': []
                    }


    # Collect results of regression analysis as MNE source space objects
    regression_results = []

    for subject in tqdm(subs):
        res = mne.read_source_estimate(os.path.join(data_dir, 
                                                    'beamforming/{2}_regressions/{0}/{1}_{3}'.format(phase, band, args.var, args.measure), 
                                                    'sub-{0}_source-space-regression-{1}__400hz__{2}-vl.stc'.format(subject, args.measure, args.var)))
        regression_results.append(res)

    # Get sample frequency
    sfreq = 1 / res.tstep

    # Get all subjects' data in source space as a numpy array
    regression_data = np.transpose(np.stack([s.data for s in regression_results]), [0, 2, 1])

    # Extract hippocampus
    extracted_signal = mne.extract_label_time_course(regression_results, labels=(os.path.join(fs_dir, r'mri/aparc.a2009s+aseg.mgz'), ['Left-Hippocampus', 'Right-Hippocampus']), src=src, allow_empty=True)
    extracted_signal = np.stack(extracted_signal)

    # Save hippocampus signal
    np.save(os.path.join(output_path, '{0}_phase-{1}_band-{2}_measure-{3}_hippocampus-signal'.format(args.var, phase, band, args.measure)), extracted_signal)
    
    ##################################
    # T-test collapsing across trial #
    ##################################

    print("Running t-tests")

    tL, pL = ttest_1samp(extracted_signal[:, 0, :].mean(axis=1), 0)
    tR, pR = ttest_1samp(extracted_signal[:, 1, :].mean(axis=1), 0)
    ttest_outputs = {'left': {'t': tL, 'p': pL}, 
                            'right': {'t': tR, 'p': pR}}
    joblib.dump(ttest_outputs, os.path.join(output_path, '{0}_phase-{1}_band-{2}_measure-{3}_ttest-output'.format(args.var, phase, band, args.measure)))
    
    
    # Convert to dataframe
    for hemi in [0, 1]:
        extracted_df['phase'] += [phase] * n_subs
        extracted_df['band'] += [band] * n_subs
        
        if hemi == 0:
            extracted_df['Hemisphere'] += ['L'] * n_subs
        else:
            extracted_df['Hemisphere'] += ['R'] * n_subs
        
        extracted_df['value'] += extracted_signal[:, hemi, :].mean(axis=1).tolist()

    extracted_df = pd.DataFrame(extracted_df)
    extracted_df['value'] /= 1e15  # Put things on a more sensible scale

    extracted_df.to_csv(os.path.join(output_path, '{0}_phase-{1}_band-{2}_measure-{3}_extracted-df.csv'.format(args.var, phase, band, args.measure)))

    ########################
    # GET CLUSTERS IN TIME #
    ########################

    print("Running t-tests over time")

    if phase == 'outcome':
        extracted_data = extracted_signal[..., :int(1.6 * sfreq)]
    else:
        extracted_data = extracted_signal[..., int(0.6 * sfreq):int(2.6 * sfreq)]
    
    T_obs_L, clusters_L, cluster_p_values_L, _ = \
        permutation_cluster_1samp_test(extracted_data[:, 0, :], n_permutations=1000, tail=0, n_jobs=1, seed=123, verbose='ERROR')
    T_obs_R, clusters_R, cluster_p_values_R, _ = \
        permutation_cluster_1samp_test(extracted_data[:, 1, :], n_permutations=1000, tail=0, n_jobs=1, seed=123, verbose='ERROR')

    cluster_output = {'L': {'tvals': T_obs_L, 'clusters': clusters_L, 'cluster_p_values': cluster_p_values_L},
                    'R': {'tvals': T_obs_R, 'clusters': clusters_R, 'cluster_p_values': cluster_p_values_R}}
    
    joblib.dump(cluster_output, os.path.join(output_path, '{0}_phase-{1}_band-{2}_measure-{3}_cluster-output'.format(args.var, phase, band, args.measure)))
    
    ###############
    # Whole brain #
    ###############

    print("Running whole-brain")

    subjects_dir = r'/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/'
    fname_t1_fsaverage = os.path.join(subjects_dir, 'fsaverage', 'mri', 'brain.mgz')

    connectivity = spatial_src_connectivity(src)

    p_threshold = 0.01
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., regression_data[0].shape[0] - 1)

    if phase == 'outcome':
        T_obs, clusters, cluster_p_values, H0 = \
            spatio_temporal_cluster_1samp_test(regression_data[:, :int(1.6 * sfreq), :], connectivity=connectivity, n_jobs=n_jobs, n_permutations=1000,
                                            threshold=t_threshold, seed=123)
    else:
        T_obs, clusters, cluster_p_values, H0 = \
            spatio_temporal_cluster_1samp_test(regression_data[:, int(.6 * sfreq):int(2.6 * sfreq), :], connectivity=connectivity, n_jobs=n_jobs, n_permutations=1000,
                                            threshold=t_threshold, seed=123)

    whole_brain_cluster_output = {'T_obs': T_obs, 'clusters': clusters, 'cluster_p_values': cluster_p_values}

    joblib.dump(whole_brain_cluster_output, os.path.join(output_path, '{0}_phase-{1}_band-{2}_measure-{3}_whole-brain-cluster-output'.format(args.var, phase, band, args.measure)))
    
    print("DONE")

