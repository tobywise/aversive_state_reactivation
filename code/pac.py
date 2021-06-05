import mne
import os
import argparse
import numpy as np
import argparse
import joblib
from sklearn.externals import joblib as skjoblib
from pactools import Comodulogram
import pandas as pd
from joblib import Parallel, delayed


def extract_hippocampus(sub, phase):
    stc = joblib.load('../data/derivatives/beamforming/unfiltered_filtered_source_estimates/_400hz/{0}/{1}_unfiltered_filtered-source-estimates'.format(phase, sub))
    
    # Deals with an issue in earlier versions of MNE
    for i in stc:
        i.vertices = [i.vertices]
    
    extracted = mne.extract_label_time_course(stc, labels=(r'/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/fsaverage/mri/aparc.a2009s+aseg.mgz', ['Left-Hippocampus', 'Right-Hippocampus']), src=src, allow_empty=True, verbose='ERROR')
    
    sfreq = 1 / stc[0].tstep

    return extracted, sfreq

def compute_comod_diff(permutation_idx, extracted_signal, hemi, phase):

    estimator = Comodulogram(sfreq, low_fq_range=np.linspace(4, 8, 30), high_fq_range=np.linspace(30, 200, 60), 
                            method='tort', n_surrogates=0, n_jobs=1, random_state=123)

    permutation_diff = []

    for _ in permutation_idx:

        n_trials = extracted_signal.shape[0]

        split = np.zeros(n_trials)
        split[:int(n_trials / 2)] = 1
        np.random.shuffle(split)
        split = split.astype(bool)

        median_split_high = split
        median_split_low = np.invert(median_split_high)

        median_splits = {'high': median_split_high, 'low': median_split_low}

        comods = {}

        for split in ['high', 'low']:

            if phase == 'outcome':
                estimator.fit(extracted_signal[median_splits[split], hemi, int(1.2 * sfreq):])
            else:
                estimator.fit(extracted_signal[median_splits[split], hemi, :])

            comod = estimator.comod_
            comods[split] = comod

        comod_diff = comods['high'] - comods['low']

        permutation_diff.append(comod_diff)

    return permutation_diff



if __name__ == "__main__":

    import os
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1" 

    parser = argparse.ArgumentParser()
    parser.add_argument("subject_idx")  # Index of the subjec to run
    parser.add_argument("phase")  # Task phase (outcome or planning)
    parser.add_argument("hemi", type=int)  # Hemisphere (0 = left, 1 = right)
    parser.add_argument("var")  # Variable to use (replay or reactivation)
    parser.add_argument('type')  # Analysis type - diff computes difference between high/low reactivation, within gives PAC estimates within condition
    args = parser.parse_args()

    # Check args
    assert args.phase in ['outcome', 'planning'], "Invalid phase argument"
    assert args.hemi in [0, 1], "Invalid hemi argument"
    assert args.var in ['reactivation', 'replay'], "Invalid var argument"

    # Number of cores to use
    n_jobs = len(os.sched_getaffinity(0))

    # Random seed
    np.random.seed(123)

    # Check MNE is correct version
    if not mne.__version__ == '0.21.0':
        raise ValueError("Wrong version of MNE. Current installed version = {0}, to replicate results version 0.21.0 is required".format(mne.__version__))

    # Get subject ID
    subject_ids = [str(i).zfill(3) for i in range(1, 29)]
    subject = 'sub-' + subject_ids[int(args.subject_idx) - 1]

    if args.hemi == 0:
        hemi_string = 'left'
    else:
        hemi_string = 'right'

    print("RUNNING SUBJECT {0}".format(subject))
    print("Phase = {0} | Hemisphere = {1} | Variable = {2}".format(args.phase, args.hemi, args.var))
    print("Current working directory = {0}".format(os.getcwd()))

    os.chdir('/central/scratch/tobywise/data')

    fs_dir = '/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/fsaverage'
    data_dir = '/central/groups/mobbslab/toby/old/meg/data/'
    src = os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
    src = mne.read_source_spaces(src)

    # Extract signal
    extracted_signal, sfreq = extract_hippocampus(subject, args.phase)
    extracted_signal = np.stack(extracted_signal)

    # Load behaviour
    beh_data_path = os.path.join(data_dir, '{0}/ses-01/beh/{0}_ses-01_task-AversiveLearningReplay_responses.csv'.format(subject))
    behaviour = pd.read_csv(beh_data_path)
    behaviour = behaviour[~behaviour['trial_number'].isnull()].reset_index()

    # Get reactivation
    if args.var == 'reactivation':

        if args.phase == 'outcome':

            # Outcomes only received on learning trials
            behaviour = behaviour[behaviour['trial_type'] == 0]

            # Load reactivation and mask
            temp_gen_pred = np.load(os.path.join(data_dir, 
                                                'derivatives/temporal_generalisation/{0}/temporal_generalisation_predicted/outcome/all_trials/{0}_end_stimulus_outcome.npy'.format(subject)))
            mask = np.load(os.path.join(data_dir, 'derivatives/temporal_generalisation/outcome_mask.npy'))

            reactivation = []

            for i in range(temp_gen_pred.shape[0]):
                reactivation.append(temp_gen_pred[i, :, 120:, 0][mask].mean())

        elif args.phase == 'planning':

            # Only use generalisation trials
            behaviour = behaviour[behaviour['trial_type'] == 1]
            
            # Load reactivation and mask
            temp_gen_pred = np.load(os.path.join(data_dir, 
                                                'derivatives/temporal_generalisation/{0}/temporal_generalisation_predicted/planning/generalisation_trials/{0}_end_stimulus_planning.npy'.format(subject)))
            mask = np.load(os.path.join(data_dir, 'derivatives/temporal_generalisation/planning_generalisation_mask.npy'))

            reactivation = []

            for i in range(temp_gen_pred.shape[0]):
                reactivation.append(temp_gen_pred[i, :, :, 0][mask].mean())
                
        reactivation = np.array(reactivation)

        behaviour['reactivation'] = reactivation

        # CHOSEN/UNCHOSEN
        behaviour.loc[behaviour['State_4_shown'] > 12, 'reactivation'] = 1 - behaviour.loc[behaviour['State_4_shown'] > 12, 'reactivation']
        behaviour['reactivation'] = behaviour['reactivation'] - 0.5

        median_split_high = (behaviour['reactivation'] > np.median(behaviour['reactivation'])).values
        median_split_low = np.invert(median_split_high)

    elif args.var == 'replay':

        if args.phase == 'outcome':
            
            # Get mean sequenceness
            seq = skjoblib.load(os.path.join(data_dir, 
                                             'derivatives/sw_sequenceness/outcome/classifier_idx_37/{0}_outcome_sequenceness_idx_37.pkl'.format(subject)))['difference']
            seq = np.abs(seq.mean(axis=2)).max(axis=2).mean(axis=0)

        elif args.phase == 'planning':

            seq = skjoblib.load(os.path.join(data_dir, 
                                            'derivatives/sw_sequenceness/planning/classifier_idx_52/{0}_planning_sequenceness_idx_52.pkl'.format(subject)))['difference']
            seq = seq[:, behaviour['trial_type'] == 1, ...]
            seq = np.abs(seq.mean(axis=2)).max(axis=2).mean(axis=0)

        median_split_high = (seq > np.median(seq))
        median_split_low = np.invert(median_split_high)

    median_splits = {'high': median_split_high, 'low': median_split_low}

    if args.type == 'within':

        # Set up PAC estimator
        estimator = Comodulogram(sfreq, low_fq_range=np.linspace(4, 8, 30), high_fq_range=np.linspace(30, 200, 60), 
                                    method='duprelatour', n_surrogates=100, n_jobs=n_jobs, random_state=123)

        # Run PAC
        for split in ['high', 'low']:

            # Set up PAC estimator
            rng = np.random.RandomState(123)
            estimator = Comodulogram(sfreq, low_fq_range=np.linspace(4, 8, 30), high_fq_range=np.linspace(30, 200, 60), 
                                        method='duprelatour', n_jobs=1, random_state=rng)
        
        
            if args.phase == 'outcome':
                estimator.fit(extracted_signal[median_splits[split], args.hemi, int(1.2 * sfreq):])
            else:
                estimator.fit(extracted_signal[median_splits[split], args.hemi, :])

            out_dir = '../data/derivatives/pac/{0}/{1}'.format(args.var, args.phase)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            joblib.dump(estimator, os.path.join(out_dir, '{0}_hemi-{1}_pac-estimator_split-{2}'.format(subject, hemi_string, split)))

    elif args.type == 'diff':

        # Set up PAC estimator
        estimator = Comodulogram(sfreq, low_fq_range=np.linspace(4, 8, 30), high_fq_range=np.linspace(30, 200, 60), 
                                    method='tort', n_jobs=1, random_state=123)
    
        comods = {}

        # Run PAC
        for split in ['high', 'low']:

            if args.phase == 'outcome':
                estimator.fit(extracted_signal[median_splits[split], args.hemi, int(1.2 * sfreq):])
            else:
                estimator.fit(extracted_signal[median_splits[split], args.hemi, :])

            # Get true estimate for this condition
            true_comod = estimator.comod_
            comods[split] = true_comod

        true_comod_diff = comods['high'] - comods['low']

        # Run permutations - run the procedure on shuffled labels and get the difference
        n_permutations = 100
        permutation_idx = np.split(np.arange(n_permutations), n_jobs)

        permutation_diffs = Parallel(n_jobs=n_jobs)(delayed(compute_comod_diff)(i, extracted_signal, args.hemi, args.phase) for i in permutation_idx)
        permutation_diffs = np.stack([i for j in permutation_diffs for i in j])

        diffs = {}
        diffs['true'] = true_comod_diff
        diffs['permutations'] = permutation_diffs

        out_dir = '../data/derivatives/pac/{0}/{1}/{2}'.format(args.var, args.phase, args.type)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        joblib.dump(diffs, os.path.join(out_dir, '{0}_hemi-{1}_pac-diff'.format(subject, hemi_string)))

