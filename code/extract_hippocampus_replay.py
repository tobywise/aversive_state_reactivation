import mne
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib as skjoblib
import joblib
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from sklearn.preprocessing import scale
from scipy.signal import hilbert
from scipy.stats import sem, ttest_1samp
from statsmodels.tsa.ar_model import AutoReg


def extract_hippocampus(sub, phase, band):
    stc = joblib.load('/central/scratch/tobywise/data/derivatives/beamforming/{2}_filtered_source_estimates/_400hz/{0}/sub-{1}_{2}_filtered-source-estimates'.format(phase, sub, band))
    
    # Deals with an issue in earlier versions of MNE
    for i in stc:
        i.vertices = [i.vertices]
    
    extracted = mne.extract_label_time_course(stc, labels=(r'/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/fsaverage/mri/aparc.a2009s+aseg.mgz', ['Left-Hippocampus', 'Right-Hippocampus']), src=src, allow_empty=True, verbose='ERROR')
    
    sfreq = 1 / stc[0].tstep

    return extracted, sfreq

def interpolate_seq(replay_sum, sfreq):
    
    new_replay_sum = []
    
    sfreq_multiplier = sfreq / 100
    
    for t in range(replay_sum.shape[1]):
        
        x = np.arange(replay_sum.shape[0])
        y = replay_sum[:, t]
        f = interp1d(x, y)

        xnew = np.linspace(0, x.max(), int(replay_sum.shape[0] * sfreq_multiplier))
        ynew = f(xnew)
        
        new_replay_sum.append(ynew)
        
    new_replay_sum = np.stack(new_replay_sum).T
    
    return new_replay_sum

def run_lm(replay, signal, order=5):
    exog = np.stack([scale(signal)]).T
    mod = AutoReg(scale(replay), order, exog=exog, old_names=False)
    res = mod.fit()
    return res.params[-1:]
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("phase")  # Task phase (outcome or planning)
    parser.add_argument('band')  # Frequency band
    args = parser.parse_args()

    if mne.__version__ != '0.21.0':
        raise ValueError('This notebook requires MNE version 0.21 to run, currently installed version = {0}'.format(mne.__version__))
    task_phase = args.phase
    band = args.band

    assert band in ['theta', 'low_gamma', 'high_gamma'], 'Band must be in one of theta, low_gamma, high_gamma'
    assert task_phase in ['outcome', 'planning'], "Invalid phase argument"


    subject_ids = [str(i).zfill(3) for i in range(1, 29)]

    fs_dir = '../data/derivatives/mri/fsaverage'
    src = os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
    src = mne.read_source_spaces(src)

    data_dir = '/central/scratch/tobywise/data/derivatives'
    output_path = os.path.join(data_dir, 'beamforming_within_trial/{0}/{1}'.format(task_phase, band))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_extracted = []
    all_replay = []
    all_reactivation = []

    print("Extracting data")

    for sub in tqdm(subject_ids):
    
        # Extract hippocampus signal
        extracted, sfreq = extract_hippocampus(sub, task_phase, band)
        extracted = np.stack(extracted)

        # High gamma has an extra sample at the end for some reason - trim this off
        if band == 'high_gamma':
            extracted = extracted[..., :np.round(extracted.shape[-1], -1)]

        all_extracted.append(extracted)

        # Process sequenceness
        beh_data_path = '../data/sub-{0}/ses-01/beh/sub-{0}_ses-01_task-AversiveLearningReplay_responses.csv'.format(sub)
        behaviour = pd.read_csv(beh_data_path)
        behaviour = behaviour[~behaviour['trial_number'].isnull()].reset_index()

        if task_phase == 'outcome':
            seq = skjoblib.load('../data/derivatives/sw_sequenceness/outcome/classifier_idx_37/sub-{0}_outcome_sequenceness_idx_37.pkl'.format(sub))['difference']
        else:
            seq = skjoblib.load('../data/derivatives/sw_sequenceness/planning/classifier_idx_52/sub-{0}_planning_sequenceness_idx_52.pkl'.format(sub))['difference']
            seq = seq[:, behaviour['trial_type'] == 1, ...]  # Select generalisation trials
        
        # Collapse across lags
        seq_all_lag = seq.mean(axis=2)

        # Abs will show any replay
        replay_sum = np.abs(seq_all_lag).max(axis=2)
        
        # Interpolate to match with hippocampus signal sampled at higher rate
        replay_sum = interpolate_seq(replay_sum, sfreq)
        
        assert replay_sum.shape[0] == extracted.shape[-1] - (50 * (sfreq / 100)), 'Incorrect number of timepoints, signal implies {0}, replay has {1}'.format(extracted.shape[-1] - (50 * (sfreq / 100)), replay_sum.shape[0])
        assert replay_sum.shape[1] == extracted.shape[0], 'Incorrect number of trials, signal has {0}, replay has {1}'.format(extracted.shape[0], replay_sum.shape[1])
                        
        all_replay.append(replay_sum)

        # Process reactivation
        if task_phase == 'outcome':
            temp_gen_pred = np.load('../data/derivatives/temporal_generalisation/sub-{0}/temporal_generalisation_predicted/outcome/all_trials/sub-{0}_end_stimulus_outcome.npy'.format(sub))
            temp_gen_pred = temp_gen_pred[:, 37, :, 0]  # Select classifier trained on 370ms, only select one stimulus as probabilities are complementary
        else:
            temp_gen_pred = np.load('../data/derivatives/temporal_generalisation/sub-{0}/temporal_generalisation_predicted/planning/generalisation_trials/sub-{0}_end_stimulus_planning.npy'.format(sub))
            temp_gen_pred = temp_gen_pred[:, 52, :, 0]  # Select classifier trained on 520ms, only select one stimulus as probabilities are complementary
        
        temp_gen_pred = interpolate_seq(temp_gen_pred.T, sfreq)
        temp_gen_pred = np.abs(temp_gen_pred - 0.5)  # Get deviation from 0.5 as evidence of any reactivation
        
        assert temp_gen_pred.shape[0] == extracted.shape[-1], 'Incorrect number of timepoints, signal has {0}, reactivation has {1}'.format(extracted.shape[-1], temp_gen_pred.shape[0])
        assert temp_gen_pred.shape[1] == extracted.shape[0], 'Incorrect number of trials, signal has {0}, reactivation has {1}'.format(extracted.shape[0], temp_gen_pred.shape[1])
        
        all_reactivation.append(temp_gen_pred)

    # Stack
    all_extracted = np.stack(all_extracted)
    all_replay = np.stack(all_replay)
    all_reactivation = np.stack(all_reactivation)

    # Save
    np.save(os.path.join(output_path, 'phase-{0}_band-{1}_hippocampus-signal'.format(task_phase, band)), all_extracted)
    np.save(os.path.join(output_path, 'phase-{0}_band-{1}_replay'.format(task_phase, band)), all_replay)
    np.save(os.path.join(output_path, 'phase-{0}_band-{1}_reactivation'.format(task_phase, band)), all_reactivation)
    

    # Get amplitude and phase
    phase = np.zeros_like(all_extracted)
    amplitude = np.zeros_like(all_extracted)

    for sub in range(phase.shape[0]):

        for hemi in [0, 1]:

            for trial in range(phase.shape[1]):    
                analytic_signal = hilbert(all_extracted[sub, trial, hemi, :])
                amplitude_envelope = np.abs(analytic_signal)
                trial_phase = np.angle(analytic_signal)
                phase[sub, trial, hemi, :] = trial_phase
                amplitude[sub, trial, hemi, :] = amplitude_envelope

    # Run linear models
    print("Running linear models")

    amplitude_betas = {'replay': np.zeros(amplitude.shape[:-1] + (1, )), 'reactivation': np.zeros(amplitude.shape[:-1] + (1, ))}
    phase_betas = {'replay': np.zeros(phase.shape[:-1] + (1, )), 'reactivation': np.zeros(phase.shape[:-1] + (1, ))}

    for sub in tqdm(range(amplitude_betas['replay'].shape[0])):

        for hemi in [0, 1]:

            for trial in range(amplitude_betas['replay'].shape[1]):
                
                # REPLAY
                # Account for missing samples due to windowing
                missing_samples = (int(np.floor((all_extracted.shape[-1] - all_replay.shape[1]) / 2)), int(np.ceil((all_extracted.shape[-1] - all_replay.shape[1]) / 2)))

                # Amplitude
                amplitude_betas['replay'][sub, trial, hemi, :] = run_lm(scale(all_replay[sub, :, trial]), amplitude[sub, trial, hemi, missing_samples[0]:-missing_samples[1]])
                
                # Phase
                phase_betas['replay'][sub, trial, hemi, :] = run_lm(scale(all_replay[sub, :, trial]), phase[sub, trial, hemi, missing_samples[0]:-missing_samples[1]])
                
                # REACTIVATION
                # Amplitude
                amplitude_betas['reactivation'][sub, trial, hemi, :] = run_lm(scale(all_reactivation[sub, :, trial]), amplitude[sub, trial, hemi, :])
                
                # Phase
                phase_betas['reactivation'][sub, trial, hemi, :] = run_lm(scale(all_reactivation[sub, :, trial]), phase[sub, trial, hemi, :])
                
    # Do t-tests across subjects
    print("Running t-tests")
    output_df = {'measure': [], 'signal_measure': [], 'hemi': [], 'mean': [], 'sem': [], 'p': [], 't': []}

    for measure in ['reactivation', 'replay']:
        for hemi in [0, 1]:
            for signal_measure in ['amplitude', 'phase']:
                
                if signal_measure == 'amplitude':
                    betas = amplitude_betas
                else:
                    betas = phase_betas
            
                mean_ = betas[measure].mean(axis=1)[:, hemi, 0].mean()
                sem_ = sem(betas[measure].mean(axis=1)[:, hemi, 0])
                t, p = ttest_1samp(betas[measure].mean(axis=1)[:, hemi, 0], 0) 

                output_df['measure'].append(measure)
                output_df['signal_measure'].append(signal_measure)
                output_df['hemi'].append(hemi)
                output_df['mean'].append(mean_)
                output_df['sem'].append(sem_)
                output_df['p'].append(p)
                output_df['t'].append(t)

    output_df = pd.DataFrame(output_df)
    
    output_df['phase'] = task_phase
    output_df['band'] = band

    output_df.to_csv(os.path.join(output_path, 'phase-{0}_band-{1}_ttests.csv'.format(task_phase, band)))

    print("DONE")