import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.stats import linear_regression
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import joblib
from sklearn.preprocessing import scale
from scipy.signal import hilbert

def get_amplitude_phase(stc_epochs):

    # Use hilbert transform to get power in the gamma band for each trial
    stcs_amplitude = []
    stcs_phase = []

    for s in tqdm(stc_epochs):
        s_temp_amplitude = s.copy()
        s_temp_phase = s.copy()
        analytic_signal = hilbert(s_temp_amplitude.data)
        amplitude_envelope = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)

        s_temp_amplitude.data = amplitude_envelope
        stcs_amplitude.append(s_temp_amplitude)

        s_temp_phase.data = phase
        stcs_phase.append(s_temp_phase)

    return stcs_amplitude, stcs_phase

def standardise_betas(res, pred_cols, fname, subject, downsample_string):
    for pred in pred_cols:
        standardised_beta = res[pred].beta / (res[pred].stderr ** 2)
        standardised_beta.crop(0, None).save(fname.format(subject, pred, downsample_string))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject_idx")  # Index of the subjec to run
    parser.add_argument('phase')
    # parser.add_argument('downsample', default=0)
    args = parser.parse_args()

    downsample = 400

    if downsample is not 0:
        downsample_string = '_{0}hz'.format(downsample)

    subject_ids = [str(i).zfill(3) for i in range(1, 29)]
    subject = subject_ids[int(args.subject_idx) - 1]

    print("RUNNING SUBJECT {0}".format(subject))
    print("Current working directory = {0}".format(os.getcwd()))

    # Check MNE is correct version
    if not mne.__version__ == '0.20.6':
        raise ValueError("Wrong version of MNE. Current installed version = {0}, to replicate results version 0.20.6 is required".format(mne.__version__))

    # Things needed for beamforming
    fs_dir = 'data/derivatives/mri/fsaverage'
    data_dir = 'data/derivatives'
    # output_dir = 'data/derivatives'
    output_dir = '/central/scratch/tobywise/data/derivatives'

    src = os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    trans = os.path.join(data_dir, 'trans/sub-{0}-trans.fif'.format(subject))

    # Simulated data
    simulated_data = pd.read_csv('data/derivatives/behavioural_modelling/simulated_data.csv')

    # Behaviour etc
    temp_sim = simulated_data[simulated_data['Subject'] == 'sub-' + subject]

    beh_data_path = 'data/sub-{0}/ses-01/beh/sub-{0}_ses-01_task-AversiveLearningReplay_responses.csv'.format(subject)
    behaviour_raw = pd.read_csv(beh_data_path)
    behaviour_raw = behaviour_raw[~behaviour_raw['trial_number'].isnull()].reset_index()

    behaviour = pd.merge(temp_sim, behaviour_raw, on='trial_number')
    behaviour = behaviour[behaviour['trial_type'] == 0]
    behaviour['intercept'] = 1
    behaviour['abs_pe'] = np.abs(behaviour['pe_chosen'])
    behaviour.loc[behaviour['Shock_received'] == 0, 'Shock_received'] = -1

    

    ###########
    # OUTCOME #
    ###########

    if args.phase == 'outcome':

        # Temporal generalisation 
        temp_gen_pred = np.load('data/derivatives/temporal_generalisation/sub-{0}/temporal_generalisation_predicted/outcome/all_trials/sub-{0}_end_stimulus_outcome.npy'.format(subject))
        mask = np.load('data/derivatives/temporal_generalisation/outcome_mask.npy')

        reactivation = []

        for i in range(temp_gen_pred.shape[0]):
            reactivation.append(temp_gen_pred[i, :, 120:, 0][mask].mean())
            
        reactivation = np.array(reactivation)

        behaviour['abs_pe:shock_received'] = behaviour['abs_pe'] * behaviour['Shock_received']
        behaviour['reactivation'] = reactivation

        # CHOSEN/UNCHOSEN
        behaviour.loc[behaviour['State_4_shown'] > 12, 'reactivation'] = 1 - behaviour.loc[behaviour['State_4_shown'] > 12, 'reactivation']
        behaviour['reactivation'] = behaviour['reactivation'] - 0.5

        pred_cols = ['Shock_received', 'abs_pe', 'abs_pe:shock_received', 'reactivation']
        behaviour[pred_cols] = behaviour[pred_cols].apply(lambda x: scale(x))

        dm = behaviour[['intercept'] + pred_cols].values.astype(float)

        # Epochs    
        phase = 'outcome'
        epochs = mne.read_epochs(os.path.join(data_dir, 'preprocessing', 'sub-{0}'.format(subject), 'task', 
                                'sub-{0}_ses-01_task-AversiveLearningReplay_run-{1}_proc_ICA{2}-epo.fif.gz'.format(subject, phase, downsample_string)))
        epochs.pick_types(meg=True, ref_meg=False)
        
        # Filter and run beamforming
        bands = {'theta': (4, 8), 'low_gamma': (30, 60), 'high_gamma': (120, 199), 'unfiltered': (None, None)}
        # bands = {'unfiltered': (None, None)}

        for band, freqs in bands.items():

            print("Frequency band = {0}, {1} - {2}".format(band, freqs[0], freqs[1]))

            epochs_resampled = epochs.copy()
            print("Epochs sfreq = {0}".format(epochs_resampled.info['sfreq']))

            # Downsample to save time
            if freqs[1] is not None and freqs[1] < 199:
                resample_freq = np.max([freqs[1] * 2 + 10, 100])
                epochs_resampled = epochs_resampled.resample(resample_freq)

            # Restrict to band
            epochs_filtered = epochs_resampled.filter(*freqs)

            epochs_filtered.apply_baseline((-0.5, 0))
            data_cov = mne.compute_covariance(epochs_filtered, tmin=0, tmax=None,
                                            method='empirical')

            # Make beamformer
            fwd = mne.make_forward_solution(epochs_filtered.info, trans=trans, src=src, bem=bem, eeg=False)
            filters = make_lcmv(epochs_filtered.info, fwd, data_cov, reg=0.05, pick_ori='max-power', weight_norm='unit-noise-gain', rank=None)

            # Apply beamforming
            stc_epochs = apply_lcmv_epochs(epochs_filtered, filters, max_ori_out='signed')

            # Save source estimates
            if not os.path.exists(os.path.join(output_dir, 'beamforming/{0}_filtered_source_estimates/{1}/outcome'.format(band, downsample_string))):
                os.makedirs(os.path.join(output_dir, 'beamforming/{0}_filtered_source_estimates/{1}/outcome'.format(band, downsample_string)))
            joblib.dump(stc_epochs, os.path.join(output_dir, 'beamforming/{0}_filtered_source_estimates/{1}/outcome'.format(band, downsample_string), 
                                                            'sub-{0}_{1}_filtered-source-estimates'.format(subject, band)))

            if band != 'unfiltered':

                # Get amplitude & phase
                stcs_amplitude, stcs_phase = get_amplitude_phase(stc_epochs)
                del stc_epochs

                # Save
                if not os.path.exists(os.path.join(output_dir, 'beamforming/{0}_amplitude_source_estimates/{1}/outcome'.format(band, downsample_string))):
                    os.makedirs(os.path.join(output_dir, 'beamforming/{0}_amplitude_source_estimates/{1}/outcome'.format(band, downsample_string)))
                joblib.dump(stcs_amplitude, os.path.join(output_dir, 'beamforming/{0}_amplitude_source_estimates/{1}/outcome'.format(band, downsample_string), 
                                                                'sub-{0}_{1}-amplitude-source-estimates'.format(subject, band)))

                if not os.path.exists(os.path.join(output_dir, 'beamforming/{0}_phase_source_estimates/{1}/outcome'.format(band, downsample_string))):
                    os.makedirs(os.path.join(output_dir, 'beamforming/{0}_phase_source_estimates/{1}/outcome'.format(band, downsample_string)))
                joblib.dump(stcs_phase, os.path.join(output_dir, 'beamforming/{0}_phase_source_estimates/{1}/outcome'.format(band, downsample_string), 
                                                                'sub-{0}_{1}-phase-source-estimates'.format(subject, band)))

                # GLM
                res_amplitude = linear_regression(stcs_amplitude, dm, ['intercept'] + pred_cols)
                res_phase = linear_regression(stcs_phase, dm, ['intercept'] + pred_cols)

                # Create output directory
                for measure in ['phase', 'amplitude']:
                    out_dir = os.path.join(output_dir, 'beamforming/reactivation_regressions/outcome/{0}_{1}'.format(band, measure))
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                # Standardise and save
                standardise_betas(res_amplitude, pred_cols, os.path.join(output_dir, 'beamforming/reactivation_regressions/outcome/{0}_amplitude'.format(band), 'sub-{0}_source-space-regression-amplitude_{2}__{1}'), subject, downsample_string)
                standardise_betas(res_phase, pred_cols, os.path.join(output_dir, 'beamforming/reactivation_regressions/outcome/{0}_phase'.format(band), 'sub-{0}_source-space-regression-phase_{2}__{1}'), subject, downsample_string)
            
                del res_amplitude
                del res_phase
                del stcs_amplitude
                del stcs_phase
                del epochs_filtered
                del epochs_resampled

    ############
    # PLANNING #
    ############

    elif args.phase == 'planning':

        # Temporal generalisation 
        temp_gen_pred = np.load('data/derivatives/temporal_generalisation/sub-{0}/temporal_generalisation_predicted/planning/generalisation_trials/sub-{0}_end_stimulus_planning.npy'.format(subject))
        mask = np.load('data/derivatives/temporal_generalisation/planning_generalisation_mask.npy')

        reactivation = []

        for i in range(temp_gen_pred.shape[0]):
            reactivation.append(temp_gen_pred[i, :, :, 0][mask].mean())

        reactivation = np.array(reactivation)

        # Get behaviour
        behaviour = pd.merge(temp_sim, behaviour_raw, on='trial_number')
        behaviour = behaviour[behaviour['trial_type'] == 1]
        behaviour['value_diff'] = np.abs(behaviour['value_A'] - behaviour['value_B'])
        behaviour['reactivation'] = reactivation

        # CHOSEN/UNCHOSEN
        behaviour.loc[behaviour['State_4_shown'] > 12, 'reactivation'] = 1 - behaviour.loc[behaviour['State_4_shown'] > 12, 'reactivation']
        behaviour['reactivation'] = behaviour['reactivation'] - 0.5
        behaviour['intercept'] = 1

        pred_cols = ['reactivation']
        behaviour[pred_cols] = behaviour[pred_cols].apply(lambda x: scale(x))

        dm = behaviour[['intercept'] + pred_cols].values.astype(float)

        # Epochs    
        phase = 'planning'
        epochs = mne.read_epochs(os.path.join(data_dir, 'preprocessing', 'sub-{0}'.format(subject), 'task', 
                                'sub-{0}_ses-01_task-AversiveLearningReplay_run-{1}_proc_ICA{2}-epo.fif.gz'.format(subject, phase, downsample_string)))
        epochs.pick_types(meg=True, ref_meg=False)

        # Select generalisation trials
        epochs = epochs[behaviour_raw['trial_type'] == 1]

        # Filter and run beamforming
        # bands = {'theta': (4, 8), 'gamma': (30, None), 'unfiltered': (None, None)}
        bands = {'theta': (4, 8), 'low_gamma': (30, 60), 'high_gamma': (120, 199), 'unfiltered': (None, None)}
        # bands = {'unfiltered': (None, None)}

        for band, freqs in bands.items():

            print("Frequency band = {0}, {1} - {2}".format(band, freqs[0], freqs[1]))

            epochs_resampled = epochs.copy()
            print("Epochs sfreq = {0}".format(epochs_resampled.info['sfreq']))

            # Downsample to save time
            if freqs[1] is not None and freqs[1] < 199:
                resample_freq = np.max([freqs[1] * 2 + 10, 100])
                epochs_resampled = epochs_resampled.resample(resample_freq)

            # Restrict to band
            epochs_filtered = epochs_resampled.filter(*freqs)
            data_cov = mne.compute_covariance(epochs_filtered, tmin=0, tmax=None,
                                            method='empirical')

            # Make beamformer
            fwd = mne.make_forward_solution(epochs_filtered.info, trans=trans, src=src, bem=bem, eeg=False)
            filters = make_lcmv(epochs_filtered.info, fwd, data_cov, reg=0.05, pick_ori='max-power', weight_norm='unit-noise-gain', rank=None)

            # Apply beamforming
            stc_epochs = apply_lcmv_epochs(epochs_filtered, filters, max_ori_out='signed')

            # Save source estimates
            if not os.path.exists(os.path.join(output_dir, 'beamforming/{0}_filtered_source_estimates/{1}/planning'.format(band, downsample_string))):
                os.makedirs(os.path.join(output_dir, 'beamforming/{0}_filtered_source_estimates/{1}/planning'.format(band, downsample_string)))
            joblib.dump(stc_epochs, os.path.join(output_dir, 'beamforming/{0}_filtered_source_estimates/{1}/planning'.format(band, downsample_string), 
                                                            'sub-{0}_{1}_filtered-source-estimates'.format(subject, band)))

            if band != 'unfiltered':

                # Get amplitude & phase
                stcs_amplitude, stcs_phase = get_amplitude_phase(stc_epochs)
                del stc_epochs

                # Save
                if not os.path.exists(os.path.join(output_dir, 'beamforming/{0}_amplitude_source_estimates/{1}/planning'.format(band, downsample_string))):
                    os.makedirs(os.path.join(output_dir, 'beamforming/{0}_amplitude_source_estimates/{1}/planning'.format(band, downsample_string)))
                joblib.dump(stcs_amplitude, os.path.join(output_dir, 'beamforming/{0}_amplitude_source_estimates/{1}/planning'.format(band, downsample_string), 
                                                                'sub-{0}_{1}-amplitude-source-estimates'.format(subject, band)))

                if not os.path.exists(os.path.join(output_dir, 'beamforming/{0}_phase_source_estimates/{1}/planning'.format(band, downsample_string))):
                    os.makedirs(os.path.join(output_dir, 'beamforming/{0}_phase_source_estimates/{1}/planning'.format(band, downsample_string)))
                joblib.dump(stcs_phase, os.path.join(output_dir, 'beamforming/{0}_phase_source_estimates/{1}/planning'.format(band, downsample_string), 
                                                                'sub-{0}_{1}-phase-source-estimates'.format(subject, band)))

                # GLM
                res_amplitude = linear_regression(stcs_amplitude, dm, ['intercept'] + pred_cols)
                res_phase = linear_regression(stcs_phase, dm, ['intercept'] + pred_cols)

                # Create output directory
                for measure in ['phase', 'amplitude']:
                    out_dir = os.path.join(output_dir, 'beamforming/reactivation_regressions/planning/{0}_{1}'.format(band, measure))
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                # Standardise and save
                standardise_betas(res_amplitude, pred_cols, os.path.join(output_dir, 'beamforming/reactivation_regressions/planning/{0}_amplitude'.format(band), 'sub-{0}_source-space-regression-amplitude_{2}__{1}'), subject, downsample_string)
                standardise_betas(res_phase, pred_cols, os.path.join(output_dir, 'beamforming/reactivation_regressions/planning/{0}_phase'.format(band), 'sub-{0}_source-space-regression-phase_{2}__{1}'), subject, downsample_string)
            
                del res_amplitude
                del res_phase
                del stcs_amplitude
                del stcs_phase
                del epochs_filtered
                del epochs_resampled

    else:
        raise ValueError("{0} is an invalid value for phase".format(args.phase))