import os
import mne
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter, GeneralizingEstimator, SlidingEstimator, cross_val_multiscore
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'code')
import argparse
from state_prediction import *
from sliding_window_classifiers import *
from sklearn.preprocessing import FunctionTransformer

np.random.seed(100)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("subject_idx")  # Index of the subjec to run
    args = parser.parse_args()

    subject_ids = [str(i).zfill(3) for i in range(1, 29)]
    session_id = subject_ids[int(args.subject_idx) - 1]
    
    output_dir = 'data/derivatives'  # Where the output data should go
    n_stim = 14  # Number of stimuli
    classifier_regularisation = 'l1'  # Type of regularisation to use, l1 or l2
    classifier_multiclass = 'ovr'  # Type of multi-class approach to use, ovr for one-vs-the-rest or multiclass
    cores = 2
    classifier_window = [-5, 6]  # Additional timepoints to use as features
    os.environ['OMP_NUM_THREADS'] = str(cores)


    localiser_epochs = mne.read_epochs(os.path.join(output_dir, 'preprocessing/sub-{0}/localiser'.format(session_id), 
                                        'sub-{0}_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz').format(session_id))

    # Get epoch data
    X_raw = localiser_epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times (exclude non MEG channels)
    y_raw = localiser_epochs.events[:, 2]  # Get event types

    # select events and time period of interest
    picks_meg = mne.pick_types(localiser_epochs.info, meg=True, ref_meg=False)
    event_selector = (y_raw < n_stim * 2 + 1)
    X_raw = X_raw[event_selector, ...]
    y_raw = y_raw[event_selector]
    X_raw = X_raw[:, picks_meg, :]

    times = localiser_epochs.times

    assert len(np.unique(y_raw)) == n_stim, "Found {0} stimuli, expected {1}".format(len(np.unique(y_raw)), n_stim)

    print("Number of unique events = {0}\n\nEvent types = {1}".format(len(np.unique(y_raw)),
                                                                    np.unique(y_raw)))


    ################################################
    # RUN ACCURACY TEST WITHOUT TEMPORAL EMBEDDING #
    ################################################

    # # Do PCA with 30 components
    pca = UnsupervisedSpatialFilter(PCA(30), average=False)
    pca_data = pca.fit_transform(X_raw)

    # CLASSIFIER
    # Logistic regression with L1 penalty, multi-class classification performed as OVR
    # Data is transformed to have zero mean and unit variance before being passed to the classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class=classifier_multiclass, C=0.1, penalty=classifier_regularisation, class_weight="balanced",
                                                            solver='saga', max_iter=1000, tol=0.3))

    # Try classifying at all time points with 5 fold CV
    time_decod = SlidingEstimator(clf, scoring='accuracy', n_jobs=1, verbose=False)
    scores = cross_val_multiscore(time_decod, pca_data, y_raw, cv=5, n_jobs=cores)

    # Mean scores across cross-validation splits
    mean_scores = np.mean(scores, axis=0)

    score_out_dir = os.path.join(output_dir, 'localiser_classifier_performance', 'time_scores')
    if not os.path.exists(score_out_dir):
        os.makedirs(score_out_dir)
    np.save(os.path.join(score_out_dir, 'sub-{0}_time_scores.npy').format(session_id), mean_scores)



    #############################################
    # RUN ACCURACY TEST WITH TEMPORAL EMBEDDING #
    #############################################

    # # Do PCA with 30 components
    pca = UnsupervisedSpatialFilter(PCA(30), average=False)
    pca_data = pca.fit_transform(X_raw)

    # Create a pipiline that combines PCA, feature augmentation, scaling, and the logistic regression classifier
    clf = make_pipeline(StandardScaler(), 
                        LogisticRegression(multi_class=classifier_multiclass, C=0.1, penalty=classifier_regularisation, 
                                          solver='saga', max_iter=1000, tol=0.3, class_weight="balanced"))

    # Try classifying at all time points with 5 fold CV
    time_decod = SlidingWindowEstimator(clf, np.sum(np.abs(classifier_window)), scoring='accuracy', n_jobs=1, verbose=False)
    scores_window = cross_val_multiscore(time_decod, pca_data, y_raw, cv=5, n_jobs=1)

    # Mean scores across cross-validation splits
    mean_scores_window = np.mean(scores_window, axis=0)

    score_out_dir = os.path.join(output_dir, 'localiser_classifier_performance', 'embedding_time_scores')
    if not os.path.exists(score_out_dir):
        os.makedirs(score_out_dir)
    np.save(os.path.join(score_out_dir, 'sub-{0}_embedding_time_scores.npy').format(session_id), mean_scores_window)

    ###############################
    # RUN TEMPORAL GENERALISATION #
    ###############################

    time_gen = GeneralizingEstimator(clf, scoring='accuracy', n_jobs=1,
                                    verbose=True)

    time_gen.fit(pca_data[..., times >= 0], y_raw)

    scores = cross_val_multiscore(time_gen, pca_data[..., times >= 0], y_raw, cv=5, n_jobs=2).mean(axis=0)

    score_out_dir = os.path.join(output_dir, 'localiser_classifier_performance', 'temporal_generalisation')
    if not os.path.exists(score_out_dir):
        os.makedirs(score_out_dir)
    np.save(os.path.join(score_out_dir, 'sub-{0}_temporal_generalisation_scores.npy').format(session_id), scores)