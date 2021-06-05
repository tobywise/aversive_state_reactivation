import papermill as pm
import argparse
import json
import os

#######################################################################
# RUNS PREPROCESSING AND TEMPORAL GENERALISATION FOR A SINGLE SUBJECT #
#######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("subject_idx")  # Index of the subjec to run
    args = parser.parse_args()

    subject_ids = [str(i).zfill(3) for i in range(1, 29)]
    session_id = subject_ids[int(args.subject_idx) - 1]

    print("RUNNING SUBJECT {0}".format(session_id))
    print("Current working directory = {0}".format(os.getcwd()))

    # Load settings
    with open('settings/preproc_temp_gen_settings.json', 'r') as f:
        settings = json.load(f)
    

    # Create directories for notebook outputs
    if not os.path.exists('notebooks/outputs/preprocessing'):
        os.makedirs('notebooks/outputs/preprocessing')
    if not os.path.exists('notebooks/outputs/temporal_generalisation'):
        os.makedirs('notebooks/outputs/temporal_generalisation')

    # Run preprocessing notebook
    pm.execute_notebook(
        r'notebooks/templates/preprocessing_template.ipynb',
        r'notebooks/outputs/preprocessing/sub-{0}_preprocessing.ipynb'.format(session_id),
        kernel_name='mne2', start_timeout=10000,
        parameters = dict(session_id=session_id, 
                        maxwell=settings['preprocessing']['use_maxwell'], # If true, use maxwell filtering to clean data
                        eye_tracking = settings['preprocessing']['use_eye_tracking'],
                        filter_low=settings['preprocessing']['filter_low'], # Band pass lower freq
                        filter_high=None, # Band pass upper freq
                        blink_components=None)
    )

    # Run temporal generalisation notebook
    pm.execute_notebook(
        r'notebooks/templates/temporal_generalisation_template.ipynb',
        r'notebooks/outputs/temporal_generalisation/sub-{0}_temporal_generalisation.ipynb'.format(session_id),
        kernel_name='mne2', start_timeout=10000,
        parameters = dict(data_dir='data',
                        session_id=session_id,
                        n_iter_search=settings['temporal_generalisation']['param_optimisation_n_iter'],  # Number of iterations of the random search parameter optimisation procedure
                        pca_n_components=settings['temporal_generalisation']['pca_n_components'],  # Number of components used for PCA prior to classification
                        classifier_regularisation=settings['temporal_generalisation']['classifier_regularisation'],  # Type of regularisation to use in the classifier
                        param_optimisation_cv=settings['temporal_generalisation']['param_optimisation_cv']  # Number of CV folds to use in evaluating the classifier
        )
    )