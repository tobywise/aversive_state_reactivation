import papermill as pm
import argparse
import json
import os

##################################################################
# RUNS CLASSIFIER TRAINING AND SEQUENCENESS FOR A SINGLE SUBJECT #
##################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("subject_idx")  # Index of the subjec to run
    parser.add_argument("classifier_center_idx")
    args = parser.parse_args()

    subject_ids = [str(i).zfill(3) for i in range(1, 29)]
    session_id = subject_ids[int(args.subject_idx) - 1]

    print("RUNNING SUBJECT {0}, CLASSIFIER IDX = {1}".format(session_id, args.classifier_center_idx))
    print("Current working directory = {0}".format(os.getcwd()))

    # Load settings
    with open('settings/sequenceness_settings.json', 'r') as f:
        settings = json.load(f)
        
    # Create directories for notebook outputs
    if not os.path.exists('notebooks/outputs/sequenceness_classifier/classifer_idx_{0}'.format(args.classifier_center_idx)):
        os.makedirs('notebooks/outputs/sequenceness_classifier/classifer_idx_{0}'.format(args.classifier_center_idx))
    if not os.path.exists('notebooks/outputs/sequenceness_window/classifer_idx_{0}'.format(args.classifier_center_idx)):
        os.makedirs('notebooks/outputs/sequenceness_window/classifer_idx_{0}'.format(args.classifier_center_idx))


    # Run classification notebook
    pm.execute_notebook(
        r'notebooks/templates/sequenceness_classifier_template.ipynb',
        r'notebooks/outputs/sequenceness_classifier/classifer_idx_{0}/sub-{1}_sequenceness_classifier_idx_{0}.ipynb'.format(args.classifier_center_idx, session_id),
        kernel_name='mne2', start_timeout=10000,
        parameters = dict(session_id=str(session_id), 
                        classifier_window=settings['sequenceness_classifier']['classifier_window'], 
                        n_iter = settings['sequenceness_classifier']['param_optimisation_n_iter'],
                        n_pca_components=settings['sequenceness_classifier']['n_pca_components'], 
                        param_optimisation_cv=settings['sequenceness_classifier']['param_optimisation_cv'],
                        classifier_regularisation=settings['sequenceness_classifier']['classifier_regularisation'],
                        classifier_multiclass=settings['sequenceness_classifier']['classifier_multiclass'], 
                        confusion_matrix_cv=settings['sequenceness_classifier']['confusion_matrix_cv'],
                        classifier_center_idx = int(args.classifier_center_idx)
        )
    )

    # Run sequenceness notebook
    pm.execute_notebook(
        r'notebooks/templates/sequenceness_window_template.ipynb',
        r'notebooks/outputs/sequenceness_window/classifer_idx_{0}/sub-{1}_sequenceness_window_idx_{0}.ipynb'.format(args.classifier_center_idx, session_id),
        kernel_name='mne2', start_timeout=10000, log_output=False,
        parameters = dict(data_dir='data',
                        session_id=str(session_id),
                        window_width=settings['sequenceness']['window_width'], 
                        classifier_window=settings['sequenceness_classifier']['classifier_window'],
                        max_lag=settings['sequenceness']['max_lag'], 
                        classifier_center_idx = int(args.classifier_center_idx)
        )
    )


