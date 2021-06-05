import mne
import os
import argparse
import numpy as np
import argparse
import joblib
from pactools import Comodulogram


def extract_hippocampus(sub, phase):
    theta_stc = joblib.load('../data/derivatives/beamforming/unfiltered_filtered_source_estimates/_400hz/{0}/{1}_unfiltered_filtered-source-estimates'.format(phase, sub))
    for i in theta_stc:
        i.vertices = [i.vertices]
    
    extracted = mne.extract_label_time_course(theta_stc, labels=(r'/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/fsaverage/mri/aparc.a2009s+aseg.mgz', ['Left-Hippocampus', 'Right-Hippocampus']), src=src, allow_empty=True, verbose='ERROR')
    
    return extracted



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject_idx")  # Index of the subjec to run
    args = parser.parse_args()
#
    subject_ids = [str(i).zfill(3) for i in range(1, 29)]
    subject = 'sub-' + subject_ids[int(args.subject_idx) - 1]

    print("RUNNING SUBJECT {0}".format(subject))
    print("Current working directory = {0}".format(os.getcwd()))


    os.chdir('/central/scratch/tobywise/data')

    fs_dir = '/central/groups/mobbslab/toby/old/meg/data/derivatives/mri/fsaverage'
    src = os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
    src = mne.read_source_spaces(src)


    n_jobs = len(os.sched_getaffinity(0))

    extracted_outcome = extract_hippocampus(subject, 'outcome')
    extracted_outcome = np.stack(extracted_outcome)

    estimator = Comodulogram(400, low_fq_range=np.linspace(4, 8, 30), high_fq_range=np.linspace(30, 200, 60), 
                                method='duprelatour', n_surrogates=100, n_jobs=n_jobs, random_state=123)
    estimator.fit(extracted_outcome[:, 0, int(1.2 * 400):])

    out_dir = '../data/derivatives/pac/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    joblib.dump(estimator, os.path.join(out_dir, '{0}_pac_estimator_2'.format(subject)))

