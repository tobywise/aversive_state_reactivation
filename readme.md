# Model-based aversive learning in humans is supported by preferential task state reactivation

## Toby Wise, Yunzhe Liu, Fatima Chowdhury & Raymond J. Dolan

This repository contains code used for the analyses in the above paper. 

### Notebooks

The `notebooks/templates` directory contains template parameterised Jupyter notebooks that are run using Papermill. These are used for some of the primary analyses.

### Code

The `/code` directory contains general analysis code, plus scripts that can be used to run the template notebooks. To run the primary analyses, these should be run as follows:

1. `/code/run_preproc_temp_gen.py` - this runs preprocessing and temporal generalisation analyses.
2. `/code/run_sequenceness.py` - this runs the classifier training for the sequenceness analysis, and then the analysis itself.

Some of these analyses rely on extensions of the decoding methods imolemented in MNE, which are included in `/code/sliding_window_classifiers.py`. These extend MNE's classifiers to be used with sliding windows.

For inference with the sequenceness data, the script `code/run_lgpr.py` is used to fit latent gaussian process regression (LGPR) models. The LGPR code itself is contained within `/code/gp_functions.py`.

### Settings

Various settings for the analysis pipeline are included as JSON files in `/settings/`.

### Dependencies

This code relies heavily on MNE. However, some scripts require different versions (new versions became available during analysis, but these broke some earlier scripts). Most scripts will run with v0.21.0, but some require v0.20.6 and will raise an error if this is not installed. In addition, the version of 0.20.6 must be [this fork](https://github.com/tobywise/mne-python/tree/v0.20.6_modified) as it includes additional beamforming capabilities.