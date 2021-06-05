import pymc3 as pm
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import scale
import theano.tensor as T

def make_GP_function(name, mean, covariance, x, n_subs=None, variance=None):
    """Generates a latent GP function using PyMC3. This functions as a prior, like any other PyMC3 random variable.

    Any PyMC3 mean and covariance functions can be used. Variance is optional, but should be provided as a float.

    X represents the X value of the function, which will typically be time.
    
    Args:
        name (str): Name of the function
        covariance: PyMC3 covariance function
        mean: PyMC3 mean function
        x (np.ndarray): X variable for GP function, normally time
        variance (float, optional): Optional variance for covariance function. This is coded as variance**2 * covariance. Defaults to None.
    
    Returns:
        [pm.gp.Latent.prior]: Latent GP prior function
    """


    if variance is not None:
        covariance = variance**2 * covariance
    else:
        covariance = covariance

    latent_gp = pm.gp.Latent(mean_func=mean, cov_func=covariance)

    # No subjects - just generate a single GP
    if n_subs is None:
        f = latent_gp.prior('Latent_GP__{0}'.format(name), x)
    
    # If we have multiple subjects, create GP for each and concatenate them.
    # This assumes the same GP for every subject as it reduces computational burden massively
    # If we assume different GPs (with or without same kernel), we have to invert a much bigger matrix
    else:
        sub_fs = []
        for n in range(n_subs):
            sub_fs.append(latent_gp.prior('Latent_GP__{0}__sub{1}'.format(name, n), x))
        f = T.concatenate(sub_fs)

    return f, latent_gp

def make_hierarchical_GP_function(name, mean, group_covariance, subject_covariance, x, n_subs, group_variance=None, subject_variance=None):
    """Creates hierarchical latent GP functions, one for the group level effect and one for the subject level effect.

    Assumes a single group mean but different group and subject covariances (and optional covariances)
    
    Args:
        name (str): Name of the function
        mean: PyMC3 mean function
        group_covariance: PyMC3 covariance function for the group GP
        subject_covariance: PyMC3 covariance function for the subject GP
        x (np.ndarray): X variable for GP function, normally time
        group_variance (float, optional): Optional variance for group covariance function. This is coded as variance**2 * covariance. Defaults to None.
        subject_variance (float, optional): Optional variance for subject covariance function. This is coded as variance**2 * covariance. Defaults to None.
    
    Returns:
        [pm.gp.Latent.prior, pm.gp.Latent.prior]: Latent GP functions for the group and the subject
    """

    f_group, gp_group = make_GP_function(name + '___group', mean, group_covariance, x, variance=group_variance)

    # Subject-level GP - hierarchy is represented by using the group function as the mean for the subject function
    # f_subject = make_GP_function(name + '__subject', pm.gp.mean.Constant(T.tile(f_group, n_subs)), subject_covariance, x, variance=subject_variance)
    f_subject, gp_subject = make_GP_function(name + '__subject', pm.gp.mean.Constant(f_group), subject_covariance, x, n_subs=n_subs, variance=subject_variance)
    # f_subject = make_GP_function(name + '__subject', pm.gp.mean.Zero(), subject_covariance, x, variance=subject_variance)

    return f_group, f_subject, gp_group, gp_subject

def check_covariance_function_params(covariance_functions, covariance_function_params, covariance_variance, n_preds):
    """Checks that covariance function parameters are passed in the correct format
    
    Args:
        covariance_functions (list): List of pm.gp.cov objects
        covariance_function_params (list): List of dicts representing parameters passed to the covariance function
        covariance_variance (list): List of dicts representing variance parameters
    """

    # Check format
    assert isinstance(covariance_functions, list), 'Covariance functions should be a list'
    assert isinstance(covariance_function_params, list), 'Covariance functions should be a list'
    assert isinstance(covariance_variance, list), 'Covariance functions should be a list'
    
    # Check lengths
    assert len(covariance_functions) == n_preds, 'Number of covariance functions must equal number of predictors'
    assert len(covariance_function_params) == n_preds, 'Number of covariance function parameters must equal number of predictors'
    assert len(covariance_variance) == n_preds, 'Number of covariance function variance parameters must equal number of predictors'

    # Check that we have covariance functions
    for i in covariance_functions:
        if not pm.gp.cov.ExpQuad.__module__ == 'pm.gp.cov':
            raise TypeError('Covariance functions must be a pm.gp.cov instance')

    # Check that covarariance function parameters are provided correctly
    for i in covariance_function_params:
        if not isinstance(i, dict):
            raise TypeError('Covariance function parameters should be provided as dicts')
        for v in i.values():
            if not 'dist' in v or not 'dist_params' in v:
                raise AttributeError('Covariance function parameters must pass each parameter as a dict containing dist and dist_params keys')

    # Check that variance parameters are correct
    for i in covariance_variance:
        if not isinstance(i, dict):
            raise TypeError('Variance parameters should be provided as dicts')
        if not 'dist' in i or not 'dist_params' in i:
            raise AttributeError('Variance parameters must pass each parameter as a dict containing dist and dist_params keys')

    return covariance_functions, covariance_function_params, covariance_variance


class LatentGPRegression():

    def __init__(self, formula, data, time_var='time', subject_identifier='Subject', trial_var='trial_number', scale_x=False,
                 subject_covariance_functions=None, group_covariance_functions=None, 
                 subject_covariance_parameters=None, group_covariance_parameters=None,
                 subject_covariance_variance=None, group_covariance_variance=None):
        """Hierarchical latent Gaussian process regression class
        
        Args:
            formula (string): R-style format (doesn't handle interactions or anything complicated)
            data (pd.DataFrame): Dataframe containing data to fit the model to
            time_var (string): Column in the dataframe representing time (or whatever is the X axis for the GP)
            subject_identifier (str, optional): Column representing subject IDs. Defaults to subject_identifier.
            trial_var (str, optional): Column representing trial number. Defaults to 'trial_number'.
            scale_x (bool, optional): Whether to scale the X data - may not work properly. Defaults to False.
            subject_covariance_functions (list, optional): List of subject-level pm.gp.cov covariance functions, one per predictor. Defaults to None.
            group_covariance_functions (list, optional): List of group-level pm.gp.cov covariance functions, one per predictor. Defaults to None.
            subject_covariance_parameters (list, optional): List of parameters for the subject-level covariance function. Each item in the list should be a dictionary with keys representing the parameter and values as dictionaries representing PyMC3 parameter. Defaults to None.
            group_covariance_parameters (list, optional): List of parameters for the group-level covariance function. Each item in the list should be a dictionary with keys representing the parameter and values as dictionaries representing PyMC3 parameter. Defaults to None.
            subject_covariance_variance (list, optional): List of variance parameters for the subject-level covariance parameters. Each item should be a dict with two keys, dist and dist params, representing a PyMC3 variable and its parameters. Defaults to None.
            group_covariance_variance (list, optional): List of variance parameters for the group-level covariance parameters. Each item should be a dict with two keys, dist and dist params, representing a PyMC3 variable and its parameters. . Defaults to None.
        
        Raises:
            ValueError: [description]
        """

        self.raw_data = data.copy()

        # Parse formula
        if not '~' in formula:
            raise ValueError("Formula doesn't look right, it should contain ~")

        # Get predictors
        self.predictor_vars = ['intercept' if i == '1' else i for i in re.findall(r"[\w:]+", formula)[1:]]

        # Get y var
        self.y_var = re.findall(r"[\w:]+", formula)[0]

        # Sort out data
        # Time - needs to be 2D - time X subjects (trials not used here as we have one beta array representing effect across all trials)
        self.time = pd.merge(self.raw_data.groupby([subject_identifier, time_var]).mean().reset_index()[[time_var, subject_identifier]], 
                             self.raw_data.groupby([subject_identifier]).mean().reset_index().reset_index()[['index', subject_identifier]].rename(columns={'index': 'Subject_pos'}), 
                             on=subject_identifier)[[time_var, 'Subject_pos']].values

        # Useful numbers
        self.n_tp = len(np.unique(self.time[:, 0]))
        self.n_subs = len(np.unique(self.time[:, 1]))
        self.n_preds = len(self.predictor_vars)
        self.n_trials = len(self.raw_data[trial_var].unique())

        # Change format of time variable - make it usable by GPs
        self.time = np.unique(self.time[:, 0])[np.newaxis, :].T

        # Create interaction columns
        for pred in self.predictor_vars:
            if ':' in pred:
                self.raw_data[pred] = self.raw_data[pred.split(':')].product(axis=1)
        
        # Scale (optional)
        if scale_x:
            preds = [i for i in self.predictor_vars if not 'intercept' in i]
            for sub in self.raw_data[subject_identifier].unique():
                self.raw_data.loc[self.raw_data[subject_identifier] == sub, preds] = self.raw_data.loc[self.raw_data[subject_identifier] == sub, preds].apply(lambda x: scale(x))

        # Pivot data - useful for things but takes a while
        pivoted_data = self.raw_data.set_index([subject_identifier, time_var]).pivot(columns=trial_var).reset_index()

        # X data = (n preds, n sub * n timepoints, n trials)
        # Add intercept first
        x_data = []

        for pred in self.predictor_vars:
            if pred == 'intercept':
                x_data.append(np.ones((self.n_subs * self.n_tp, self.n_trials)))
            else:
                pred_data = pivoted_data[pred].values
                x_data.append(pred_data)
        self.x_data = np.stack(x_data)

        #Y data
        self.y_data = pivoted_data[self.y_var].values

        # Covariance functions

        # Subject level
        if subject_covariance_functions is not None:
            
            # User-assigned covariance functions
            self.subject_covariance_functions, self.subject_covariance_parameter, \
                self.subject_covariance_variance = check_covariance_function_params(subject_covariance_functions, subject_covariance_parameters, subject_covariance_variance, self.n_preds)

        else:
            # ExpQuad for all, with gamma priors on everything
            self.subject_covariance_functions = [pm.gp.cov.ExpQuad] * self.n_preds 

            # Covariance function parameters take the form of a dictionary
            # Keys = parameters of the covariance function
            # Values = a dictionary representing a PyMC3 random variable, to be used as the prior. Must have two keys, dist and dist_params
            self.subject_covariance_parameters = [dict(ls=dict(dist=pm.Gamma, dist_params=dict(alpha=3, beta=1)))] * self.n_preds 

            # Covariance variance parameters
            self.subject_covariance_variance = [dict(dist=pm.Gamma, dist_params=dict(alpha=3, beta=5))] * self.n_preds 

        # Group level
        if group_covariance_functions is not None:

            # User-assigned covariance functions
            self.group_covariance_functions, self.group_covariance_parameter, \
                self.group_covariance_variance = check_covariance_function_params(group_covariance_functions, group_covariance_parameters, group_covariance_variance, self.n_preds)

        else:
            
            # ExpQuad for all, with gamma priors on everything
            self.group_covariance_functions = [pm.gp.cov.ExpQuad] * self.n_preds 

            # Covariance function parameters take the form of a dictionary
            # Keys = parameters of the covariance function
            # Values = a dictionary representing a PyMC3 random variable, to be used as the prior. Must have two keys, dist and dist_params
            self.group_covariance_parameters = [dict(ls=dict(dist=pm.Gamma, dist_params=dict(alpha=3, beta=1)))] * self.n_preds 

            # Covariance variance parameters
            self.group_covariance_variance = [dict(dist=pm.Gamma, dist_params=dict(alpha=3, beta=5))] * self.n_preds

        # PyMC3 model
        self.model = pm.Model()
        self.built = False
        self.latent_gps = dict()  # Used to store GP variables


    def build(self):

        with self.model:

            y = np.zeros_like(self.x_data)

            # ERROR
            σ = pm.HalfCauchy("σ", beta=2)    

            # LATENT GPS
            for n, pred in enumerate(self.predictor_vars):

                # Constant
                constant = pm.Normal('constant_{0}'.format(pred), 0, 5)
                mean = pm.gp.mean.Constant(constant)

                ### GROUP COVARIANCE ###
                # Get variance parameter
                if self.group_covariance_variance[n]['dist'] is None:
                    cov_group_variance = self.group_covariance_variance[n]['value']
                else:
                    cov_group_variance = self.group_covariance_variance[n]['dist'](name='{0}__group_variance'.format(pred), **self.group_covariance_variance[n]['dist_params'])

                # Get parameters for covariance function itself
                group_cov_params = {}
                for k, v in self.group_covariance_parameters[n].items():
                    group_cov_params[k] = v['dist'](name='{0}__group_{1}'.format(pred, k), **v['dist_params'])
                
                # Create covariance 
                cov_group = self.group_covariance_functions[n](input_dim=1, **group_cov_params)

                ### SUBJECT COVARIANCE ###
                # Get variance parameter
                if self.subject_covariance_variance[n]['dist'] is None:
                    cov_subject_variance = self.subject_covariance_variance[n]['value']
                else:
                    cov_subject_variance = self.subject_covariance_variance[n]['dist'](name='{0}__subject_variance'.format(pred), **self.subject_covariance_variance[n]['dist_params'])

                # Get parameters for covariance function itself
                subject_cov_params = {}
                for k, v in self.subject_covariance_parameters[n].items():
                    subject_cov_params[k] = v['dist'](name='{0}__subject_{1}'.format(pred, k), **v['dist_params'])  # Second parameter is just to set cross-subject covariance to zero
                
                # Create covariance 
                cov_subject = self.subject_covariance_functions[n](input_dim=1, **subject_cov_params)
                
                ## GET LATENT FUNCTIONS ##
                _, f_subject, gp_group, gp_subject = make_hierarchical_GP_function(pred, mean, cov_group, cov_subject, self.time, self.n_subs, cov_group_variance, cov_subject_variance)
                self.latent_gps['{0}__gp_group'.format(pred)] = gp_group
                self.latent_gps['{0}__gp_subject'.format(pred)] = gp_subject

                ## REGRESSION
                y += self.x_data[n, ...] * f_subject[:, np.newaxis]

            # LIKELIHOOD
            likelihood = pm.Normal('likelihood', y, σ, observed=self.y_data)

        self.built = True

    def fit(self, method='advi', **kwargs):

        if not self.built:
            self.build()

        if method.lower() == 'advi':
            with self.model:
                self.approx = pm.fit(**kwargs)
        
        elif method.lower() == 'mcmc':
            with self.model:
                self.trace = pm.sample(**kwargs)

    # def predict(self, time):
        
    #     with self.model:


    def _reset(self):
        self.built = False
        self.model = pm.Model()

    def aaa(self):
        self.bu



