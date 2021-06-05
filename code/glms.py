import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from tqdm import tqdm
from scipy.stats import zscore

def check_dimensions(sequenceness, phase, measure, n_trials, n_lags, n_arms):
    if sequenceness.ndim != 3:
        raise AttributeError("{0} {1} sequenceness should have 3 dimensions, found {2}".format(phase, measure, sequenceness.ndim))
    if sequenceness.shape[0] != n_trials:
        raise AttributeError('Too few trials in {0} {1} sequenceness, expected {2}, found {3}'.format(phase, measure, n_trials, sequenceness.shape[0]))
    if sequenceness.shape[1] != n_lags:
        raise AttributeError("Too few lags in {0} {1} sequenceness, expected {2}, found {3}".format(phase, measure, n_lags, sequenceness.shape[1]))
    if sequenceness.shape[2] != n_arms:
        raise AttributeError("Too few arms in {0} {1} sequenceness, expected {2}, found {3}".format(phase, measure, n_arms, sequenceness.shape[2]))

def check_missing(sequenceness, phase, measure):

    if np.any(np.isnan(sequenceness)):
        raise ValueError("{0} {1} sequenceness contains NaNs".format(phase, measure))
    if np.any(np.isinf(sequenceness)):
        raise ValueError("{0} {1} sequenceness contains infs".format(phase, measure))        
    
def get_chosen_unchosen(sequenceness, behaviour, exclude_outcome_only, type='shown', arm_idx=(1, 2)):

    """
    Converts arm 1/2 to chosen/unchosen. 
    TODO check that unchosen/chosen are the correct way round
    TODO chosen moves on next trial
    
    Args:
        sequenceness: Array of sequenceness, shape (n_subjects, n_trials, 3) where the last dimension represents (both arms, arm 1, arm 2)
        behaviour: Behavioural data
        exclude_outcome_only: Exclude outcome only trials
        type: 'shown' or 'chosen' - if 'shown' uses the shown move in failed trials
        arm_idx: Tuple containing indices of array to be used as "arms", default = (1, 2)

    Returns:
        Modified version of the input array, where the last dimension now represents (both arms, chosen arm, unchosen arm)
    """

    if exclude_outcome_only:
        behaviour = behaviour[behaviour['trial_type'] == 0].copy()

    # Sequenceness should be n_trials X n_lags X n_arms
    chosen = sequenceness[..., arm_idx[0]].copy()
    unchosen = sequenceness[..., arm_idx[1]].copy()

    # Whether we're using the move that was shown or the move they chose
    if type == 'chosen':
        column = 'chosen_move'
    elif type == 'shown':
        column = 'shown_move'

    chosen[behaviour[column] == 1, :] = sequenceness[behaviour[column] == 1, :, arm_idx[1]]
    unchosen[behaviour[column] == 1, :] = sequenceness[behaviour[column] == 1, :, arm_idx[0]]

    sequenceness[..., arm_idx[0]] = chosen
    sequenceness[..., arm_idx[1]] = unchosen

    return sequenceness


class Sequenceness(object):

    def __init__(self, sequenceness, behaviour, subject, n_trials=100, chosen=True, accuracy=None):

        """
        Args:
            sequenceness: List of (phase, sequenceness dictionary pickle, expected shape, exclude outcome only trials, chosen variable) tuples
            behaviour: Either pandas dataframe or path to csv file containing behavioural data
            n_trials: Number of trials in the behavioural data
            chosen: If true, changes the third dimension in the sequenceness data to chosen/unchosen arm rather than arm 1/2
        """
        
        # Subject ID
        self.subject = subject
        
        # Accuracy
        self.accuracy = accuracy

        # N trials
        self.n_trials = n_trials

        # Load behaviour
        if isinstance(behaviour, str):
            self.behaviour = pd.read_csv(behaviour)
        elif isinstance(behaviour, pd.DataFrame):
            self.behaviour = behaviour

        # Remove trials that need to be removed
        self.behaviour = self.behaviour[~self.behaviour['trial_number'].isnull()]

        # Calculate useful things
        self.behaviour['shown_move'] = self.behaviour['State_3_shown'] - 5  # Get the state that was show on each trial
        self.behaviour['chosen_move'] = self.behaviour['State_1_chosen'] - 1  # Get the state that the subject chose (coded as 1/2 originally)

        # Check behaviour shapes etc
        if len(self.behaviour) != n_trials:
            raise AttributeError("Behavioural data should have {0} trials, found {1}".format(n_trials, len(self.behaviour)))
        if np.any(np.diff(self.behaviour['trial_number']) != 1):
            raise ValueError("Trial numbers in behavioural data don't increase linearly, some trials may be missing")

        # Load sequenceness
        self.sequenceness = dict()

        # Store whether we've swapped arms around for chosen/unchosen
        self.chosen = dict()

        for seq in sequenceness:
            phase = seq[0]
            data = seq[1]
            shape = seq[2]
            exclude_outcome_only = seq[3]
            chosen = seq[4]

            # Read in pickle
            self.sequenceness[phase] = data
            self.chosen[phase] = dict()

            # Go through measures (i.e. forwards/backwards/difference)
            # print(self.sequenceness)
            for measure in self.sequenceness[phase].keys():
                # Check sequenceness shapes and missing data
                check_dimensions(self.sequenceness[phase][measure] , phase, measure, *shape)
                check_missing(self.sequenceness[phase][measure], phase, measure)

                # Change to chosen / unchosen
                if chosen is not None:
                    self.chosen[phase][measure] = True
                    self.sequenceness[phase][measure] = get_chosen_unchosen(self.sequenceness[phase][measure], self.behaviour, exclude_outcome_only, type=chosen)
                else:
                    self.chosen[phase][measure] = False


    def __repr__(self):
        return '<' + self.subject + ' sequenceness data | {0} trials>'.format(self.n_trials)

    def trialwise(self, phase, measure='difference', exclude_outcome_only=False, predictor_shifts=(), zscore_sequenceness=False):

        """
        Produces a trialwise dataframe of behaviour and sequenceness data
        
        Returns:
            phase (str): Task phase
            measure (str): Sequenceness measure (e.g. difference)
            exclude_outcome_only (bool): Exclude outcome only trials
            predictor_shifts (list): List of dictionaries containing two keys, 'name' and 'shift'. Name specifies the column in the behavioural dataframe, 
        """

        # Copy the behavioural data to avoid changing the original
        behaviour = self.behaviour.copy()
        # Copy sequenceness so we don't affect the original data
        sequenceness = self.sequenceness[phase][measure].copy()

        # Exclude outcome only trials from behaviour if needed
        if exclude_outcome_only:
            if sequenceness.shape[0] == len(behaviour):
                sequenceness = sequenceness[behaviour.trial_type == 0]
            behaviour = behaviour[behaviour.trial_type == 0]
        # Get data for each arm
        seq_dfs = []

        for arm in range(sequenceness.shape[2]):
            # Convert sequenceness to pandas dataframe with each lag for each arm as a different column
            if zscore_sequenceness:
                # print(sequenceness[..., arm])
                seq_data = zscore(sequenceness[..., arm], axis=1)
                seq_data[np.isnan(seq_data)] = 0
                # print(seq_data)
            else:
                seq_data = sequenceness[..., arm]
            seq_dfs.append(pd.DataFrame(seq_data, columns=['arm_{0}__lag_'.format(arm) + str(i) for i in range(sequenceness.shape[1])]))


        # Concatenate behaviour and sequenceness
        behav_seq = pd.concat([behaviour.reset_index()] + [seq_df.reset_index() for seq_df in seq_dfs], axis=1)

        # Shift predictors if needed
        for pred in predictor_shifts:
            behav_seq.loc[:, pred['name']] = np.roll(behav_seq[pred['name']],
                                            pred['shift'])  # shift any predictors from the previous trial
            behav_seq.loc[:, pred['name']][pred['shift']] = np.nan  # Used to drop trials that get shifted to weird places

        # Drop trials that we've lost when shifting predictors
        behav_seq = behav_seq.dropna(axis=0, subset=[i['name'] for i in predictor_shifts])

        behav_seq['trial_number_new'] = np.arange(len(behav_seq))

        return behav_seq

        
def create_df_dict(predictors, arms=['both', 'chosen', 'unchosen'], n_lags=40):

    """
    Creates a dictionary to hold results of GLM - later turned into a DataFrame

    Args:
        predictors: List of predictors

    Returns:
        Dictionary with keys (arm, lag, predictor, beta)
    
    """

    result_df = dict()
    result_df['lag'] = np.tile(np.repeat(np.arange(n_lags), len(predictors)), 1)
    result_df['predictor'] = []
    result_df['beta'] = []

    return result_df


def trialwise_glm(sequenceness, formula, phase, measure='difference', predictor_shifts=(), exclude_outcome_only=False, n_lags=40, zscore_sequenceness=False):
    """
    Runs a trialwise GLMs predicting replay on a given trial from behavioural predictors. This is run across
    all time lags to demonstrate where replay intensity is related to a variable of interest across trials.

    Args:
        sequenceness: Object of the Sequenceness class or a dataframe with behavioural data and sequenceness data
        formula: Formula of the form arm_n__lag ~ predictors, where n is the arm number of interest
        phase (str): Phase of the task. Accepts either 'rest' or 'planning'
        predictor_shifts (list): List of dictionaries containing two keys, 'name' and 'shift'. Name specifies the column in the behavioural dataframe, 
        shift specifies in which direction the values of this columnn should be shifted (e.g. moving back one trial to line up MEG data with behaviour 
        on the subsequent trial).
        exclude_outcome_only (bool): If true and using 'rest' phase, excludes outcome only trials
        n_lags: Number of time lags in the sequenceness data

    Returns:
        A dataframe containing beta values

    """

    # Check things
    if not isinstance(predictor_shifts, list) and not isinstance(predictor_shifts, tuple):
        raise TypeError("Predictors should be specified as a list of dictionaries")

    if not all([isinstance(i['name'], str) for i in predictor_shifts]):
        raise TypeError("Predictor list should contain dictionaries with keys 'name' and 'shift'")

    if not all([isinstance(i['shift'], int) for i in predictor_shifts]):
        raise TypeError("Predictor shift values should be specified as integers")

    # Stringify predictors
    # pred_string = stringify_predictors(predictors)
    predictor_names = re.findall('(?<=[~+] )[A-Za-z_:]+', formula)
    arm = re.search('arm_[0-9]', formula).group()
    formula = re.search('(?<=~ ).+', formula).group()

    # Dictionary to store coefficients, which will later be turned into a pandas DataFrame
    result_df = create_df_dict(predictor_names, arms=['both', 'chosen', 'unchosen'], n_lags=40)

    if isinstance(sequenceness, pd.DataFrame):
        behav_seq = sequenceness
    else:
        # Create a dataframe containing the necessary data
        behav_seq = sequenceness.trialwise(phase, measure, exclude_outcome_only, predictor_shifts, zscore_sequenceness=zscore_sequenceness)
        try:
            behav_seq[[i for i in predictor_names if not ':' in i]] = behav_seq[[i for i in predictor_names if not ':' in i]].astype(np.float64)
            behav_seq[[i for i in predictor_names if not ':' in i]] = scale(behav_seq[[i for i in predictor_names if not ':' in i]])
        except Exception as e:
            print(behav_seq)
            raise e

    # Run GLM across all time lags
    for lag in range(n_lags):
        model = smf.ols(formula='{0}__lag_{1} ~ {2}'.format(arm, lag, formula), data=behav_seq)
        res = model.fit()
        for p in predictor_names:
            result_df['predictor'].append(p)
            result_df['beta'].append(res.params[p])

    result_df = pd.DataFrame(result_df)
    result_df['Subject'] = sequenceness.subject

    return result_df

def trialwise_logistic(sequenceness, arms, phase, measure='difference', predictor_shifts=(), exclude_outcome_only=False, n_lags=40, zscore_sequenceness=False):
    """
    Runs a trialwise GLMs predicting choices on a given trial from replay. This is run across
    all time lags to find lags that predict choices

    Args:
        sequenceness: Object of the Sequenceness class or a dataframe with behavioural data and sequenceness data
        arms: List of arms to include (must have a length of 2 as the difference between arms is used as predictor)
        phase (str): Phase of the task. Accepts either 'rest' or 'planning'
        measure: Forwards, backwards, or difference
        n_lags: Number of time lags in the sequenceness data

    Returns:
        A dataframe containing beta values

    """

    # Check that arms = states rather than chosen/unchosen
    if sequenceness.chosen[phase][measure]:
        raise AttributeError("Sequenceness data has chosen/unchosen arms assigned")

    if len(arms) != 2:
        raise ValueError("Must provide two arms to use as predictors")

    # Stringify predictors
    # pred_string = stringify_predictors(predictors)
    predictor_names = ['sequenceness']

    # Dictionary to store coefficients, which will later be turned into a pandas DataFrame
    result_df = create_df_dict(predictor_names, arms=['both', 'chosen', 'unchosen'], n_lags=40)

    if isinstance(sequenceness, pd.DataFrame):
        behav_seq = sequenceness
    else:
        # Create a dataframe containing the necessary data
        behav_seq = sequenceness.trialwise(phase, measure, exclude_outcome_only=True, zscore_sequenceness=zscore_sequenceness)
        for lag in range(n_lags):
            behav_seq['difference__lag_{0}'.format(lag)] = behav_seq['arm_{0}__lag_{1}'.format(arms[0], lag)] - behav_seq['arm_{0}__lag_{1}'.format(arms[1], lag)]
    # Previous chosen move
    behav_seq['chosen_move_prev'] = (np.roll(behav_seq['chosen_move'], 1))
    behav_seq = behav_seq.dropna(axis=0, subset=['chosen_move_prev'])
    behav_seq['chosen_move_prev'] = zscore(behav_seq['chosen_move_prev'])

    print(behav_seq)

    # raise ValueError()
    # Run GLM across all time lags
    for lag in range(n_lags):
        model = sm.Logit.from_formula(formula='chosen_move ~ trial_number + chosen_move_prev + difference__lag_{0}'.format(lag), data=behav_seq, missing='drop')
        res = model.fit(method='bfgs', disp=0)
        result_df['predictor'].append('sequenceness')
        result_df['beta'].append(res.params['difference__lag_{0}'.format(lag)])
        # print(result_df)
        # print(res.summary())
        # raise ValueError()

    result_df = pd.DataFrame(result_df)
    result_df['Subject'] = sequenceness.subject

    return result_df, behav_seq

def trialwise_glm_reactivation(sequenceness, formula, phase, measure='difference', predictor_shifts=(), exclude_outcome_only=False, n_lags=40):
    """
    Runs a trialwise GLMs predicting replay on a given trial from behavioural predictors. This is run across
    all time lags to demonstrate where replay intensity is related to a variable of interest across trials.

    Args:
        sequenceness: Object of the Sequenceness class or a dataframe with behavioural data and sequenceness data
        formula: Formula of the form arm_n__lag ~ predictors, where n is the arm number of interest
        phase (str): Phase of the task. Accepts either 'rest' or 'planning'
        predictor_shifts (list): List of dictionaries containing two keys, 'name' and 'shift'. Name specifies the column in the behavioural dataframe, 
        shift specifies in which direction the values of this columnn should be shifted (e.g. moving back one trial to line up MEG data with behaviour 
        on the subsequent trial).
        exclude_outcome_only (bool): If true and using 'rest' phase, excludes outcome only trials
        n_lags: Number of time lags in the sequenceness data

    Returns:
        A dataframe containing beta values

    """

    # Check things
    if not isinstance(predictor_shifts, list) and not isinstance(predictor_shifts, tuple):
        raise TypeError("Predictors should be specified as a list of dictionaries")

    if not all([isinstance(i['name'], str) for i in predictor_shifts]):
        raise TypeError("Predictor list should contain dictionaries with keys 'name' and 'shift'")

    if not all([isinstance(i['shift'], int) for i in predictor_shifts]):
        raise TypeError("Predictor shift values should be specified as integers")

    # Stringify predictors
    # pred_string = stringify_predictors(predictors)
    predictor_names = re.findall('(?<=[~+] )[A-Za-z_:]+', formula)
    state = re.search('state_[0-9]', formula).group()
    formula = re.search('(?<=~ ).+', formula).group()

    # Dictionary to store coefficients, which will later be turned into a pandas DataFrame
    result_df = create_df_dict(predictor_names, arms=['both', 'chosen', 'unchosen'], n_lags=40)

    if isinstance(sequenceness, pd.DataFrame):
        behav_seq = sequenceness
    else:
        # Create a dataframe containing the necessary data
        behav_seq = sequenceness.trialwise(phase, measure, exclude_outcome_only, predictor_shifts)
        try:
            behav_seq[[i for i in predictor_names if not ':' in i]] = scale(behav_seq[[i for i in predictor_names if not ':' in i]])
        except Exception as e:
            print(behav_seq)
            raise e

    # Run GLM across all time lags
    for lag in range(n_lags):
        model = smf.ols(formula='{0}__lag_{1} ~ {2}'.format(arm, lag, formula), data=behav_seq)
        res = model.fit()
        for p in predictor_names:
            result_df['predictor'].append(p)
            result_df['beta'].append(res.params[p])

    result_df = pd.DataFrame(result_df)
    result_df['Subject'] = sequenceness.subject

    return result_df

class Reactivations(object):

    def __init__(self, reactivations, behaviour, subject, accuracy=None):

        """
        Args:
            reactivations: List of (phase, reactivation array, expected shape, exclude outcome only trials, chosen variable) tuples
            behaviour: Either pandas dataframe or path to csv file containing behavioural data
        """

        if not isinstance(reactivations, list):
            raise TypeError("Reactivations should be provided as a list")

        # Subject ID
        self.subject = subject
        
        # Accuracy
        self.accuracy = accuracy

        # Load behaviour
        if isinstance(behaviour, str):
            self.behaviour = pd.read_csv(behaviour)
        elif isinstance(behaviour, pd.DataFrame):
            self.behaviour = behaviour

        # Remove trials that need to be removed
        self.behaviour = self.behaviour[~self.behaviour['trial_number'].isnull()]

        # Calculate useful things
        self.behaviour['shown_move'] = self.behaviour['State_3_shown'] - 5  # Get the state that was show on each trial
        self.behaviour['chosen_move'] = self.behaviour['State_1_chosen'] - 1  # Get the state that the subject chose (coded as 1/2 originally)

        # Check behaviour shapes etc
        if np.any(np.diff(self.behaviour['trial_number']) != 1):
            raise ValueError("Trial numbers in behavioural data don't increase linearly, some trials may be missing")

        # Load reactivations
        self.reactivation = dict()

        for reactivation in reactivations:
            phase = reactivation[0]
            data = reactivation[1]
            shape = reactivation[2]
            exclude_outcome_only = reactivation[3]
            chosen = reactivation[4]

            # Read in pickle
            self.reactivation[phase] = np.load(data)

            # Check reactivation shapes and missing data
            if self.reactivation[phase].shape != shape:
                raise AttributeError("Reactivation for phase {0} is the wrong shape, expected {1} found {2}".format(phase, shape, self.reactivation[phase].shape))
            check_missing(self.reactivation[phase], phase, 'reactivations')

            # Change to chosen / unchosen
            if chosen is not None:
                self.reactivation[phase] = get_chosen_unchosen(self.reactivation[phase], self.behaviour, exclude_outcome_only, type=chosen, arm_idx=(5, 6))
                self.reactivation[phase] = get_chosen_unchosen(self.reactivation[phase], self.behaviour, exclude_outcome_only, type=chosen, arm_idx=(3, 4))
                self.reactivation[phase] = get_chosen_unchosen(self.reactivation[phase], self.behaviour, exclude_outcome_only, type=chosen, arm_idx=(1, 2))

    def __repr__(self):
        return '<' + self.subject + ' reactivation data>'

    def trialwise(self, phase, exclude_outcome_only=False, predictor_shifts=(), scale_cols=(), normalise=True):

        """
        Produces a trialwise dataframe of behaviour and sequenceness data
        
        Returns:
            phase (str): Task phase
            measure (str): Sequenceness measure (e.g. difference)
            exclude_outcome_only (bool): Exclude outcome only trials
            predictor_shifts (list): List of dictionaries containing two keys, 'name' and 'shift'. Name specifies the column in the behavioural dataframe, 
        """

        # Copy the behavioural data to avoid changing the original
        behaviour = self.behaviour.copy()
        # Copy sequenceness so we don't affect the original data
        reactivation = self.reactivation[phase].copy()

        # Exclude outcome only trials from behaviour if needed
        if exclude_outcome_only:
            if reactivation.shape[0] == len(behaviour):
                reactivation = reactivation[behaviour.trial_type == 0]
            behaviour = behaviour[behaviour.trial_type == 0]

        # Get mean reactivations
        mean_reactivation = reactivation.mean(axis=1)

        # Normalise data with respect to the overall mean for the trial
        if normalise:
            mean_reactivation /= mean_reactivation.mean(axis=0)
        
        # Get data for each arm
        reactivation_df = pd.DataFrame(mean_reactivation, columns=['state_' + str(i) for i in range(reactivation.shape[2])])

        # Concatenate behaviour and sequenceness
        behav_seq = pd.concat([behaviour.reset_index(), reactivation_df.reset_index()], axis=1)

        # Shift predictors if needed
        for pred in predictor_shifts:
            behav_seq.loc[:, pred['name']] = np.roll(behav_seq[pred['name']],
                                            pred['shift'])  # shift any predictors from the previous trial
            behav_seq.loc[:, pred['name']][pred['shift']] = np.nan  # Used to drop trials that get shifted to weird places

        # Drop trials that we've lost when shifting predictors
        behav_seq = behav_seq.dropna(axis=0, subset=[i['name'] for i in predictor_shifts])

        behav_seq['trial_number_new'] = np.arange(len(behav_seq))

        if len(scale_cols):
            behav_seq[scale_cols] = scale(behav_seq[scale_cols].astype(np.float64))

        return behav_seq

    


class GroupReactivation(object):

    def __init__(self, reactivations):

        """
        Class for representing a group of reactivation arrays

        # TODO generalise and inherit for sequenceness/reactivation?

        Args:
            reactivations: List of Reactivation instances

        """ 

        if not isinstance(reactivations, list):
            raise TypeError("Reactivations should be a list")

        # Get subjects
        self.subjects = [i.subject for i in reactivations]

        # Get behaviour
        self.behaviour = [i.behaviour for i in reactivations]

        # Get reactivations
        self.reactivation = dict()

        self.subject_reactivations = reactivations

        # Iterate over subjects
        for r in reactivations:
            # Add each phase to a list
            for phase, values in r.reactivation.items():
                if not phase in self.reactivation:
                    self.reactivation[phase] = []
                self.reactivation[phase].append(values)

        # Stack arrays
        for k, v in self.reactivation.items():
            self.reactivation[k] = np.stack(v)

    def trialwise_glm(self, formula, phase, exclude_outcome_only=False, scale_predictors=True, predictor_shifts=()):

        predictor_names = re.findall('(?<=[~+] )[A-Za-z_:]+', formula)

        scale_cols = []
        if scale_predictors:
            scale_cols = [p for p in predictor_names if not ':' in p]

        result_df = dict(subject=[], predictor=[], beta=[])

        for sub in tqdm(self.subject_reactivations):
            model = smf.ols(formula=formula, data=sub.trialwise(phase, exclude_outcome_only=exclude_outcome_only, 
                            scale_cols=scale_cols, predictor_shifts=predictor_shifts))
            res = model.fit()
            for p in predictor_names:
                result_df['predictor'].append(p)
                result_df['beta'].append(res.params[p])
                result_df['subject'].append(sub.subject)

        result_df = pd.DataFrame(result_df)

        return result_df

    def subject_summary(self, phase):

        summary = self.reactivation[phase].mean(axis=2).mean(axis=1)

        summary_df = pd.DataFrame(summary)
        summary_df.columns = ['State_' + str(i) for i in range(summary_df.shape[1])]

        summary_df['subject'] = self.subjects

        return summary_df

