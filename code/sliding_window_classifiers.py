from utils import add_features
from mne.parallel import parallel_func
from mne.utils import ProgressBar, _validate_type, array_split_idx
from mne.decoding.base import _check_estimator
from mne.decoding.search_light import _fix_auc, _check_method, _sl_init_pred
from mne.utils import verbose
from mne.decoding.search_light import _gl_transform
from mne.decoding import SlidingEstimator
import numpy as np

class SlidingWindowEstimator(SlidingEstimator):
    
    def __init__(self, base_estimator, window_size, scoring=None, n_jobs=1,
                 verbose=None):  # noqa: D102
        _check_estimator(base_estimator)
        self._estimator_type = getattr(base_estimator, "_estimator_type", None)
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = verbose
        self.window_size = window_size

        _validate_type(self.n_jobs, 'int', 'n_jobs')
        
        
    def fit(self, X, y, **fit_params):
        """Fit a series of independent estimators to the dataset.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The training input samples. For each data slice, a clone estimator
            is fitted independently. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks)
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.
        **fit_params : dict of string -> object
            Parameters to pass to the fit method of the estimator.
        Returns
        -------
        self : object
            Return self.
        """
        self._check_Xy(X, y)
        self.estimators_ = list()
        self.fit_params = fit_params
        # For fitting, the parallelization is across estimators.
        parallel, p_func, n_jobs = parallel_func(_sl_window_fit, self.n_jobs,
                                                 verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        mesg = 'Fitting %s' % (self.__class__.__name__,)
        with ProgressBar(X.shape[-1], verbose_bool='auto',
                         mesg=mesg) as pb:
            estimators = parallel(
                p_func(self.base_estimator, split, y, pb.subset(pb_idx), window_size=self.window_size,
                       **fit_params)
                for pb_idx, split in array_split_idx(X, n_jobs, axis=-1))

        # Each parallel job can have a different number of training estimators
        # We can't directly concatenate them because of sklearn's Bagging API
        # (see scikit-learn #9720)
        self.estimators_ = np.empty(X.shape[-1] - self.window_size, dtype=object)
        idx = 0
        for job_estimators in estimators:
            for est in job_estimators:
                self.estimators_[idx] = est
                idx += 1
        return self
    
    def score(self, X, y):
        """Score each estimator on each task.
        The number of tasks in X should match the number of tasks/estimators
        given at fit time, i.e. we need
        ``X.shape[-1] == len(self.estimators_)``.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The input samples. For each data slice, the corresponding estimator
            scores the prediction, e.g.:
            ``[estimators[ii].score(X[..., ii], y) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks)
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.
        Returns
        -------
        score : array, shape (n_samples, n_estimators)
            Score for each estimator/task.
        """  # noqa: E501
        from sklearn.metrics.scorer import check_scoring
        self._check_Xy(X)
        if X.shape[-1] - self.window_size != len(self.estimators_):
            raise ValueError('The number of estimators does not match '
                             'X.shape[-1]')

        scoring = check_scoring(self.base_estimator, self.scoring)
        y = _fix_auc(scoring, y)

        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        parallel, p_func, n_jobs = parallel_func(_sl_window_score, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        est_splits = np.array_split(self.estimators_, n_jobs)
        score = parallel(p_func(est, scoring, x, y, self.window_size)
                         for (est, x) in zip(est_splits, X_splits))

        score = np.concatenate(score, axis=0)[:len(self.estimators_)]
        return score
    
    @verbose  # to use the class value
    def _transform(self, X, method):
        """Aux. function to make parallel predictions/transformation."""
        self._check_Xy(X)
        method = _check_method(self.base_estimator, method)
        if X.shape[-1] -self.window_size != len(self.estimators_):
            raise ValueError('The number of estimators does not match '
                             'X.shape[-1]')
        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        mesg = 'Transforming %s' % (self.__class__.__name__,)
        parallel, p_func, n_jobs = parallel_func(
            _sl_window_transform, self.n_jobs, verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        idx, est_splits = zip(*array_split_idx(self.estimators_, n_jobs))
        with ProgressBar(X.shape[-1] - self.window_size, verbose_bool='auto', mesg=mesg) as pb:
            y_pred = parallel(p_func(est, x, method, pb.subset(pb_idx), self.window_size)
                              for pb_idx, est, x in zip(
                                  idx, est_splits, X_splits))

        y_pred = np.concatenate(y_pred, axis=1)
        return y_pred

class GeneralizingWindowEstimator(SlidingWindowEstimator):
    """Generalization Light.
    Fit a search-light along the last dimension and use them to apply a
    systematic cross-tasks generalization.
    Parameters
    ----------
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    scoring : callable | string | None
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
        Note that the predict_method is automatically identified if scoring is
        a string (e.g. scoring="roc_auc" calls predict_proba) but is not
        automatically set if scoring is a callable (e.g.
        scoring=sklearn.metrics.roc_auc_score).
    %(n_jobs)s
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    %(verbose)s
    """

    def __repr__(self):  # noqa: D105
        repr_str = super(GeneralizingWindowEstimator, self).__repr__()
        if hasattr(self, 'estimators_'):
            repr_str = repr_str[:-1]
            repr_str += ', fitted with %i estimators>' % len(self.estimators_)
        return repr_str

    @verbose  # use class value
    def _transform(self, X, method):
        """Aux. function to make parallel predictions/transformation."""
        self._check_Xy(X)
        method = _check_method(self.base_estimator, method)
        mesg = 'Transforming %s' % (self.__class__.__name__,)
        parallel, p_func, n_jobs = parallel_func(
            _gl_transform, self.n_jobs, verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        with ProgressBar(X.shape[-1] * len(self.estimators_),
                         verbose_bool='auto', mesg=mesg) as pb:
            y_pred = parallel(
                p_func(self.estimators_, x_split, method, pb.subset(pb_idx))
                for pb_idx, x_split in array_split_idx(
                    X, n_jobs, axis=-1, n_per_split=len(self.estimators_)))

        y_pred = np.concatenate(y_pred, axis=2)
        return y_pred

    def transform(self, X):
        """Transform each data slice with all possible estimators.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The input samples. For estimator the corresponding data slice is
            used to make a transformation. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
        Returns
        -------
        Xt : array, shape (n_samples, n_estimators, n_slices)
            The transformed values generated by each estimator.
        """
        return self._transform(X, 'transform')

    def predict(self, X):
        """Predict each data slice with all possible estimators.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. For each data slice, a fitted estimator
            predicts each slice of the data independently. The feature
            dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices) | (n_samples, n_estimators, n_slices, n_targets)
            The predicted values for each estimator.
        """  # noqa: E501
        return self._transform(X, 'predict')

    def predict_proba(self, X):
        """Estimate probabilistic estimates of each data slice with all possible estimators.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. For each data slice, a fitted estimator
            predicts a slice of the data. The feature dimension can be
            multidimensional e.g.
            ``X.shape = (n_samples, n_features_1, n_features_2, n_estimators)``.
        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices, n_classes)
            The predicted values for each estimator.
        Notes
        -----
        This requires base_estimator to have a `predict_proba` method.
        """  # noqa: E501
        return self._transform(X, 'predict_proba')

    def decision_function(self, X):
        """Estimate distances of each data slice to all hyperplanes.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. Each estimator outputs the distance to
            its hyperplane, e.g.:
            ``[estimators[ii].decision_function(X[..., ii]) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            ``X.shape = (n_samples, n_features_1, n_features_2, n_estimators)``.
        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices, n_classes * (n_classes-1) // 2)
            The predicted values for each estimator.
        Notes
        -----
        This requires base_estimator to have a ``decision_function`` method.
        """  # noqa: E501
        return self._transform(X, 'decision_function')

    @verbose  # to use class value
    def score(self, X, y):
        """Score each of the estimators on the tested dimensions.
        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The input samples. For each data slice, the corresponding estimator
            scores the prediction, e.g.:
            ``[estimators[ii].score(X[..., ii], y) for ii in range(n_slices)]``.
            The feature dimension can be multidimensional e.g.
            ``X.shape = (n_samples, n_features_1, n_features_2, n_estimators)``.
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.
        Returns
        -------
        score : array, shape (n_samples, n_estimators, n_slices)
            Score for each estimator / data slice couple.
        """  # noqa: E501
        from sklearn.metrics.scorer import check_scoring
        self._check_Xy(X)
        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        mesg = 'Scoring %s' % (self.__class__.__name__,)
        parallel, p_func, n_jobs = parallel_func(_gl_window_score, self.n_jobs,
                                                 verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        scoring = check_scoring(self.base_estimator, self.scoring)
        y = _fix_auc(scoring, y)
        with ProgressBar(X.shape[-1] * len(self.estimators_),
                         verbose_bool='auto', mesg=mesg) as pb:
            score = parallel(p_func(self.estimators_, scoring, x, y,
                                    pb.subset(pb_idx), self.window_size)
                             for pb_idx, x in array_split_idx(
                                 X, n_jobs, axis=-1,
                                 n_per_split=len(self.estimators_)))

        score = np.concatenate(score, axis=1)
        return score


def _sl_window_fit(estimator, X, y, pb, window_size, **fit_params):
    """Aux. function to fit SlidingWindowEstimator in parallel.
    Fit a clone estimator to each slice of data.
    Parameters
    ----------
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    X : array, shape (n_samples, nd_features, n_estimators)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    y : array, shape (n_sample, )
        The target values.
    fit_params : dict | None
        Parameters to pass to the fit method of the estimator.
    Returns
    -------
    estimators_ : list of estimators
        The fitted estimators.
    """
    from sklearn.base import clone
    
    if window_size < 1:
        raise ValueError("Window size must be 1 or greater")
    if window_size >= X.shape[-1] -1:
        raise ValueError("Window size must be less than the number of tasks")
    
    estimators_ = list()
    for ii in range(X.shape[-1] - window_size):
        est = clone(estimator)
        X_windowed = add_features(X[..., ii:ii+window_size])
        est.fit(X_windowed, y, **fit_params)
        estimators_.append(est)
        pb.update(ii + 1)
    return estimators_

def _sl_window_score(estimators, scoring, X, y, window_size):
    """Aux. function to score SlidingEstimator in parallel.
    Predict and score each slice of data.
    Parameters
    ----------
    estimators : list, shape (n_tasks,)
        The fitted estimators.
    X : array, shape (n_samples, nd_features, n_tasks)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_tasks)
    scoring : callable, string or None
        If scoring is None (default), the predictions are internally
        generated by estimator.score(). Else, we must first get the
        predictions to pass them to ad-hoc scorer.
    y : array, shape (n_samples,) | (n_samples, n_targets)
        The target values.
    Returns
    -------
    score : array, shape (n_tasks,)
        The score for each task / slice of data.
    """
    n_tasks = X.shape[-1]
    score = np.zeros(n_tasks)
    for ii, est in enumerate(estimators):
        X_windowed = add_features(X[..., ii:ii+window_size])
        score[ii] = scoring(est, X_windowed, y)
    return score

def _sl_window_transform(estimators, X, method, pb, window_size):
    """Aux. function to transform SlidingEstimator in parallel.
    Applies transform/predict/decision_function etc for each slice of data.
    Parameters
    ----------
    estimators : list of estimators
        The fitted estimators.
    X : array, shape (n_samples, nd_features, n_estimators)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    method : str
        The estimator method to use (e.g. 'predict', 'transform').
    Returns
    -------
    y_pred : array, shape (n_samples, n_estimators, n_classes * (n_classes-1) // 2)
        The transformations for each slice of data.
    """  # noqa: E501
    for ii, est in enumerate(estimators):
        transform = getattr(est, method)
        X_windowed = add_features(X[..., ii:ii+window_size])
        _y_pred = transform(X_windowed)
        # Initialize array of predictions on the first transform iteration
        if ii == 0:
            y_pred = _sl_init_pred(_y_pred, X)
        y_pred[:, ii, ...] = _y_pred
        pb.update(ii + 1)
    return y_pred

def _gl_window_score(estimators, scoring, X, y, pb, window_size):
    """Score GeneralizingEstimator in parallel.
    Predict and score each slice of data.
    Parameters
    ----------
    estimators : list of estimators
        The fitted estimators.
    scoring : callable, string or None
        If scoring is None (default), the predictions are internally
        generated by estimator.score(). Else, we must first get the
        predictions to pass them to ad-hoc scorer.
    X : array, shape (n_samples, nd_features, n_slices)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    y : array, shape (n_samples,) | (n_samples, n_targets)
        The target values.
    Returns
    -------
    score : array, shape (n_estimators, n_slices)
        The score for each slice of data.
    """
    # FIXME: The level parallelization may be a bit high, and might be memory
    # consuming. Perhaps need to lower it down to the loop across X slices.
    score_shape = [len(estimators), X.shape[-1] - window_size]
    for jj in range(X.shape[-1] - window_size):
        for ii, est in enumerate(estimators):
            X_windowed = add_features(X[..., jj:jj+window_size])
            _score = scoring(est, X_windowed, y)
            # Initialize array of predictions on the first score iteration
            if (ii == 0) and (jj == 0):
                dtype = type(_score)
                score = np.zeros(score_shape, dtype)
            score[ii, jj, ...] = _score
            pb.update(jj * len(estimators) + ii + 1)
    return score