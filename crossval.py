"""Cross-validation wrappers and utilities for Kaggle House Price competition."""
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

from utils import rmsle, rmsle_sk


def calc_kfold_score_helper(model, kf, df, y):
    """
    A generator helper function for `calc_kfold_score`. Does a fit for each split and yields the RMSLE score. 
    
    :param model: an instance of sklearn-model
    :param kf: an instance of sklearn KFold
    :param df: the dataframe with training data 
    :param y: dependent value
    :returns a generator that yields RMSLE scores for each split.
    """

    for train_index, test_index in kf.split(df):
        df_train = df.ix[train_index + 1]
        y_train = y.ix[train_index + 1]

        df_cv = df.ix[test_index + 1]
        y_cv = y.ix[test_index + 1]

        model.fit(df_train, y_train)

        yield rmsle(y_cv, model.predict(df_cv))


def calc_kfold_score(model, df, y, n_splits=3, shuffle=True):
    """
    Calculate crossvalidation score for the given model and data. Uses sklearn's KFold with shuffle=True.
    
    :param model: an instance of sklearn-model 
    :param df: the dataframe with training data
    :param y: dependent value
    :param n_splits: the amount of splits (i.e. K in K-fold)
    :param shuffle: whether to shuffle or not
    :return: mean, std
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    scores = list(calc_kfold_score_helper(model, kf, df, y))
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std


def gridcv(df_train, y_train, clf, params, n_jobs=1, n_folds=3, n_repeats=1, verbose=True):
    """
    Wrapper for Cross-validated Grid Search.

    :param df_train: training data
    :param y_train: true values
    :param clf: predictor
    :param params: the params
    :param n_jobs: how many threads to use
    :param n_folds: how many folds to use
    :param n_repeats: how often to repeat the crossvalidation
    :param verbose: to be chatty or not
    :return 3-tuple: best estimator (sklearn predictor object), best params (dict), best score (float)
    """
    cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)
    grid_cv = GridSearchCV(clf, params, n_jobs=n_jobs, scoring=rmsle_sk, cv=cv)
    grid_cv.fit(df_train, y_train)
    score = -1 * grid_cv.best_score_

    if verbose:
        print("{} best score: {:.4f}, best params: {}"
              .format(datetime.now().replace(second=0, microsecond=0),
                      score, grid_cv.best_params_))

    return grid_cv.best_estimator_, grid_cv.best_params_, score
