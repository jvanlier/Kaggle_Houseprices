"""Cross-validation wrappers and utilities for Kaggle House Price competition."""

from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from utils import rmsle, rmsle_sk


def calc_kfold_score_helper(model, kf, df, y):
    """
    A generator helper function for `calc_kfold_score`. Does a fit for each split and yields the RMSLE score. 
    
    :param model: an instance of sklearn-model
    :param kf: an instance of sklearn KFold
    :param df: the dataframe with training data 
    :param y: the labels
    :returns a generator that yields RMSLE scores for each split.
    """
    for kf_train, kf_cv in kf.split(df):
        df_train_fold = df.ix[kf_train + 1]
        y_train_fold = y.ix[kf_train + 1]

        df_cv_fold = df.ix[kf_cv + 1]
        y_cv_fold = y.ix[kf_cv + 1]

        model.fit(df_train_fold, y_train_fold)

        yield rmsle(y_cv_fold, model.predict(df_cv_fold))


def calc_kfold_score(model, df, y, n_splits=3):
    """
    Calculate crossvalidation score for the given model and data. Uses sklearn's KFold with shuffle=True.
    
    :param model: an instance of sklearn-model 
    :param df: 
    :param y: 
    :param n_splits: 
    :return: 
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = list(calc_kfold_score_helper(model, kf, df, y))
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std


def gridcv(df_train, y_train, clf, params, n_jobs=8, verbose=True):
    """
    Wrapper for Cross-validated Grid Search.

    :param df_train: training data
    :param y_train: true values
    :param clf: predictor
    :param params: the params
    :param n_jobs: how many threads to use
    :param verbose: to be chatty or not
    :return 3-tuple: best estimator (sklearn predictor object), best params (dict), best score (float)
    """
    grid_cv = GridSearchCV(clf, params, n_jobs=n_jobs, scoring=rmsle_sk, )
    grid_cv.fit(df_train, y_train)
    score = -1 * grid_cv.best_score_

    if verbose:
        print("{} best score: {:.4f}, best params: {}"
              .format(datetime.now().replace(second=0, microsecond=0),
                      score, grid_cv.best_params_))

    return grid_cv.best_estimator_, grid_cv.best_params_, score
