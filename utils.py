"""Utilities such as scoring functions, submission generators, transformation helpers and plotting functions
for Kaggle House Price competition.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

COL_Y = 'SalePrice'


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithic Error"""
    return np.sqrt(mean_squared_error(np.log(np.exp(y_true)), np.log(np.exp(y_pred))))


def rmsle_sk(estimator, X, y):
    """Sklearn-compatible version of Root Mean Squared Logarithmic Error. Results must be * -1'd."""
    return -rmsle(y, estimator.predict(X))


def create_submission(est, df_test):
    """
    Create a Kaggle submission file using a single estimator. Writes 'submission.csv' to the filesystem.
    Automatically undoes the log transform on the Sale Price.
    
    :param est: sklearn estimator
    :param df_test: test data 
    """
    y_test = est.predict(df_test)
    df_sub = pd.DataFrame({'Id': df_test.index, COL_Y: np.exp(y_test)})
    df_sub = df_sub.set_index('Id')
    df_sub.to_csv('submission.csv')


def create_submission_from_ensemble(ests, df_test):
    """
    Create a Kaggle submission file using a list of estimators (i.e. an ensemble). 
    Writes 'submission_ensemble.csv' to the filesystem.
    Uses a simple unweighted average.
    Automatically undoes the log transform on the Sale Price.

    :param ests: sklearn estimator list
    :param df_test: test data 
    :return: y_test, the averaged predictions 
    """
    predictions = []
    for est in ests:
        predictions.append(np.exp(est.predict(df_test)))
    y_test = np.mean(predictions, axis=0)
    df_sub = pd.DataFrame({'Id': df_test.index, COL_Y: y_test})
    df_sub = df_sub.set_index('Id')
    df_sub.to_csv('submission_ensemble.csv')
    return y_test


def sqrt_transform_helper(df, col):
    """Applies a square root transform on a numeric column. All values must be >= 0."""
    df.loc[:, col] = np.sqrt(df[col])
    return df


def log_transform_helper(df, col):
    """Applies a log transform on a numeric column. Values are +1'd to prevent log(0) = -inf. 
    All values must be >= 0."""
    df.loc[:, col] = np.log1p(df[col])
    return df


def ihs_transform_helper(df, col):
    """Applies inverse hyperbolic sine transformation, which more naturally handles the log(0) issue than the standard
    log transform. Results are similar otherwise. Values must still be >= 0."""
    df.loc[:, col] = np.log(df[col] + np.sqrt((df[col] ** 2 + 1)))
    return df


def boxplot_sort_order(df, field):
    """Get order of boxpot labels, sorted on SalePrice (median)"""
    return sort_order_helper(df, field, 'median')


def barchart_sort_order(df, field):
    """Get order of barchart labels, sorted on SalePrice (sum)"""
    return sort_order_helper(df, field, 'sum')


def sort_order_helper(df, field, aggregation_method):
    return df.groupby(field).agg({COL_Y: aggregation_method}).sort_values(COL_Y).index.values


def data_profile(df, field, dtype=None, logy=False):
    """
    Plots/prints generic information about a field.

    :param df: the DataFrame
    :param field: name of field in DataFrame
    :param dtype: None to infer from data, can be set explicitly to 'categorical' to force violinplots for ordinal vars.
    :param logy: if True, will copy the DataFrame and apply log transformation on df[COL_Y].
    """
    if logy:
        df = df.copy()
        df[COL_Y] = np.log(df[COL_Y])

    if dtype is None:
        # try to infer
        if df[field].dtype == 'O':
            dtype = 'categorial'
        else:
            dtype = 'numeric'

    if dtype is 'numeric':
        sns.jointplot(field, COL_Y, data=df, kind='reg')
        plt.show()

    else:
        plt.figure(figsize=(12, 4))
        vp = sns.violinplot(x=field, y=COL_Y, data=df, order=boxplot_sort_order(df, field))
        vp.set_xticklabels(vp.get_xticklabels(), rotation=30)

        sale_price_medians = df[[field, COL_Y]].groupby(field)[COL_Y].median()
        if not logy:
            plt.ylim(sale_price_medians.min() - 50000, sale_price_medians.max() + 50000)

        plt.show()
        plt.figure(figsize=(12, 4))
        bp = sns.barplot(x='index', y=field,
                         data=pd.DataFrame(df[field].value_counts().reset_index()),
                         order=boxplot_sort_order(df, field))
        bp.set_xticklabels(bp.get_xticklabels(), rotation=30)

        plt.xlabel(field)
        plt.ylabel('Count')
        plt.show()

    missing = df[df[field].isnull()]
    print(len(missing), " missing entries")


def hist_compare(condition, label1, label2, df):
    """Compare histograms of SalePrice of 2 groups. The condition is applied on group 1 and !condition is applied
    on group 2."""
    df_c = df[condition][COL_Y]
    df_nc = df[~condition][COL_Y]
    print("{}: {} | {}: {}".format(label1, len(df_c), label2, len(df_nc)))

    df_c.hist(label=label1, alpha=.4, normed=True)
    df_nc.hist(label=label2, alpha=.4, normed=True, color='orange')
    plt.legend()
    plt.show()

