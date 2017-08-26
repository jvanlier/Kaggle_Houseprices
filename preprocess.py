"""Data-preprocessing functions for Kaggle House Price competition."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


# The max. amount of values we'd expect in a numeric column that actually contains ordinal values:
MAX_VALUES_ORDINAL = 20


def prep_data(df, col_y, manual_transformations, verbose=False):
    """Takes a DataFrame, makes a local copy, and does the following:
    1) apply manual transformations
    2) apply KNN imputer to fill missing data
    3) apply automatic transformations (i.e. standardization, label encoder, one hot encoding)
    4) split df into df_train, y_train (a Series), df_test
    Returns the transformed dataframes.

    :param df: a single DF containing all records (test and train).
    :param col_y: (string) the name of the 'y' column, i.e. the column that we want to predict.
    :param manual_transformations: list of manual transformation functions
    :param verbose: whether or not to print stuff
    :return: 3-tuple: training_X, training_y, test_X
    """
    df = df.copy()

    if verbose:
        print(" ** Applying manual transformations")

    for trans in manual_transformations:
        if verbose:
            print(" -> Manual transformation {}".format(trans.__name__))
        df = trans(df)

    y = df[col_y]
    df = df.drop(col_y, axis=1)

    if verbose:
        print("\n ** Starting knn_impute")
    df = knn_impute(df)

    assert not np.any([df[c].isnull().any() for c in df.columns])

    if verbose:
        print("\n ** Starting auto-transform")

    df = auto_transform(df, verbose)
    df[col_y] = y

    y_train = df[~pd.isnull(df[col_y])][col_y]
    df_train = df[~pd.isnull(df[col_y])].drop(col_y, axis=1)
    df_test = df[pd.isnull(df[col_y])].drop(col_y, axis=1)

    if verbose:
        print("\n ** Training df has {} columns and {} rows, test df has {} columns and {} rows"
              .format(len(df_train.columns), len(df_train), len(df_test.columns), len(df_test)))

    return df_train, y_train, df_test


def auto_transform(df: pd.DataFrame, verbose=False):
    """Automatically transform a DataFrame using best-effort guesses. Converts all strings to numbers so that it can be
    used with sklearn.

    Assumptions:
    - There are no more missing values in the data. These have already been filled/imputed.
    - Numeric columns with only a few values are ordinal. If you have numeric categorical values in which order 
      is irrelevant, make sure they're strings (i.e. column has 'object' dtype), so that they can get 
      one-hot-encoded properly.
    - For string columns with max. 2 unique values it's OK to label encode these in a single column (0/1).
    - String columns with more than 2 values will be one-hot encoded across multiple columns. Make sure there's not
      too many distinct strings because it'll explode the amount of columns in the return DF (and likely won't help
      with your model...).

    :param df: the input DataFrame. Make sure the column you want to predict is _NOT_ included.
    :param verbose: whether to be chatty or not 
    :return: transformed DataFrame
    """
    if verbose:
        print(" -> {} columns in df before transformations".format(len(df.columns)))

    cols_cat = {c for c in df.columns if df[c].dtype == 'object'}
    cols_numeric = set(df.columns) - cols_cat
    cols_numeric_ordinal = {c for c in cols_numeric if len(df[c].value_counts()) <= MAX_VALUES_ORDINAL}
    cols_numeric -= cols_numeric_ordinal
    # Need to use dropna=False with nunique(), because otherwise NaN is ignored and not counted as unique:
    cols_cat_labelenc = {c for c in df.columns if df[c].dtype == 'object' and
                         df[c].nunique(dropna=False) <= 2}
    cols_cat_ohe = cols_cat - cols_cat_labelenc

    if verbose:
        print(" -> Result of automatic column splitting: ")
        print(" ---> Numeric/Ordinal: [No further transformation]", sorted(cols_numeric_ordinal))
        print(" ---> Numerical: [Standardized: mean removal and unit std]", sorted(cols_numeric))
        print(" ---> Categorical with <= 2 vars [LabelEnc]", sorted(cols_cat_labelenc))
        print(" ---> Categorical with > 2 vars [OneHot]", sorted(cols_cat_ohe))

    for col in cols_numeric:
        df.loc[:, col] = (df[col] - df[col].mean()) / df[col].std()

    for col in cols_cat_labelenc:
        le = LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col])

    df_without_ohe_cols = df.drop(cols_cat_ohe, axis=1)
    df_ohe_cols_dummies = pd.get_dummies(df[list(cols_cat_ohe)])
    df = pd.concat((df_without_ohe_cols, df_ohe_cols_dummies), axis=1)

    if verbose:
        print(" -> {} columns in df after transformations".format(len(df.columns)))

    return df


def knn_impute(df):
    """Perform K-nearest neighbour imputation for missing values.
    
    :param df: input dataframe (will not be modified)
    :return df: dataframe with missing values imputed.
    """
    missing_cols = list(gen_missing_columns(df))
    df_sub = df_without_columns(df, missing_cols)
    df_sub_t = auto_transform(df_sub.copy())
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(df_sub_t)

    for col in missing_cols:
        # In original df, find indices that are missing for current column. Then lookup those indices
        # and find the first value that's filled in, ordered based on neighbour distance (closest first).
        idxes_missing = df[df[col].isnull()].index.values
        for idx_missing in idxes_missing:
            neighbours = nn.kneighbors(df_sub_t.loc[idx_missing, :].values.reshape(1, -1), return_distance=False)
            neighbour_values = df.loc[neighbours.flatten(), col].dropna()

            if len(neighbour_values) == 0:
                raise ValueError("None of the neighbours have a value in {}, increase n_neighbours".format(col))

            if df[col].dtype == 'object':
                # Fill the most frequent value
                new_value = neighbour_values.value_counts().index[0]
            else:
                new_value = neighbour_values.mean()

            df.loc[idx_missing, col] = new_value
    return df


def gen_missing_columns(df, verbose=False):
    """A generator that yields the names of the columns that have at least one value missing.

    :param df: dataframe
    :param verbose: to be chatty or not
    :return a generator that yields names of columns that have missing data
    """
    for col in df.columns:
        n_missing = len(df[df[col].isnull()])
        if n_missing > 0:
            if verbose:
                print(col, n_missing)
            yield col


def df_without_columns(df, cols_to_remove):
    """
    Returns a copy of the input dataframe, without specified columns. 

    :param df: the input DataFrame
    :param cols_to_remove: list or set of columns to remove
    :return: a copy of the input dataframe, without specified columns.
    """
    df_sub = df[list(set(df.columns.values.tolist()) - set(cols_to_remove))]
    return df_sub.copy()
