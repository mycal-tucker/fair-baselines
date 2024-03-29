import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype


def normalize(df):
    result = df.copy()
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result


def rebalance(x, p, y, dist):
    keep_x = []
    keep_p = []
    keep_y = []
    for i in range(x.shape[0]):
        y_key = y[i]
        proportion = dist.get(y_key)
        min_proportion = min(dist.values())
        if np.random.random() < (min_proportion/ proportion):
            keep_x.append(x[i, :])
            keep_p.append(p[i])
            keep_y.append(y[i])
    return np.asarray(keep_x), np.asarray(keep_p), np.asarray(keep_y)


def get_german_data(filepath, wass_setup=False):
    # From: https://www.kaggle.com/twunderbar/german-credit-risk-classification-with-keras
    data = pd.read_csv(filepath, index_col=0, sep=',')
    labels = data.columns
    # protected_label = 'Sex  # Confirmed that in MMD paper, sensitive value is sex.
    # ^ You can't believe everything you read these days. What they meant was age.
    protected_label = 'Age'
    # lets go through column 2 column
    for col in labels:
        if is_string_dtype(data[col]):
            if col == 'Risk':
                # we want 'Risk' to be a binary variable
                data[col] = pd.factorize(data[col])[0]
                continue
            if col == protected_label:  # Whatever you want for protected, if it's a categorical variable
                data[col] = pd.factorize(data[col])[0]
                continue
            # the other categorical columns should be one-hot encoded
            data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
            data.drop(col, axis=1, inplace=True)
        else:
            if col == protected_label:
                # The standard is to threshold age by 25, but Wass uses 30
                age_threshold = 30 if wass_setup else 25
                data.loc[data[col] <= age_threshold, col] = 0
                data.loc[data[col] > age_threshold, col] = 1
                continue
            data[col] = normalize(data[col])

    # move 'Risk' back to the end of the df
    data = data[[c for c in data if c not in ['Risk', protected_label]] + [protected_label] + ['Risk']]
    # Shuffle all the data
    data = data.sample(frac=1)
    frac = 0.8 if not wass_setup else 0.67
    train_idx = int(1000 * frac)
    data_train = data.iloc[:train_idx]
    data_test = data.iloc[train_idx:]
    x_train = data_train.iloc[:, :-2]
    p_train = data_train.iloc[:, -2]
    y_train = data_train.iloc[:, -1]
    x_test = data_test.iloc[:, :-2]
    p_test = data_test.iloc[:, -2]
    y_test = data_test.iloc[:, -1]
    balanced_x, balanced_p, balanced_y = x_train.values, p_train.values, y_train.values

    return balanced_x, balanced_y, balanced_p, x_test.values, y_test.values, p_test.values

