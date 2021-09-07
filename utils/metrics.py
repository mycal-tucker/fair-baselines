import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def get_disparate_impact(encs, predictions, ys):
    protected_totals = {}
    protected_positive = {}
    protected_vals = ys[1]
    for i, protected_val in enumerate(protected_vals):
        if protected_val not in protected_totals.keys():
            protected_totals[protected_val] = 0
            protected_positive[protected_val] = 0
        protected_totals[protected_val] += 1
        prediction = np.argmax(predictions[i])
        if prediction == 1:
            protected_positive[protected_val] += 1
    sorted_keys = sorted(protected_positive.keys())
    max_key = max(protected_totals, key=lambda x: protected_positive.get(x) / protected_totals.get(x))
    min_key = min(protected_totals, key=lambda x: protected_positive.get(x) / protected_totals.get(x))
    random.shuffle(sorted_keys)
    sorted_keys = [max_key, min_key]

    key0 = max_key
    key1 = min_key
    fraction_0 = protected_positive.get(key0) / protected_totals.get(key0)
    fraction_1 = protected_positive.get(key1) / protected_totals.get(key1)
    if fraction_0 == 0 or fraction_1 == 0:
        disparity_impact = 0
    else:
        disparity_impact = min([fraction_0 / fraction_1, fraction_1 / fraction_0])

    # From the Wasserstein paper, compute the "demographic disparity"
    mean_prob = sum(protected_positive.values()) / sum(protected_totals.values())
    dem_disparity = 0
    for key in sorted_keys:
        fraction = protected_positive.get(key) / protected_totals.get(key)
        dem_disparity += abs(fraction - mean_prob)

    # Use a linear model to predict protected field for s_diff
    regression_model = LogisticRegression()
    training_frac = 0.5
    enc_train = encs[:int(training_frac * len(encs))]
    y_train = ys[1][:int(training_frac * len(encs))]
    scaler = preprocessing.StandardScaler().fit(enc_train)
    enc_train = scaler.transform(enc_train)
    regression_model.fit(enc_train, y_train)
    enc_test = encs[int(training_frac * len(encs)):]
    enc_test = scaler.transform(enc_test)
    y_test = ys[1][int(training_frac * len(encs)):]
    score = regression_model.score(enc_test, y_test)
    return disparity_impact, dem_disparity, score


def accuracy(y_hat, y):
    num_total = 0
    num_correct = 0
    for i, one_hot in enumerate(y_hat):
        num_total += 1
        if np.argmax(one_hot) == y[i]:
            num_correct += 1
    return num_correct / num_total