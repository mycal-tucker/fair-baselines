import numpy as np
import pandas as pd

from models.wasserstein_fairness.basic_costs import predict
from models.wasserstein_fairness.combined_costs import gradient_smoothed_logistic
from utils.metrics import get_disparate_impact


class Wass:
    def __init__(self, input_size, alpha=0.5, eta=0.01):
        self.input_size = input_size
        self.theta = np.random.random((input_size + 1))  # FIXME: different init.
        self.alpha = alpha
        self.eta = eta

    def train(self, training_data, num_epochs=10, test_data=None, batch_size=100):
        inputs, y, protected = training_data
        protected_inputs = {}
        for idx, protected_val in enumerate(protected):
            if protected_val not in protected_inputs.keys():
                protected_inputs[protected_val] = []
            protected_inputs[protected_val].append(inputs[idx])
        protected_inputs = [np.asarray(inp) for inp in protected_inputs.values()]
        for epoch in range(num_epochs):
            batch = np.random.permutation(inputs.shape[0])[:batch_size]
            batch_inputs = (pd.DataFrame(inputs[batch]), pd.DataFrame(y[batch]))
            protected_batch_ids = [np.random.permutation(protected.shape[0])[:batch_size] for protected in protected_inputs]
            batch_protected = [protected[batch_ids] for protected, batch_ids in zip(protected_inputs, protected_batch_ids)]

            grad, cost_log, cost_wass = gradient_smoothed_logistic(batch_inputs,
                                                                  batch_protected,
                                                                  self.theta,
                                                                  lambda_=1,
                                                                  beta=1,
                                                                  alpha=self.alpha,
                                                                   # distance='wasserstein-1')
                                                                   distance='wasserstein-2')
            self.theta = self.theta - self.eta * grad

    def evaluate(self, test_data):
        inputs, y, protected = test_data
        predictions = predict(inputs, self.theta, threshold=0.5)
        num_correct = 0
        np_preds = np.zeros((len(predictions), 2))
        for idx, true_val in enumerate(y):
            pred_val = predictions[idx]
            if pred_val == true_val:
                num_correct += 1
            if pred_val:
                np_preds[idx, 1] = 1
            else:
                np_preds[idx, 0] = 1
        accuracy = num_correct / len(predictions)
        print("Accuracy", accuracy)
        di, dd, s_acc = get_disparate_impact(np_preds, np_preds, [y, protected])
        return accuracy, di, dd, s_acc
