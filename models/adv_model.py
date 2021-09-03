import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers

from models.flip_gradient_tf import GradReverse


class AdvModel:
    def __init__(self, input_size, output_sizes, weights):
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.latent_dim = 64
        self.weights = weights
        parts = self._build_parts()
        self.encoder = parts[0]
        self.model = self._compose_parts(parts)

    def _build_parts(self):
        enc_inp = keras.Input(shape=(self.input_size,), name='img_input')
        x = enc_inp
        feature_enc = layers.Dense(self.latent_dim, activation="relu")(x)
        encoder = keras.Model(inputs=enc_inp, outputs=feature_enc, name='encoder')

        pred_inp = keras.Input(shape=(self.latent_dim,), name='pre_input')
        x = pred_inp
        y = layers.Dense(self.output_sizes[0], activation='softmax')(x)
        pred = keras.Model(inputs=pred_inp, outputs=y, name='predictor')

        adv_inp = keras.Input(shape=(self.latent_dim,), name='adv_input')
        x = layers.Dense(128, activation="relu")(adv_inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        a = layers.Dense(self.output_sizes[1], activation='softmax')(x)
        adv = keras.Model(inputs=adv_inp, outputs=a, name='adversary')

        return encoder, pred, adv

    def _compose_parts(self, parts):
        encoder, pred, adv = parts
        inp = keras.Input(shape=(self.input_size,))
        z = encoder(inp)
        y = pred(z)

        flipped_z = GradReverse()(z)
        a = adv(flipped_z)

        overall = keras.Model(inputs=inp, outputs=[y, a])
        overall.compile(optimizer='adam',
                        loss=['categorical_crossentropy', 'categorical_crossentropy'],
                        loss_weights=self.weights,
                        metrics=['accuracy'])
        return overall

    def train(self, training_data, num_epochs):
        inps, outs = training_data
        self.model.fit(inps, outs, batch_size=16, epochs=num_epochs, verbose=1, validation_split=0.0)

    def evaluate(self, x, y_one_hot, ys):
        eval_results = self.model.evaluate(x, y_one_hot)
        print("Eval results", eval_results)
        predictions = self.model.predict(x)
        di, dd, s_acc = self.get_disparate_impact(x, predictions, ys)
        return eval_results[-2], di, dd, s_acc

    def get_disparate_impact(self, x, predictions, ys):
        protected_totals = {}
        protected_positive = {}
        protected_vals = ys[1]
        for i, protected_val in enumerate(protected_vals):
            if protected_val not in protected_totals.keys():
                protected_totals[protected_val] = 0
                protected_positive[protected_val] = 0
            protected_totals[protected_val] += 1
            prediction = np.argmax(predictions[0][i])
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
        x_train = x[:int(training_frac * len(x))]
        enc_train = self.encoder.predict(x_train)
        y_train = ys[1][:int(training_frac * len(x))]
        scaler = preprocessing.StandardScaler().fit(enc_train)
        enc_train = scaler.transform(enc_train)
        regression_model.fit(enc_train, y_train)
        x_test = x[int(training_frac * len(x)):]
        enc_test = self.encoder.predict(x_test)
        enc_test = scaler.transform(enc_test)
        y_test = ys[1][int(training_frac * len(x)):]
        score = regression_model.score(enc_test, y_test)
        return disparity_impact, dem_disparity, score