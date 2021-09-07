from tensorflow import keras
from tensorflow.keras import layers
from utils.metrics import get_disparate_impact
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
        di, dd, s_acc = get_disparate_impact(self.encoder.predict(x), predictions[0], ys)
        return eval_results[-2], di, dd, s_acc

