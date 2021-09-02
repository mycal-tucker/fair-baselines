import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.flip_gradient_tf import GradReverse

class AdvModel:
    def __init__(self, input_size, output_sizes):
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.latent_dim = 32
        parts = self._build_parts()
        self.model = self._compose_parts(parts)

    def _build_parts(self):
        enc_inp = keras.Input(shape=(self.input_size,), name='img_input')
        x = layers.Dense(128, activation="relu")(enc_inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        feature_enc = layers.Dense(self.latent_dim, activation="linear")(x)
        encoder = keras.Model(inputs=enc_inp, outputs=feature_enc, name='encoder')

        pred_inp = keras.Input(shape=(self.latent_dim,), name='pre_input')
        x = layers.Dense(128, activation="relu")(pred_inp)
        x = layers.Dense(128, activation="relu")(x)
        y = layers.Dense(self.output_sizes[0], activation='softmax')(x)
        pred = keras.Model(inputs=pred_inp, outputs=y, name='predictor')

        adv_inp = keras.Input(shape=(self.latent_dim,), name='adv_input')
        x = layers.Dense(128, activation="relu")(adv_inp)
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
                        loss_weights=[1, 10],
                        metrics=['accuracy'])
        return overall

    def train(self, training_data):
        inps, outs = training_data
        self.model.fit(inps, outs, batch_size=32, epochs=100)
