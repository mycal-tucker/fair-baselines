from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

import numpy as np
from utils.metrics import get_disparate_impact, accuracy
from utils.plotting import plot_encodings

class VFAE:
    def __init__(self, input_size, output_sizes, weights):
        self.input_size = input_size
        self.gamma = 1
        self.output_sizes = output_sizes
        self.latent_dim = 30  # From paper
        # self.latent_dim = 2  # For debugging
        self.batch_size = 100
        parts = self._build_parts()
        self.encoder = parts[0]
        self.model, self.inference_model, self.viz_model = self._compose_parts(parts)

    def _build_parts(self):
        x_inp = keras.Input(shape=(self.input_size,), name='x_inp')
        enc_s_inp = keras.Input(shape=(1,), name='enc_s_input')
        enc_inp = layers.Concatenate()([x_inp, enc_s_inp])
        x = layers.Dense(60, activation="relu")(enc_inp)
        x = layers.Dense(self.latent_dim, activation="relu")(x)
        enc_mu = layers.Dense(self.latent_dim, activation='linear')(x)
        enc_log_var = layers.Dense(self.latent_dim, activation='linear')(x)
        feature_enc = layers.Lambda(VFAE.sampling, output_shape=(self.latent_dim,), name='z')(
            [enc_mu, enc_log_var])
        encoder = keras.Model(inputs=[x_inp, enc_s_inp], outputs=[feature_enc, enc_mu, enc_log_var], name='encoder')

        pred_inp = keras.Input(shape=(self.latent_dim,), name='pre_input')
        # x = layers.Dense(60, activation='relu')(pred_inp)
        x = pred_inp
        y = layers.Dense(self.output_sizes[0], activation='softmax')(x)
        pred = keras.Model(inputs=pred_inp, outputs=y, name='predictor')

        s_inp = keras.Input(shape=(1,), name='s_input')
        dec_z_input = keras.Input(shape=(self.latent_dim,))
        dec_input = layers.Concatenate()([s_inp, dec_z_input])
        x = layers.Dense(128, activation='relu')(dec_input)
        reconstruction = layers.Dense(self.input_size, activation='sigmoid')(x)
        decoder = keras.Model(inputs=[s_inp, dec_z_input], outputs=reconstruction, name="decoder")
        return encoder, pred, decoder

    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    def loss_func(self, encoder_mu, encoder_log_variance, s0_encs, s1_encs):
        def vae_classification(y_true, y_predict):
            pred_factor = 10
            pred_loss = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, y_predict))
            return pred_factor * pred_loss

        def vae_kl_loss(mu, log_var):
            kl_factor = 0.5
            kl_loss = -0.5 * keras.backend.mean(
                1.0 + log_var - keras.backend.square(mu) - keras.backend.exp(
                    log_var))
            return kl_loss * kl_factor

        def psi(x):
            w = tf.random.normal((self.latent_dim, 500),
                                 stddev=tf.sqrt(0.5 / self.latent_dim))
            b = tf.random.uniform((self.batch_size, 500), 0, 2 * np.pi)
            return tf.pow(2./self.latent_dim, 0.5) * tf.cos(tf.pow(2./self.gamma, 0.5) * tf.matmul(x, w) + b)

        def fast_mmd(x1, x2):
            mmd_factor = 1
            inner_diff = tf.reduce_mean(psi(x1), axis=0) - tf.reduce_mean(psi(x2), axis=0)
            dotted = keras.backend.mean(tf.tensordot(inner_diff, inner_diff, axes=1))
            return mmd_factor * dotted

        def total_loss(y_true, y_predict):
            class_loss = vae_classification(y_true, y_predict)
            kl_loss = vae_kl_loss(encoder_mu, encoder_log_variance)
            mmd_loss = fast_mmd(s0_encs, s1_encs)
            loss = class_loss + kl_loss + mmd_loss
            return loss
        return total_loss

    def _compose_parts(self, parts):
        encoder, pred, decoder = parts
        x = keras.Input(shape=(self.input_size,))
        s = keras.Input(shape=(1,))
        z, mu, log_var = encoder([x, s])

        s0_encs = tf.math.multiply(z, tf.tile(s, [1, self.latent_dim]))
        s1_encs = tf.math.multiply(z, tf.tile(1 - s, [1, self.latent_dim]))
        y = pred(z)
        recons = decoder([s, z])
        overall = keras.Model(inputs=[x, s], outputs=[y, recons])

        overall.compile(optimizer='adam',
                        loss=[self.loss_func(mu, log_var, s0_encs, s1_encs),
                                'mse'],
                        loss_weights=[1, 0.0],
                        metrics=['accuracy'])

        viz_model = keras.Model(inputs=[x, s], outputs=[s0_encs, s1_encs])

        # y_inf = pred(mu)
        y_inf = pred(z)
        inference = keras.Model(inputs=[x, s], outputs=[y_inf])
        inference.compile(optimizer='adam',
                        loss='mse')
        return overall, inference, viz_model

    def train(self, training_data, num_epochs):
        inps, outs = training_data
        y, _, protected = outs
        self.model.fit([inps, protected], [y, inps], batch_size=self.batch_size, epochs=num_epochs, verbose=0, validation_split=0.0)

    def evaluate(self, x, y_one_hot, ys):
        label, protected_one_hot = y_one_hot
        _, protected = ys

        eval_results = self.inference_model.evaluate([x, protected], label, batch_size=self.batch_size)
        print("Eval results", eval_results)
        predictions = self.inference_model.predict([x, protected])
        acc = accuracy(predictions, ys[0])
        print("Accuracy", acc)
        di, dd, s_acc = get_disparate_impact(self.encoder.predict([x, protected])[0], predictions, ys)

        # Visualize
        # viz_outputs = self.viz_model.predict([x, protected])
        # s0_enc, s1_enc = viz_outputs
        # plot_encodings([s0_enc, s1_enc])
        return acc, di, dd, s_acc
