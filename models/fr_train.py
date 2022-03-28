from tensorflow import keras
from tensorflow.keras import layers
from utils.metrics import get_disparate_impact
from models.flip_gradient_tf import GradReverse


class FRTrain:
    def __init__(self, input_sizes, output_sizes, weights):
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.latent_dim = 64
        self.weights = weights
        parts = self._build_parts()
        self.encoder = parts[0]
        self.model = self._compose_parts(parts)

    def _build_parts(self):
        enc_inp = keras.Input(shape=(self.input_sizes[0],), name='img_input')
        protected_input = keras.Input(shape=(self.input_sizes[1]))
        full_enc_inp = layers.Concatenate()([enc_inp, protected_input])
        feature_enc = layers.Dense(self.latent_dim, activation="relu")(full_enc_inp)
        encoder = keras.Model(inputs=[enc_inp, protected_input], outputs=feature_enc, name='encoder')

        pred_inp = keras.Input(shape=(self.latent_dim,), name='pre_input')
        y = layers.Dense(self.output_sizes[0], activation='softmax')(pred_inp)
        pred = keras.Model(inputs=pred_inp, outputs=y, name='predictor')

        adv_fair_inp = keras.Input(shape=(self.output_sizes[0],), name='adv_input')  # Pred
        # a = layers.Dense(64, activation='relu')(adv_fair_inp)
        # a = layers.Dense(self.output_sizes[1], activation='softmax')(a)
        x = layers.Dense(128, activation="relu")(adv_fair_inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        a = layers.Dense(self.output_sizes[1], activation='softmax')(x)
        fair_adv = keras.Model(inputs=adv_fair_inp, outputs=a, name='fairAdv')

        adv_robust_inp = keras.Input(shape=(self.input_sizes[0] + sum(self.output_sizes[:2]),))
        x = layers.LeakyReLU(0.2)(adv_robust_inp)
        a2 = layers.Dense(self.output_sizes[2], activation='softmax')(x)
        adv_robust = keras.Model(inputs=adv_robust_inp, outputs=a2, name="robustAdv")

        return encoder, pred, fair_adv, adv_robust

    def _compose_parts(self, parts):
        encoder, pred, f_adv, r_adv = parts
        x = keras.Input(shape=(self.input_sizes[0],))
        s = keras.Input(shape=(self.input_sizes[1]))
        z = encoder([x, s])
        y = pred(z)

        flipped_y = GradReverse()(y)
        fair = f_adv(flipped_y)

        r_inp = layers.Concatenate()([flipped_y, s, x])
        robust = r_adv(r_inp)
        overall = keras.Model(inputs=[x, s], outputs=[y, fair, robust])
        overall.compile(optimizer='adam',
                        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                        loss_weights=self.weights,
                        metrics=['accuracy'])
        return overall

    def train(self, training_data, num_epochs):
        inps, outs = training_data
        self.model.fit(inps, outs, batch_size=16, epochs=num_epochs, verbose=0, validation_split=0.0)

    def evaluate(self, x, y_one_hot, ys):
        eval_results = self.model.evaluate(x, y_one_hot)
        print("Eval results", eval_results)
        predictions = self.model.predict(x)
        di, dd, s_acc = get_disparate_impact(predictions[0], predictions[0], ys)
        return eval_results[-3], di, dd, s_acc

