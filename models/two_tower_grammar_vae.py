from keras import backend as K
from keras.losses import binary_crossentropy, mse
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Concatenate, Reshape
from keras.layers import Dense, Flatten, RepeatVector
from keras.layers import TimeDistributed, GRU
from keras.layers import Convolution1D
from keras.optimizers import Adam
import tensorflow as tf
import zinc_grammar as G

masks_K = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 277
MAX_LEN_FEATURES = 200
DIM = G.D


class MoleculeVAE:
    autoencoder = None

    def create(self,
               charset,
               max_length=MAX_LEN,
               max_length_features=MAX_LEN_FEATURES,
               latent_rep_size=128,
               weights_file=None,
               learning_rate=0.001,
               features_weight=10):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        f = Input(shape=(max_length_features, 1))

        _, z = self._buildEncoder(x, f, latent_rep_size, max_length, max_length_features, features_weight)
        self.encoder = Model([x, f], z)

        encoded_input = Input(shape=(latent_rep_size,))
        o, fo = self._buildDecoder(
            encoded_input,
            latent_rep_size,
            max_length,
            max_length_features,
            charset_length
        )
        self.decoder = Model(
            encoded_input,
            [o, fo]
        )

        x1 = Input(shape=(max_length, charset_length))
        f1 = Input(shape=(max_length_features, 1))
        vae_loss, z1 = self._buildEncoder(x1, f1, latent_rep_size, max_length, max_length_features, features_weight)
        o1, fo1 = self._buildDecoder(
            z1,
            latent_rep_size,
            max_length,
            max_length_features,
            charset_length
        )
        self.autoencoder = Model(
            [x1, f1],
            [o1, fo1]
        )

        # for obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(max_length, charset_length))
        f2 = Input(shape=(max_length_features, 1))
        (z_m, z_l_v) = self._encoderMeanVar(x2, f2, latent_rep_size)
        self.encoderMV = Model(inputs=[x2, f2], outputs=[z_m, z_l_v])

        if weights_file:
            self.autoencoder.load_weights(weights_file, by_name=True)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.encoderMV.load_weights(weights_file, by_name=True)

        self.autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                                 loss={'decoded_mean': vae_loss, 'decoded_mean_2': vae_loss})
        print(K.eval(self.autoencoder.optimizer.lr))

    # Encoder tower structure
    def _towers(self, x, f):
        # Tower 1
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)

        # Tower 2
        hf = Flatten(name='tower_2_flatten_enc')(f)
        hf = Dense(256, activation='relu', name='tower_2_dense_1')(hf)

        # Merge
        h = Concatenate()([h, hf])
        return Dense(512, activation='relu', name='dense_1')(h)

    def _encoderMeanVar(self, x, f, latent_rep_size):
        h = self._towers(x, f)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        return z_mean, z_log_var

    def _buildEncoder(self, x, f, latent_rep_size, max_length, max_length_features, features_weight, epsilon_std=0.01):
        h = self._towers(x, f)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        # this function is the main change.
        # essentially we mask the training data so that we are only allowed to apply
        #   future rules based on the current non-terminal
        def conditional(x_true, x_pred):
            most_likely = K.argmax(x_true)
            most_likely = tf.reshape(most_likely, [-1])  # flatten most_likely
            ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely), 1)  # index ind_of_ind with res
            ix2 = tf.cast(ix2, tf.int32)  # cast indices as ints
            M2 = tf.gather_nd(masks_K, ix2)  # get slices of masks_K with indices
            M3 = tf.reshape(M2, [-1, MAX_LEN, DIM])  # reshape them
            P2 = tf.multiply(K.exp(x_pred), M3)  # apply them to the exp-predictions
            P2 = tf.divide(P2, K.sum(P2, axis=-1, keepdims=True))  # normalize predictions
            return P2

        def vae_loss(x, x_decoded_mean):
            if K.int_shape(x_decoded_mean)[1] == max_length:
                x_decoded_mean = conditional(x, x_decoded_mean)  # we add this new function to the loss
                x = K.flatten(x)
                x_decoded_mean = K.flatten(x_decoded_mean)
                xent_loss = max_length * binary_crossentropy(x, x_decoded_mean)
            elif K.int_shape(x_decoded_mean)[1] == max_length_features:
                x = K.flatten(x)
                x_decoded_mean = K.flatten(x_decoded_mean)
                xent_loss = features_weight * mse(x, x_decoded_mean)
            else:
                raise ValueError('UNRECOGNIZED SHAPE')

            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

            return xent_loss + kl_loss

        return vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

    def _buildDecoder(self, z, latent_rep_size, max_length, max_length_features, charset_length):
        l = Dense(latent_rep_size, name='latent_input', activation='relu')(z)

        # Tower 1
        h = RepeatVector(max_length, name='repeat_vector')(l)
        h = GRU(512, return_sequences=True, name='gru_1', reset_after=False)(h)
        h = GRU(512, return_sequences=True, name='gru_2', reset_after=False)(h)
        h = GRU(512, return_sequences=True, name='gru_3', reset_after=False)(h)
        h = TimeDistributed(Dense(charset_length), name='decoded_mean')(h)

        # Tower 2
        hf = Dense(max_length_features, name='dense_tower_2', activation='sigmoid')(l)
        hf = Reshape((max_length_features, 1), name='decoded_mean_2')(hf)

        return h, hf

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=128, features_weight=10, learning_rate=0.0001):
        self.create(charset, weights_file=weights_file, latent_rep_size=latent_rep_size, learning_rate=learning_rate,
                    features_weight=features_weight)
