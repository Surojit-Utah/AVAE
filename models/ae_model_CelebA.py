import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose, Lambda, ReLU, Reshape
from tensorflow.keras import regularizers


class Sampling():
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, axis_samples=10000, scatter_use_var=True):

        super(Sampling, self).__init__()
        self.axis_samples = axis_samples
        self.scatter_use_var = scatter_use_var

    @tf.function
    def get_sample(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @tf.function
    def kl_get_sample(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        z_mean_expand = tf.expand_dims(z_mean, axis=2)
        z_mean_repeat_axis = tf.repeat(z_mean_expand, repeats=[self.axis_samples], axis=-1)
        z_log_var_expand = tf.expand_dims(z_log_var, axis=2)
        z_log_var_repeat_axis = tf.repeat(z_log_var_expand, repeats=[self.axis_samples], axis=-1)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, self.axis_samples))
        samples_along_axis = z_mean_repeat_axis + tf.exp(0.5 * z_log_var_repeat_axis) * epsilon

        del z_mean_expand, z_mean_repeat_axis, z_log_var_expand, z_log_var_repeat_axis

        return samples_along_axis, z_log_var

    @tf.function
    def beta_get_sample(self, inputs):
        z_mean, z_log_var = inputs
        if self.scatter_use_var:
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            samples_along_axis = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        else:
            samples_along_axis = z_mean

        return samples_along_axis


class Encoder(tf.keras.Model):

    def __init__(self, latent_dim=128, num_filter=128, kernel_size=4, stride_size=2, conv_kernel_initializer_method='glorot_uniform', scatter_use_var=True, axis_samples=10000):

        super(Encoder, self).__init__()

        if conv_kernel_initializer_method=='he_normal':
            initializer = tf.keras.initializers.HeNormal(seed=0)
        elif conv_kernel_initializer_method=='glorot_uniform':
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self._Conv2D = Conv2D
        self._Dense = Dense

        self.latent_dim = latent_dim

        self.conv_layer_1 = self._Conv2D(num_filter, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)
        self.conv_layer_2 = self._Conv2D(num_filter*2, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)
        self.conv_layer_3 = self._Conv2D(num_filter*4, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)
        self.conv_layer_4 = self._Conv2D(num_filter*8, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)

        # The same goes here, a repeated batch normalization layer
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()

        # Linear layer to the latent space
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer_1 = self._Dense(self.latent_dim + self.latent_dim, activation='linear', name='latent_z')

        self.axis_samples = axis_samples
        self.scatter_use_var = scatter_use_var
        self.sampler = Sampling(axis_samples, scatter_use_var)

    # Remember to pass training=True in the training loop!
    # otherwise the batch_norm won't work
    @tf.function
    def call(self, inputs, use_batch_norm=False, training=False):

        # Pass the input throught the layers
        x = Lambda(lambda x: x * 2.0 - 1.0)(inputs)

        x = self.conv_layer_1(x)
        if use_batch_norm:
            x = self.batch_norm_1(x, training=training)
        x = ReLU()(x)

        x = self.conv_layer_2(x)
        if use_batch_norm:
            x = self.batch_norm_2(x, training=training)
        x = ReLU()(x)

        x = self.conv_layer_3(x)
        if use_batch_norm:
            x = self.batch_norm_3(x, training=training)
        x = ReLU()(x)

        x = self.conv_layer_4(x)
        if use_batch_norm:
            x = self.batch_norm_4(x, training=training)
        x = ReLU()(x)

        x = self.flatten_layer(x)
        x = self.dense_layer_1(x)

        return x


class Decoder(tf.keras.Model):

    def __init__(self, latent_dim=128, num_filter=128, kernel_size=4, stride_size=2, input_channels=3, reg_strength=1e-06, conv_kernel_initializer_method='glorot_uniform'):

        super(Decoder, self).__init__()

        if conv_kernel_initializer_method=='he_normal':
            initializer = tf.keras.initializers.HeNormal(seed=0)
        elif conv_kernel_initializer_method=='glorot_uniform':
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self._Dense = Dense
        self._Conv2DTranspose = Conv2DTranspose

        self.latent_dim = latent_dim

        self.encoder_last_channel_depth = num_filter*8
        self.dense_layer_1 = self._Dense(8*8*self.encoder_last_channel_depth)

        self.trans_conv_layer_1 = self._Conv2DTranspose(num_filter*4, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='linear', kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer=initializer)
        self.trans_conv_layer_2 = self._Conv2DTranspose(num_filter*2, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='linear', kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer=initializer)
        self.trans_conv_layer_3 = self._Conv2DTranspose(num_filter*1, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='linear', kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer=initializer)
        self.trans_conv_layer_4 = self._Conv2DTranspose(input_channels, (kernel_size, kernel_size), padding='same', activation='tanh', kernel_regularizer=regularizers.l2(reg_strength))

        # The same goes here, a repeated batch normalization layer
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()


    # Remember to pass training=True in the training loop!
    # otherwise the batch_norm won't work
    @tf.function
    def call(self, inputs, use_batch_norm=True, training=False):

        x = self.dense_layer_1(inputs)
        x = Reshape((8, 8, self.encoder_last_channel_depth))(x)

        x = self.trans_conv_layer_1(x)
        if use_batch_norm:
            x = self.batch_norm_1(x, training=training)
        x = ReLU()(x)

        x = self.trans_conv_layer_2(x)
        if use_batch_norm:
            x = self.batch_norm_2(x, training=training)
        x = ReLU()(x)

        x = self.trans_conv_layer_3(x)
        if use_batch_norm:
            x = self.batch_norm_3(x, training=training)
        x = ReLU()(x)

        x = self.trans_conv_layer_4(x)

        return x