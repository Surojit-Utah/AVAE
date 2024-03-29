import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose, ReLU, Reshape
from tensorflow.keras import regularizers


class Encoder(tf.keras.Model):

    def __init__(self, latent_dim=128, num_filter=128, kernel_size=4, stride_size=2, conv_kernel_initializer_method='glorot_uniform'):

        super(Encoder, self).__init__()

        if conv_kernel_initializer_method=='he_normal':
            initializer = tf.keras.initializers.HeNormal(seed=0)
        elif conv_kernel_initializer_method=='glorot_uniform':
            initializer = tf.keras.initializers.GlorotUniform(seed=0)
        elif conv_kernel_initializer_method=='random_normal':
            initializer = tf.random_normal_initializer()

        self._Conv2D = Conv2D
        self._Dense = Dense
        self.num_filter = num_filter
        self.latent_dim = latent_dim

        self.conv_layer_1 = self._Conv2D(num_filter*1, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)
        self.conv_layer_2 = self._Conv2D(num_filter*1, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)
        self.conv_layer_3 = self._Conv2D(num_filter*2, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)
        self.conv_layer_4 = self._Conv2D(num_filter*2, (kernel_size, kernel_size), padding='same', activation='linear', strides=(stride_size, stride_size), kernel_initializer=initializer)

        # The same goes here, a repeated batch normalization layer
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()

        # Linear layer to the latent space
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer_1 = self._Dense(num_filter*8, activation='linear')
        self.dense_layer_2 = self._Dense(self.latent_dim, activation='linear')

    # Remember to pass training=True in the training loop!
    # otherwise the batch_norm won't work
    @tf.function
    def call(self, inputs, use_batch_norm=False, training=False):

        x = self.conv_layer_1(inputs)
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
        # x = Reshape((4*4*self.num_filter*2, ))(x)

        x = self.dense_layer_1(x)
        if use_batch_norm:
            x = self.batch_norm_5(x, training=training)
        x = ReLU()(x)

        x = self.dense_layer_2(x)

        return x


class Decoder(tf.keras.Model):

    def __init__(self, latent_dim=128, num_filter=128, kernel_size=4, stride_size=2, input_channels=3, reg_strength=1e-06, conv_kernel_initializer_method='glorot_uniform'):

        super(Decoder, self).__init__()

        if conv_kernel_initializer_method=='he_normal':
            initializer = tf.keras.initializers.HeNormal(seed=0)
        elif conv_kernel_initializer_method=='glorot_uniform':
            initializer = tf.keras.initializers.GlorotUniform(seed=0)
        elif conv_kernel_initializer_method=='random_normal':
            initializer = tf.random_normal_initializer()

        self.num_filter = num_filter
        self._Dense = Dense
        self._Conv2DTranspose = Conv2DTranspose

        self.latent_dim = latent_dim

        self.input_channels = num_filter*8
        self.dense_layer_1 = self._Dense(self.input_channels, activation='linear')
        self.dense_layer_2 = self._Dense(4*4*self.num_filter*2, activation='linear')

        self.trans_conv_layer_1 = self._Conv2DTranspose(num_filter*2, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='linear', kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer=initializer)
        self.trans_conv_layer_2 = self._Conv2DTranspose(num_filter*1, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='linear', kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer=initializer)
        self.trans_conv_layer_3 = self._Conv2DTranspose(num_filter*1, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='linear', kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer=initializer)
        self.trans_conv_layer_4 = self._Conv2DTranspose(input_channels, (kernel_size, kernel_size), padding='same', strides=(stride_size, stride_size), activation='sigmoid', kernel_regularizer=regularizers.l2(reg_strength))

        # The same goes here, a repeated batch normalization layer
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()


    # Remember to pass training=True in the training loop!
    # otherwise the batch_norm won't work
    @tf.function
    def call(self, inputs, use_batch_norm=False, training=False):
        x = self.dense_layer_1(inputs)
        if use_batch_norm:
            x = self.batch_norm_1(x, training=training)
        x = ReLU()(x)

        x = self.dense_layer_2(x)
        if use_batch_norm:
            x = self.batch_norm_2(x, training=training)
        x = ReLU()(x)
        x = Reshape((4, 4, self.num_filter*2))(x)

        x = self.trans_conv_layer_1(x)
        if use_batch_norm:
            x = self.batch_norm_3(x, training=training)
        x = ReLU()(x)

        x = self.trans_conv_layer_2(x)
        if use_batch_norm:
            x = self.batch_norm_4(x, training=training)
        x = ReLU()(x)

        x = self.trans_conv_layer_3(x)
        if use_batch_norm:
            x = self.batch_norm_5(x, training=training)
        x = ReLU()(x)

        x = self.trans_conv_layer_4(x)

        return x