import tensorflow as tf
tf.keras.backend.set_floatx('float32')


@tf.function
def kld_loss(sample_mean, sample_log_var, alpha, beta):
    # kld part 1
    kld_1 = -1 - sample_log_var

    # kld part 2
    target_var = tf.math.divide(beta, alpha)
    target_var = tf.expand_dims(target_var, 0)
    scaled_val = tf.divide((tf.square(sample_mean) + tf.exp(sample_log_var)), target_var)
    kld_2 = target_var + scaled_val

    vae_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(kld_1 + kld_2, axis=1))

    return vae_loss


def kld_loss_wo_const(sample_mean, sample_log_var, alpha, beta):
    # kld part 1
    kld_1 = -sample_log_var

    # kld part 2
    target_var = tf.math.divide(beta, alpha)
    target_var = tf.expand_dims(target_var, 0)
    scaled_val = tf.divide((tf.square(sample_mean) + tf.exp(sample_log_var)), target_var)
    kld_2 = scaled_val

    vae_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(kld_1 + kld_2, axis=1))

    return vae_loss


@tf.function
def autoencoder_loss(x, logits):

    batch_size = tf.shape(x)[0]
    inputs = tf.reshape(x, (batch_size, -1))
    logits = tf.reshape(logits, (batch_size, -1))

    reconstruction = tf.reduce_mean(tf.reduce_sum((inputs - logits) ** 2, 1))

    return reconstruction

@tf.function
def autoencoder_ce_loss(x, logits):

    batch_size = tf.shape(x)[0]
    inputs = tf.reshape(x, (batch_size, -1))
    logits = tf.reshape(logits, (batch_size, -1))

    reconstruction = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits), axis=1))

    return reconstruction