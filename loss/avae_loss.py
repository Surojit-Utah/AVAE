import tensorflow as tf
# tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float32')
TARGET_FLOAT_EPS = 1e-30


@tf.function
def logbase(x, base):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
  return numerator / denominator


@tf.function
def row_pairwise_distances(x, y, dist_mat=None):
    '''
    Direct row-wise

    This is numerically stable and has a lower memory footprint
    than expanded_pairwise_distances but it is far slower than approach pairwise_distances.
    '''

    sq_dist_list = []
    for i, row in enumerate(tf.split(x, num_or_size_splits=x.shape[0])):
        r_v = tf.cast(tf.broadcast_to(row, y.shape), tf.float32)
        sq_diff = (r_v - y) ** 2
        sq_dist = tf.reduce_sum(sq_diff, axis=1)
        sq_dist_reshaped = tf.reshape(sq_dist, [1, -1])
        sq_dist_list.append(sq_dist_reshaped)

    dist_mat = tf.squeeze(tf.stack(sq_dist_list))

    return dist_mat


@tf.function
def kde_for_samples(q_samples,
                    # mini batch samples from the source distribution Q, these will require gradients for updates
                    q_samples_lagging_for_kde,
                    # lagging samples from Q to construct the kde, these will not require gradients
                    kernel_band_width
                    # Standard Deviation of the Gaussian Kernel
                    ):

    """
    Compute the KDE estimate for multiple samples

    """
    dimension = tf.cast(tf.convert_to_tensor(q_samples_lagging_for_kde.shape[1]), tf.float32)
    num_kde_samples = tf.cast(tf.convert_to_tensor(q_samples_lagging_for_kde.shape[0]), tf.float32)
    kernel_band_width = tf.cast(tf.convert_to_tensor(kernel_band_width), tf.float32)

    # compute pairwise distances
    z_zlag_distances_sqr = row_pairwise_distances(q_samples, q_samples_lagging_for_kde)

    # gaussian kernel for each pair of z and zlagging
    gauss_z_zlag = tf.exp(-1.0 * z_zlag_distances_sqr / (2.0 * (kernel_band_width ** 2)))

    # compute Q(z) for each z
    q_z = tf.reduce_sum(gauss_z_zlag, axis=1)

    # adjust for the std normalization of the kernel and the number of samples
    q_z = q_z * (1.0 / ((((kernel_band_width**2))**(dimension/2)) * num_kde_samples))

    return q_z



@tf.function
def standard_normal_for_samples(q_samples,
                                target_gaussian_std=1.0
                                ):
    """
    Compute the standard normal prob for multiple samples

    """

    dimension = tf.cast(tf.convert_to_tensor(q_samples.shape[1]), tf.float32)
    target_gaussian_std = tf.cast(tf.convert_to_tensor(target_gaussian_std), tf.float32)

    # compute P(z) for each z, i.e. against the standard normal
    normalization_const = (1.0 / ((target_gaussian_std)**(dimension/2)))
    z_distances_sqr = tf.reduce_sum((q_samples**2), axis=1)
    p_z = tf.exp(-1.0 * z_distances_sqr / (2.0 * (target_gaussian_std**2))) * normalization_const

    return p_z


@tf.function
def compute_kde_expectation(q_samples,
                            # mini batch samples from the source distribution Q, these will require gradients for updates
                            q_samples_lagging_for_kde,
                            # lagging samples from Q to construct the kde, these will not require gradients
                            kernel_band_width,
                            # Bandwidth of the Kernel
                            target_gaussian_std = 1.0,
                            # Standard Deviation of the Std. Gaussian
                            encoded_epsilon=TARGET_FLOAT_EPS,
                            target_epsilon=TARGET_FLOAT_EPS
                            ):
    """
    Compute the expectation of the KDE estimate of log(KDE/(KDE + normal) = log Q / (P+Q)

    KL[Q|M] where M = (P+Q)/2
    """

    num_q_samples = tf.cast(tf.convert_to_tensor(q_samples.shape[0]), tf.float32)
    encoded_epsilon = tf.cast(tf.convert_to_tensor(encoded_epsilon), tf.float32)
    target_epsilon = tf.cast(tf.convert_to_tensor(target_epsilon), tf.float32)

    # compute Q(z) for each z, i.e. against the kde estimate
    q_z = kde_for_samples(q_samples, q_samples_lagging_for_kde, kernel_band_width)

    # compute P(z) for each z, i.e. against the standard normal
    p_z = standard_normal_for_samples(q_samples, target_gaussian_std)

    # compute log Q/(P+Q) - add things up for the expectation
    log_Q_by_PQ = logbase((q_z + encoded_epsilon), base=10) - logbase((p_z + target_epsilon), base=10)

    # accumulate for expectation computation
    E_log_Q_by_PQ = tf.reduce_sum(log_Q_by_PQ)

    # normalize for data size/samples
    E_log_Q_by_PQ /= num_q_samples

    return E_log_Q_by_PQ


@tf.function
def autoencoder_loss(x, logits, sigma_sq):

    batch_size = tf.shape(x)[0]
    inputs = tf.reshape(x, (batch_size, -1))
    logits = tf.reshape(logits, (batch_size, -1))

    reconstruction = tf.reduce_mean(tf.reduce_sum((inputs - logits) ** 2 / sigma_sq, 1))
    reconstruction_wo_const = tf.reduce_mean(tf.reduce_sum((inputs - logits)**2, 1))

    return reconstruction, reconstruction_wo_const


@tf.function
def autoencoder_ce_loss(x, logits, sigma_sq):

    batch_size = tf.shape(x)[0]
    inputs = tf.reshape(x, (batch_size, -1))
    logits = tf.reshape(logits, (batch_size, -1))

    reconstruction = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits), 1))/sigma_sq
    reconstruction_wo_const = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits), 1))

    return reconstruction, reconstruction_wo_const


@tf.function
def kld_loss_computation(latent_encoding, kde_samples, kde_bandwidth, ENCODED_FLOAT_EPS):
    cur_kde_expectation = compute_kde_expectation(latent_encoding, kde_samples, kde_bandwidth, encoded_epsilon=ENCODED_FLOAT_EPS)
    return cur_kde_expectation