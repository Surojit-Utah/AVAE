import os
import numpy as np
from latent_encoding import get_encoded_statistics
import argparse
from config.local_config import configurations
import scipy.stats as stats
import scipy.spatial.distance as distance


def get_epsilon(band_width, noise_dim, max_cdf=0.999):

    variance = np.round(band_width**2, 2)
    sq_dist_chi2 = stats.chi2.ppf(max_cdf, df=noise_dim, loc=0, scale=variance)
    # epsilon = np.exp(-sq_dist_chi2 / (2*variance))/(((2*np.pi)**(noise_dim/2))*(band_width**noise_dim))
    epsilon = np.exp(-sq_dist_chi2 / (2 * variance)) * ((1 / bandwidth) ** noise_dim)

    return epsilon


def whiten(X, fudge=1E-4):
    # the matrix X should be observations-by-components

    observations = X.shape[0]
    # get the covariance matrix
    Xcov = (np.matmul(X.T, X))/(observations - 1.0)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d + fudge))

    # whitening matrix
    W = np.matmul(np.matmul(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.matmul(X, W)

    return X_white, W


def entropy(band_width, samples_samples_dist_sq, dimension, max_cdf, use_eps=True):

    band_var = band_width ** 2
    data_size = samples_samples_dist_sq.shape[0]

    # Weights for samples with respect to each Kernel
    # B = np.exp(samples_samples_dist_sq / (-2.0 * band_var)) / (((2*np.pi) ** (dimension/2))*(band_width ** dimension))
    B = np.exp(samples_samples_dist_sq / (-2.0 * band_var)) / (band_width ** dimension)
    B_proc = B*(1-np.eye(data_size, dtype=np.uint8))

    Prob_Samples = np.sum(B_proc, axis=1)
    Prob_Samples /= (data_size - 1.0)

    if use_eps:
        FLOAT_EPS = get_epsilon(band_width, dimension, max_cdf)
        Prob_Samples = Prob_Samples + FLOAT_EPS

    Entropy = -np.log(Prob_Samples)

    return np.mean(Entropy)


def evaluate_entropy(q_samples_lagging_for_kde, noise_dim, bandwidth, data_size, max_cdf, whiten_data=False):

        # Whitening the Latent Values
        if whiten_data:
            q_samples_lagging_for_kde, _ = whiten(q_samples_lagging_for_kde)

        # Entropy of the Trained Model
        q_samples_lagging_for_kde_sq = distance.squareform(np.power(distance.pdist(q_samples_lagging_for_kde), 2))
        model_entropy = entropy(bandwidth, q_samples_lagging_for_kde_sq, noise_dim, max_cdf)

        # Entropy of the Standard Normal Distribution
        mean = np.zeros(noise_dim)
        cov = np.eye(noise_dim)
        std_normal_data = np.random.multivariate_normal(mean, cov, data_size)
        std_normal_data_dist_sq = distance.squareform(np.power(distance.pdist(std_normal_data), 2))
        std_normal_samples_entropy = entropy(bandwidth, std_normal_data_dist_sq, noise_dim, max_cdf)

        # Closed Form Standard Normal Entropy
        # target_var = (1-bandwidth**2)
        # cov = np.eye(latent_dim)*target_var
        cov = np.eye(latent_dim)
        # analytical_entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov))
        analytical_entropy = 0.5 * (noise_dim + np.log(np.linalg.det(cov)))

        return model_entropy, std_normal_samples_entropy, analytical_entropy


def set_seed(seed=0):
    np.random.seed(seed)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = configurations[0][args.config_id]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    batch_size                  = config['batch_size']
    latent_dim                  = config['latent_dim']
    num_filter                  = config['num_filter']
    entropy_samples             = config['entropy_samples']
    encoder_use_batch_norm      = config['encoder_use_batch_norm']
    ori_bandwidth               = config['ori_bandwidth']
    alpha                       = np.sqrt(1/(1+ori_bandwidth**2))
    bandwidth                   = ori_bandwidth*alpha
    max_cdf_epsilon             = 1 - 5*1e-4
    whiten_data                 = True
    basedir                     = os.path.join('..', '..', 'logs', dataset_name)
    eval_ids                    = np.arange(1, 6, 1).tolist()


    entropy_scores = np.zeros((3, len(eval_ids)))
    entropy_index = 0
    for run_id in eval_ids:
        # different std normal samples
        set_seed(run_id-1)

        # latent encodings
        latent_vectors, latent_stat = \
            get_encoded_statistics.get_statistics_encoded_data(dataset_name, run_id, basedir, latent_dim,
                                                               encoder_use_batch_norm, num_filter, entropy_samples,
                                                               batch_size)
        # Entropy computation
        model_entropy, std_normal_samples_entropy, analytical_entropy = evaluate_entropy(latent_vectors, latent_dim, bandwidth, entropy_samples, max_cdf_epsilon, whiten_data=whiten_data)

        entropy_scores[0, entropy_index] = model_entropy
        entropy_scores[1, entropy_index] = std_normal_samples_entropy
        entropy_scores[2, entropy_index] = analytical_entropy
        entropy_index += 1

    log_dir = os.path.join('logs', dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'entropy_stat.txt')
    log_fileptr = open(log_filepath, 'w')
    for index in range(3):
        entropy_score = entropy_scores[index, :]
        avg_entropy_score = np.around(np.mean(entropy_score), 2)
        stddev_entropy_score = np.around(np.std(entropy_score), 2)
        if index==0:
            log_fileptr.write('Model entropy                : ' + str(avg_entropy_score) + ' \u00B1 ' + str(stddev_entropy_score) + '\n')
            log_fileptr.write(str(np.around(entropy_score, 2)) + '\n\n')
        elif index==1:
            log_fileptr.write('Std normal samples entropy   : ' + str(avg_entropy_score) + ' \u00B1 ' + str(stddev_entropy_score) + '\n')
            log_fileptr.write(str(np.around(entropy_score, 2)) + '\n\n')
        if index==2:
            log_fileptr.write('Analytical entropy           : ' + str(avg_entropy_score) + ' \u00B1 ' + str(stddev_entropy_score) + '\n')
            log_fileptr.write(str(np.around(entropy_score, 2)) + '\n\n')
        log_fileptr.flush()
    log_fileptr.close()