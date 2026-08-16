import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from plots import generate_plots
from gen_samples import get_encoded_statistics, get_generated_samples
import argparse
from config.local_config import configurations
import imageio

###############
# Interpolation
###############
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def slerpolate(x, y, C, num_pts):
    alphas = np.linspace(0, 1.0, num_pts)
    if C is not None:
        a = np.linalg.cholesky(C)
        a_inv = np.linalg.inv(a)
        x_inv = np.dot(a_inv, x)
        y_inv = np.dot(a_inv, y)
        res_inv = [slerp(alpha, x_inv, y_inv) for alpha in alphas]
        res = np.dot(a, np.array(res_inv).T)
        # res = np.transpose(res_inv)
        return res
    else:
        return np.array([slerp(alpha, x, y) for alpha in alphas]).T


def interpolation_samples(sample_count, latent_vectors):
    random_test_pairs = np.random.choice(list(range(sample_count)) * 2, (sample_count, 2), replace=False)

    sample_latent_vectors = np.zeros((sample_count, latent_dim))
    max_iter = sample_count//batch_size
    print("Number of Iterations are : " + str(max_iter))
    for iter in range(max_iter):

        cur_pairs = random_test_pairs[(iter*batch_size):(iter*batch_size+batch_size)]

        lerp_minibatch = np.zeros((batch_size, latent_dim))
        for pair_index in range(batch_size):
            first_latent_code_index = cur_pairs[pair_index][0]
            second_latent_code_index = cur_pairs[pair_index][1]
            first_latent_code = latent_vectors[first_latent_code_index]
            second_latent_code = latent_vectors[second_latent_code_index]
            lerp = slerpolate(first_latent_code, second_latent_code, None, 3).T[1]
            # lerp = 0.5*(first_latent_code + second_latent_code)
            lerp_minibatch[pair_index] = lerp

        sampled_z = lerp_minibatch
        start_index = iter*batch_size
        end_index = start_index + batch_size
        sample_latent_vectors[start_index:end_index] = sampled_z

    return sample_latent_vectors


###################
# generated samples
###################
def get_std_normnal_samples(noise_dim, sample_count, variance=1):
    target_distribution_mean = np.zeros(noise_dim)
    target_distribution_cov = np.eye(noise_dim)*variance
    std_normal_data = np.random.multivariate_normal(target_distribution_mean, target_distribution_cov, sample_count).astype(np.float32)

    return std_normal_data


def set_seed(seed=0):
    np.random.seed(seed)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--gen_type", type=str, required=True, help="Can be one of the following options: generation, interpolation, reconstruction")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    mode = args.gen_type
    config = configurations[0][args.config_id]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    batch_size                  = config['batch_size']
    latent_dim                  = config['latent_dim']
    num_filter                  = config['num_filter']
    fid_samples                 = config['fid_samples']
    encoder_use_batch_norm      = config['encoder_use_batch_norm']
    decoder_use_batch_norm      = config['decoder_use_batch_norm']
    ori_bandwidth               = config['ori_bandwidth']
    basedir                     = os.path.join('..',  '..', '..', 'logs', dataset_name)
    eval_ids                    = np.arange(4, 6, 1).tolist()

    for run_id in eval_ids:

        # Produces encoded data and its associated statistics
        if mode=='reconstruction':
            latent_vectors, latent_stat = \
                get_encoded_statistics.get_statistics_encoded_data(dataset_name, run_id, basedir, latent_dim, encoder_use_batch_norm, num_filter, fid_samples, batch_size)
        elif mode=='interpolation':
            latent_vectors, latent_stat = \
                get_encoded_statistics.get_statistics_encoded_data(dataset_name, run_id, basedir, latent_dim, encoder_use_batch_norm, num_filter, fid_samples, batch_size)
            latent_vectors = interpolation_samples(fid_samples, latent_vectors)
        elif mode == 'generation':
            if model_name=='AVAE':
                alpha = np.sqrt(1 / (1 + ori_bandwidth ** 2)).astype(np.float32)
                bandwidth = ori_bandwidth * alpha
                target_var = 1 - bandwidth**2
            latent_vectors = get_std_normnal_samples(latent_dim, fid_samples, variance=target_var)

        ###############################################
        # Generate samples using encoded representation
        ###############################################
        gen_images_array = get_generated_samples.get_generated_data(dataset_name, run_id, basedir, latent_dim, decoder_use_batch_norm, num_filter, latent_vectors, batch_size)
        generated_image_dir = os.path.join('logs', dataset_name, mode, 'run_id_' + str(run_id))
        os.makedirs(generated_image_dir, exist_ok=True)
        if 'MNIST' in dataset_name:
            for image_index in range(gen_images_array.shape[0]):
                save_image = gen_images_array[image_index, :, :, :]
                gen_image_path = os.path.join(generated_image_dir, "generated_image_" + str(image_index) + ".jpg")
                imageio.imwrite(gen_image_path, save_image)
        elif 'DSprites' in dataset_name:
            for image_index in range(gen_images_array.shape[0]):
                save_image = gen_images_array[image_index, :, :, :]
                gen_image_path = os.path.join(generated_image_dir, "generated_image_" + str(image_index) + ".jpg")
                imageio.imwrite(gen_image_path, save_image)
        else:
            save_nparray_path = os.path.join(generated_image_dir, 'generated_images.npy')
            np.save(save_nparray_path, gen_images_array)