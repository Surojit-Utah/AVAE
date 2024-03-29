import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__=="__main__":

    # dataset_name_list = ['MNIST', 'CelebA', 'CIFAR10']
    dataset_name_list = ['MNIST']
    eval_ids = np.arange(1, 2, 1).tolist()
    latent_dim_dict = {'MNIST': 128, 'CelebA': 128, 'CIFAR10': 128}
    color_dict = {'MNIST': 'b', 'CelebA': 'm', 'CIFAR10': 'g'}

    mode = 'generation'
    for dataset_name in dataset_name_list:
        # Log number of relevant axes
        dataset_color = color_dict[dataset_name]
        latent_dim = latent_dim_dict[dataset_name]
        log_dir = os.path.join('..', 'logs', dataset_name, mode, 'Dim_' + str(latent_dim))

        for run_id in eval_ids:

            relevance_stat_dir = os.path.join(log_dir, 'run_id_' + str(run_id), 'relevance_stat')
            scaled_est_var_grads_norm = np.load(os.path.join(relevance_stat_dir, 'scaled_est_var_grads_norm.npy'))
            sort_scaled_est_var_grads_norm = np.sort(scaled_est_var_grads_norm)[::-1]

            ##########################
            # Spectrum of the variance
            ##########################
            # Regular scale
            x_axis = np.arange(1, len(sort_scaled_est_var_grads_norm) + 1, 1)
            y_axis = sort_scaled_est_var_grads_norm
            plt.plot(x_axis, y_axis, label=dataset_name+' (L = ' + str(latent_dim) + ')', color=dataset_color)

    plt.legend()
    spec_mean_cov_image_path = os.path.join("weighted_variance.png")
    plt.savefig(spec_mean_cov_image_path)
    plt.close(plt.gcf())