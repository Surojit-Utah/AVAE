import numpy as np
import scipy.stats as stats
import torch
import math
from scipy.stats import chi2
FLOAT_EPS = 1e-100
torch.set_default_dtype(torch.float64)

def Generate_KDE_Samples(result_dir, dimension, num_kde_samples, num_par, start_cdf, max_cdf):

    list_annulus_distsq = list()
    start_annulus_distsq = np.zeros((1, num_kde_samples))
    annulus_width_distsq = np.zeros((1, num_kde_samples))
    kde_samples_distsq = np.zeros((1, num_kde_samples))
    kde_samples_tensor = torch.zeros([num_kde_samples, dimension])
    kde_samples_density_per_annulus = list()
    list_sample_count_per_annulus = list()


    mean_vector = np.zeros(dimension)
    covariance_mat = np.eye(dimension)
    kde_samples = np.random.multivariate_normal(mean_vector, covariance_mat, num_kde_samples)
    kde_samples_mag_sq = np.sum(np.power(kde_samples, 2), axis=1)


    # Generating samples within the first annulus
    prev_cdf_distsq = start_cdf_distsq = stats.chi2.ppf(start_cdf, df=dimension)
    list_annulus_distsq.append(start_cdf_distsq)
    start_index = 0

    if num_par>1:
        # Step size of the CDF, related to the number of parameters
        cdf_step_size = (max_cdf - start_cdf)/(num_par-1)
        for cur_cdf_val in np.arange(start_cdf+cdf_step_size, max_cdf+cdf_step_size-1e-06, cdf_step_size).tolist():

            cur_cdf_distsq = stats.chi2.ppf(cur_cdf_val, df=dimension)
            list_annulus_distsq.append(cur_cdf_distsq)

            sample_indices_within_cur_annulus = np.where(np.logical_and((prev_cdf_distsq <= kde_samples_mag_sq), (kde_samples_mag_sq < cur_cdf_distsq)))[0]
            cur_annulus_samples = kde_samples[sample_indices_within_cur_annulus]
            distance_square = np.sum(np.power(cur_annulus_samples, 2), axis=1)
            num_samples_cur_annulus = len(sample_indices_within_cur_annulus)
            list_sample_count_per_annulus.append(num_samples_cur_annulus)

            volume1 = prev_cdf_distsq
            volume2 = cur_cdf_distsq
            annulus_vol = volume2 - volume1

            end_index = start_index + num_samples_cur_annulus
            kde_samples_tensor[start_index:end_index] = torch.tensor(cur_annulus_samples, dtype=torch.float64)
            kde_samples_distsq[0, start_index:end_index] = distance_square
            start_annulus_distsq[0, start_index:end_index] = list_annulus_distsq[-2]
            annulus_width_distsq[0, start_index:end_index] = list_annulus_distsq[-1] - list_annulus_distsq[-2]

            cur_annulus_sample_density = num_samples_cur_annulus/annulus_vol
            kde_samples_density_per_annulus.append(cur_annulus_sample_density)

            prev_cdf_distsq = cur_cdf_distsq
            start_index = end_index

    max_cdf_distsq = stats.chi2.ppf(1-1e-03, df=dimension)
    list_annulus_distsq.append(max_cdf_distsq)
    sample_indices_within_cur_annulus = np.where(kde_samples_mag_sq > prev_cdf_distsq)[0]
    cur_annulus_samples = kde_samples[sample_indices_within_cur_annulus]
    distance_square = np.sum(np.power(cur_annulus_samples, 2), axis=1)
    num_samples_cur_annulus = len(sample_indices_within_cur_annulus)
    list_sample_count_per_annulus.append(num_samples_cur_annulus)

    volume1 = prev_cdf_distsq
    volume2 = max_cdf_distsq
    annulus_vol = volume2 - volume1

    end_index = start_index + num_samples_cur_annulus
    kde_samples_tensor[start_index:end_index] = torch.tensor(cur_annulus_samples, dtype=torch.float64)
    kde_samples_distsq[0, start_index:end_index] = distance_square
    start_annulus_distsq[0, start_index:end_index] = 0
    annulus_width_distsq[0, start_index:end_index] = distance_square

    cur_annulus_sample_density = num_samples_cur_annulus/annulus_vol
    kde_samples_density_per_annulus.append(cur_annulus_sample_density)

    if num_par>1:
        filename = 'KDE_sample_count_' + str(dimension) + '_' + str(num_par) + '_' + str(num_kde_samples) + '.png'
        filepath = os.path.join(result_dir, filename)
        if(not os.path.exists(filepath)):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            input_plot_chi2 = np.linspace(0, max_cdf_distsq, 50000).astype(np.float64)
            plot_chi2_pdf = chi2.pdf(input_plot_chi2, dimension, loc=0, scale=1).astype(np.float64)
            pdf_plot, = ax1.plot(input_plot_chi2, plot_chi2_pdf, 'g-', label='PDF_Chi2',)
            ax1.set_xlabel('distance^2')
            ax1.set_ylabel('Chi2_PDF', color='g')

            list_annulus_distsq_plot = list()
            for index in range(len(list_annulus_distsq)-1):
                list_annulus_distsq_plot.append(0.5*(list_annulus_distsq[index] + list_annulus_distsq[index+1]))
            sample_count_plot_1, = ax2.plot(np.round(np.asarray(list_annulus_distsq_plot), 1), np.asarray(kde_samples_density_per_annulus), 'r-', label='KDE_sample_density')
            sample_count_plot_2, = ax2.plot(np.round(np.asarray(list_annulus_distsq_plot), 1), np.asarray(kde_samples_density_per_annulus), 'bo')
            sample_count_plot_3, = ax2.plot(np.round(np.asarray(list_annulus_distsq), 1), np.asarray([0]+kde_samples_density_per_annulus), 'm*', label='BW_par_pos')
            ax2.set_ylabel('sample_density', color='r')
            plt.legend(handles=[pdf_plot, sample_count_plot_1, sample_count_plot_3])
            plt.savefig(filepath)
            plt.close(plt.gcf())

    return kde_samples_tensor, list_sample_count_per_annulus, list_annulus_distsq, kde_samples_distsq, start_annulus_distsq, annulus_width_distsq


def Encoded_Distribution(kde_samples_set, kde_bandwidth_tensor, weight_KDE_samples, test_samples_tensor):

    dimension = kde_samples_set.shape[1]
    pairwise_distance = torch.cdist(test_samples_tensor, kde_samples_set)
    sq_dist_mat = torch.pow(pairwise_distance, 2)
    scaled_exp_dist = torch.exp(torch.div(sq_dist_mat, (-2.0*(kde_bandwidth_tensor**2))))
    scaling_factor = torch.pow(kde_bandwidth_tensor, dimension)*weight_KDE_samples
    encoded_distribution = torch.sum(torch.div(scaled_exp_dist, scaling_factor), axis=1)
    encoded_distribution = torch.reshape(encoded_distribution, (encoded_distribution.shape[0], 1))

    return encoded_distribution


def Target_Distribution(test_samples_tensor):
    dimension = test_samples_tensor.shape[1]
    target_bandwidth = 1.0
    target_var = target_bandwidth ** 2
    sq_dist_arr = torch.sum(torch.pow(test_samples_tensor, 2), axis=1)
    target_distribution = (torch.exp(sq_dist_arr / (-2.0 * target_var)) / (target_bandwidth ** dimension))
    target_distribution = torch.reshape(target_distribution, (target_distribution.shape[0], 1))

    return target_distribution


def Expectation_Target_Distribution(kde_samples, kde_bandwidth_tensor, weight_KDE_samples, samples_target_distribution):

    # Probability of the samples for encoded and target distribution
    encoded_distribution = Encoded_Distribution(kde_samples, kde_bandwidth_tensor, weight_KDE_samples, samples_target_distribution)
    target_distribution = Target_Distribution(samples_target_distribution)

    # Ratio of the probability for samples from encoded distribution
    distribution_ratio = -torch.mean(torch.log2((encoded_distribution + FLOAT_EPS) / (target_distribution + FLOAT_EPS)))

    return distribution_ratio


def objective(kde_samples, kde_bandwidth_tensor, weight_KDE_samples, samples_target_distribution):

    # Objective is to maximize the JSD between the Encoded and Target distribution
    min_KLD = Expectation_Target_Distribution(kde_samples, kde_bandwidth_tensor, weight_KDE_samples, samples_target_distribution)

    return min_KLD
