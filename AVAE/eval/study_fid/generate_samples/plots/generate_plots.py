import scipy.stats as stats
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools
import os


def get_hist_plot_for_annulus_data(annulus_samples, annulus_count, annulus_sample_count, save_image_path):
    colors = itertools.cycle(["r", "b", "g", "c", "m", "y"])
    fig, ax1 = plt.subplots()
    n_bins = 200
    for annulus_index in range(annulus_count):
        sample_start_index = annulus_index*annulus_sample_count
        sample_end_index = sample_start_index + annulus_sample_count
        cur_annulus_samples = annulus_samples[sample_start_index:sample_end_index]
        ax1.hist(cur_annulus_samples, bins=n_bins, histtype='stepfilled', color=next(colors)) #histtype='stepfilled'
    ax1.set_ylabel('sample_count', color='b')
    ax1.set_xlabel('distance')
    save_image_path = os.path.join(save_image_path)
    plt.savefig(save_image_path)
    plt.close(plt.gcf())


def get_hist_plot_for_data(annulus_samples, save_image_path):
    fig, ax1 = plt.subplots()
    n_bins = 2000
    ax1.hist(annulus_samples, bins=n_bins, histtype='stepfilled', color='b')
    ax1.set_ylabel('sample_count', color='b')
    ax1.set_xlabel('distance')
    save_image_path = os.path.join(save_image_path)
    plt.savefig(save_image_path)
    plt.close(plt.gcf())


def get_hist_plot_for_annulus_data_with_Chi2(latent_dim, chi2_scale, annulus_samples, annulus_count, annulus_sample_count, save_image_path, max_cdf=1-1e-05):
    colors = itertools.cycle(["r", "b", "g", "c", "m", "y"])
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    n_bins = 200
    for annulus_index in range(annulus_count):
        sample_start_index = annulus_index*annulus_sample_count
        sample_end_index = sample_start_index + annulus_sample_count
        cur_annulus_samples = annulus_samples[sample_start_index:sample_end_index]
        ax1.hist(cur_annulus_samples, bins=n_bins, histtype='stepfilled', color=next(colors)) #histtype='stepfilled'
    ax1.set_ylabel('sample_count', color='b')
    max_cdf_distsq = stats.chi2.ppf(max_cdf, df=latent_dim, loc=0, scale=chi2_scale)
    input_plot_chi2 = np.linspace(0, max_cdf_distsq, 50000).astype(np.float64)
    plot_chi2_pdf = stats.chi2.pdf(input_plot_chi2, latent_dim, loc=0, scale=chi2_scale).astype(np.float64)
    ax2.plot(input_plot_chi2, plot_chi2_pdf, 'r-', label='PDF_Chi2')
    ax2.set_xlabel('distance^2')
    ax2.set_ylabel('Chi2_PDF', color='r')
    save_image_path = os.path.join(save_image_path)
    plt.savefig(save_image_path)
    plt.close(plt.gcf())


def get_hist_plot_with_Chi2(latent_dim, chi2_scale, annulus_samples, save_image_path, max_cdf=1-1e-05):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    n_bins = 2000
    ax1.hist(annulus_samples, bins=n_bins, histtype='stepfilled', color='b') #histtype='stepfilled'
    ax1.set_ylabel('sample_count', color='b')
    max_cdf_distsq = stats.chi2.ppf(max_cdf, df=latent_dim, loc=0, scale=chi2_scale)
    input_plot_chi2 = np.linspace(0, max_cdf_distsq, 50000).astype(np.float64)
    plot_chi2_pdf = stats.chi2.pdf(input_plot_chi2, latent_dim, loc=0, scale=chi2_scale).astype(np.float64)
    ax2.plot(input_plot_chi2, plot_chi2_pdf, 'r-', label='PDF_Chi2')
    ax2.set_xlabel('distance^2')
    ax2.set_ylabel('Chi2_PDF', color='r')
    save_image_path = os.path.join(save_image_path)
    plt.savefig(save_image_path)
    plt.close(plt.gcf())
