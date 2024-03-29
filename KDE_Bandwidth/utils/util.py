import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

def plot_graphs(trial_dir, num_iter, est_bandwidth, bandwidth_update_array, learning_rate_decay):

    # Learning rate adjustment
    plt.figure()
    plt.plot(np.arange(num_iter), learning_rate_decay[:num_iter], label='Learning rate')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate decay')
    plt.savefig(os.path.join(trial_dir, 'Learning_rate_decay_' + str(num_iter) + '.png'))
    plt.close(plt.gcf())

    # Optimization of the bandwidth parameters
    plt.figure()
    num_par = len(est_bandwidth)
    for par_index in range(num_par):
        plt.plot(np.arange(num_iter), bandwidth_update_array[:num_iter, par_index])
    plt.xlabel('Iterations')
    plt.ylabel('Bandwidth update')
    savefile = os.path.join(trial_dir, 'BW_update_iter_' + str(num_iter) + '.png')
    plt.savefig(savefile)
    plt.close(plt.gcf())

    return


def save_numpy_array(trial_dir, bandwidth_update_array):

    np.save(os.path.join(trial_dir, "bandwidth_update.npy"), bandwidth_update_array)

    return