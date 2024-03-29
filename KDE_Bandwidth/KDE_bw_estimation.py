import argparse
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from optim.MLE_optim import Generate_KDE_Samples, Encoded_Distribution
from optim.MLE_optim import Target_Distribution, Expectation_Target_Distribution, objective
from utils.util import plot_graphs, save_numpy_array


def torch_init(to_device):
    cuda_avail = torch.cuda.is_available()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cpu")
    if cuda_avail and 'cuda' in to_device:
        device = torch.device(to_device)
        torch.cuda.set_device(device)

    return cuda_avail, device


# Define a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def main():

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument('--latent-dim-list', type=list_of_ints, required=True)
    parser.add_argument('--kde-samples-list', type=list_of_ints, required=True)
    parser.add_argument("--num-trial", type=int, default=3)
    parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.",
                        type=str, default="cuda:0")
    parser.add_argument("--debug_mode", type=bool, default=False)

    args = parser.parse_args()
    print(args)
    list_dimension = args.latent_dim_list                   # [40, 70 100]
    list_num_kde_samples = args.kde_samples_list            # [5000, 10000, 20000]


    _, device = torch_init(args.device)
    print("pytorch using device", device)

    # KDE bandwidth estimation configuration
    start_cdf = 0
    max_cdf = 0.95
    num_target_samples = 1000
    num_par = 1

    # Optimizer configuration
    num_iter = 1000
    learning_rate = 0.001
    adjust_lr_iter = 2000
    lr_decay_factor = 0.5

    # Exp config
    basedir = 'Output'
    num_trial = args.num_trial
    print_stat = 100
    iter_save_optim_state = 1000


    for dim_index in range(len(list_dimension)):

        dimension = list_dimension[dim_index]
        mean_vector = np.zeros(dimension)
        covariance_mat = np.eye(dimension)

        ###########
        # Dimension
        ###########
        result_dir = os.path.join(basedir, "Dim_" + str(dimension))
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)

        for num_kde_samples in list_num_kde_samples:

            #############
            # KDE samples
            #############
            sample_dir = os.path.join(result_dir, str(num_kde_samples)+'_samples')
            if not os.path.isdir(sample_dir):
                os.makedirs(sample_dir, exist_ok=True)

            ###########################
            # File Pointers for Logging
            ###########################
            file_name = "Experimental_SetUp_Dim_" + str(dimension) + '_KDE_' + str(num_kde_samples) + ".txt"
            file_path = os.path.join(sample_dir, file_name)
            if (os.path.exists(file_path)):
                print("Please check the setup....")
                print("Bandwidth estimation is done under this setting....")
                print(f"Dimension : {dimension} and KDE samples: {num_kde_samples}")
                print("Press enter to REDO the estimation....")
                print("To SKIP please change the input arguments....")
                input()

            exp_spec_file_ptr = open(file_path, "w")
            exp_spec_file_ptr.write("Experimental SetUp for Dim : " + str(dimension) + "\n")
            exp_spec_file_ptr.write("Number of trials           : " + str(num_trial) + "\n")
            exp_spec_file_ptr.write("Total KDE samples          : " + str(num_kde_samples) + "\n")
            exp_spec_file_ptr.write("Target Distr Samples       : " + str(num_target_samples) + "\n")
            exp_spec_file_ptr.write("Learning Rate              : " + str(learning_rate) + "\n")
            exp_spec_file_ptr.write("Learning Rate Adjust Iter  : " + str(adjust_lr_iter) + "\n")
            exp_spec_file_ptr.write("Learning Rate Decay Factor : " + str(lr_decay_factor) + "\n")
            exp_spec_file_ptr.write("Optim Iter                 : " + str(num_iter) + "\n\n")
            exp_spec_file_ptr.flush()

            final_bandwidth_update = np.zeros((num_trial, num_par))
            for trial_index in range(num_trial):

                ########################
                # Results for each trial
                ########################
                trial_dir = os.path.join(sample_dir, 'Trial_' + str(trial_index))
                if not os.path.isdir(trial_dir):
                    os.makedirs(trial_dir, exist_ok=True)

                print(f"Procesing trial {trial_index+1}....")

                ############################################
                # Initialization of the bandwidth parameters
                ############################################
                min_val = 1.0
                max_val = 1.5
                init_param = np.random.uniform(min_val, max_val, num_par)

                # Bandwidth, parameters to be optimized for maximization of JSD Loss
                param_tensor = torch.tensor(init_param, requires_grad=True)

                # Optimizer
                optim = torch.optim.Adam([param_tensor], lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=lr_decay_factor,
                                                                       patience=adjust_lr_iter,
                                                                       threshold=0.0001, threshold_mode='rel',
                                                                       cooldown=0, min_lr=0, eps=1e-08)
                objective_function = np.zeros(num_iter)
                expectation_target_distribution = np.zeros(num_iter)
                norm_grad = np.zeros(num_iter)
                learning_rate_decay = np.zeros(num_iter)
                bandwidth_update_array = np.empty((0, num_par))
                torch.autograd.set_detect_anomaly(True)

                ###########################
                # Generation of KDE samples
                ###########################
                multiple_kde_samples_set, list_num_kde_samples_per_annulus, list_annulus_distsq, kde_samples_distsq, start_annulus_distsq, annulus_width_distsq \
                    = Generate_KDE_Samples(sample_dir, dimension, num_kde_samples, num_par, start_cdf, max_cdf)
                multiple_kde_samples_set = multiple_kde_samples_set.to(device)

                ###########################
                # Weight of the KDE samples
                ###########################
                weight_KDE_samples = torch.ones([1, num_kde_samples]) * num_kde_samples
                weight_KDE_samples = weight_KDE_samples.to(device)

                for iter_index in tqdm(range(num_iter)):

                    ############################################
                    # Bandwidth associated with the KDE samples
                    # Computed by linear interpolation
                    ############################################
                    kde_bandwidth_tensor_start = torch.zeros((1, num_kde_samples))
                    kde_bandwidth_tensor_end = torch.zeros((1, num_kde_samples))
                    start_index = 0
                    for par_index in range(param_tensor.shape[0]):
                        cur_num_kde_samples_per_annulus = list_num_kde_samples_per_annulus[par_index]
                        end_index = start_index + cur_num_kde_samples_per_annulus
                        if par_index == (param_tensor.shape[0] - 1):
                            kde_bandwidth_tensor_start[0, start_index:end_index] = param_tensor[par_index]
                            kde_bandwidth_tensor_end[0, start_index:end_index] = param_tensor[par_index]
                            continue
                        kde_bandwidth_tensor_start[0, start_index:end_index] = param_tensor[par_index]
                        kde_bandwidth_tensor_end[0, start_index:end_index] = param_tensor[par_index + 1]
                        start_index = end_index

                    distsq_weight = torch.tensor(1 - np.divide((kde_samples_distsq - start_annulus_distsq), annulus_width_distsq))
                    kde_bandwidth_tensor = torch.mul(distsq_weight, kde_bandwidth_tensor_start) + torch.mul((1 - distsq_weight), kde_bandwidth_tensor_end)
                    kde_bandwidth_tensor = kde_bandwidth_tensor.to(device)

                    ###################################
                    # Samples for expectation operation
                    ###################################
                    # Samples from the Target Distribution
                    samples_target_distribution = torch.tensor(np.random.multivariate_normal(mean_vector, covariance_mat, num_target_samples).astype(np.float64))
                    samples_target_distribution = samples_target_distribution.to(device)

                    ########################################
                    # Optimization of the objective function
                    ########################################
                    # Objective function evaluation with new samples
                    objective_func = objective(multiple_kde_samples_set, kde_bandwidth_tensor, weight_KDE_samples,
                                               samples_target_distribution)

                    # Optimizing the parameters
                    optim.zero_grad()
                    objective_func.backward(retain_graph=True)
                    optim.step()
                    scheduler.step(objective_func)

                    ###################################
                    # Tracking the optimization process
                    ###################################
                    # Expectation over the target distribution
                    target_distribution_expectation = Expectation_Target_Distribution(multiple_kde_samples_set,
                                                                                      kde_bandwidth_tensor,
                                                                                      weight_KDE_samples,
                                                                                      samples_target_distribution)
                    expectation_target_distribution[
                        iter_index] = target_distribution_expectation.data.cpu().detach().numpy()

                    cur_objective_func = objective_func.data.cpu().detach().numpy()
                    grad_param = param_tensor.grad.data.cpu().detach().numpy()

                    objective_function[iter_index] = cur_objective_func
                    norm_grad[iter_index] = np.linalg.norm(grad_param)

                    cur_bandwidth = param_tensor.data.cpu().detach().numpy()
                    cur_bandwidth = np.expand_dims(cur_bandwidth, axis=0)
                    bandwidth_update_array = np.append(bandwidth_update_array, cur_bandwidth, axis=0)

                    learning_rate_decay[iter_index] = optim.param_groups[0]['lr']

                    ######################################################
                    # Active in debug mode
                    # Print the optimization results after some iterations
                    ######################################################
                    if args.debug_mode and (iter_index > 0 and iter_index % print_stat == 0):
                        # Evaluation of probability for samples from the target distribution
                        test_samples_tensor = torch.tensor(
                            np.random.multivariate_normal(mean_vector, covariance_mat, num_target_samples).astype(
                                np.float64))
                        test_samples_tensor = test_samples_tensor.to(device)
                        prob_encoded_distr = Encoded_Distribution(multiple_kde_samples_set, kde_bandwidth_tensor,
                                                                  weight_KDE_samples,
                                                                  test_samples_tensor).data.cpu().detach().numpy()
                        prob_target_distr = Target_Distribution(test_samples_tensor).data.cpu().detach().numpy()

                        print("Iter index                       : " + str(iter_index))
                        print("Number of dimensions             : " + str(dimension))
                        print("Number of KDE samples            : " + str(num_kde_samples))
                        print("Updated KDE bandwdith            : " + str(cur_bandwidth[0][0]))
                        print("Objective function               : " + str(objective_function[iter_index]))
                        print("Expectation w.r.t. Target Distr  : " + str(expectation_target_distribution[iter_index]))
                        print("Prob Encoded Distribution        : ")
                        print("Min Probability                  : " + str(np.min(prob_encoded_distr)))
                        print("Max Probability                  : " + str(np.max(prob_encoded_distr)))
                        print("Prob Target Distribution         : ")
                        print("Min Probability                  : " + str(np.min(prob_target_distr)))
                        print("Max Probability                  : " + str(np.max(prob_target_distr)))
                        print("Current learning rate            : " + str(learning_rate_decay[iter_index]))

                    # plots showing the bandwidth estimation over time
                    if iter_index > 0 and iter_index % iter_save_optim_state == 0:
                        cur_est_bandwidth = param_tensor.data.cpu().detach().numpy()
                        plot_graphs(trial_dir, iter_index, cur_est_bandwidth, bandwidth_update_array, learning_rate_decay)

                ########################################
                # Results at the end of the optimization
                ########################################
                # KDE bandwidth
                save_numpy_array(trial_dir, bandwidth_update_array)

                # plots showing the bandwidth estimation over time
                final_bandwidth = param_tensor.data.cpu().detach().numpy()
                plot_graphs(trial_dir, iter_index, final_bandwidth, bandwidth_update_array, learning_rate_decay)

                # Saving the final bandwidth estimate for plotting the error bars over multiple trials
                final_bandwidth_update[trial_index] = final_bandwidth


            # Statistics of the estimated bandwidth
            mean_bandwidth_update = np.mean(final_bandwidth_update, axis=0)[0]
            std_dev_bandwidth_update = np.std(final_bandwidth_update, axis=0)[0]
            exp_spec_file_ptr.write(f"Estimated bandiwidth over {num_trial} trials \n")
            exp_spec_file_ptr.write(str(np.around(mean_bandwidth_update, 2)) + ' \u00B1 ' + str(np.around(std_dev_bandwidth_update, 3)))
            exp_spec_file_ptr.flush()
            exp_spec_file_ptr.close()


main()
