configurations = \
{0: {'model_name': "AVAE",
     'dataset_name': "CIFAR10",
     'batch_size': 100,
     'epochs': 100,
     'latent_dim': 90, #128, This change is done for Run_7
     'num_filter': 128,
     'kde_samples': 10000,
     'update_q_iter_count': 10,         # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,    # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 10,                     # "patience window for the LR schedular"
     'factor': 0.5,                     # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 1.09, #1.12, This change is done for Run_7             # "KDE bandwidth"
     'max_cdf_epsilon': 1 - 2e-01, #1 - 1e-01     # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 200,
     'fid_samples': 10000,
},
1: {'model_name': "AVAE",
     'dataset_name': "CelebA",
     'batch_size': 100,
     'epochs': 50,
     'latent_dim': 64,
     'num_filter': 64,
     'kde_samples': 20000,
     'update_q_iter_count': 10,         # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,    # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 5,                     # "patience window for the LR schedular"
     'factor': 0.5,                     # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 1.02,             # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-02,      # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
    'sigma_init_val': 1000,
    'fid_samples': 10000,
    },
 2: {'model_name': "AVAE",
     'dataset_name': "MNIST",
     'batch_size': 100,
     'epochs': 50,
     'latent_dim': 8, #16,
     'num_filter': 64,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 5,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.54, #0.72,  # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 200, #1000,
     'fid_samples': 10000,
     },
 3: {'model_name': "AVAE",
     'dataset_name': "DSprites",
     'batch_size': 100,
     'epochs': 35,
     'latent_dim': 6,
     'num_filter': 32,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 5,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.47, # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': False,
     'decoder_use_batch_norm': False,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 200,
     'fid_samples': 10000,
     },
 4: {'model_name': "AVAE",
     'dataset_name': "Shapes3D",
     'batch_size': 100,
     'epochs': 60,
     'latent_dim': 6,
     'num_filter': 32,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 100,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.47, # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': False,
     'decoder_use_batch_norm': False,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 1000,
     'fid_samples': 10000,
     },
 5: {'model_name': "AVAE",
     'dataset_name': "Shapes3D_dim_10",
     'batch_size': 100,
     'epochs': 60,
     'latent_dim': 10,
     'num_filter': 32,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 100,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.6, # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': False,
     'decoder_use_batch_norm': False,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 1000,
     'fid_samples': 10000,
     },
 6: {'model_name': "AVAE",
     'dataset_name': "Shapes3D_dim_2",
     'batch_size': 100,
     'epochs': 60,
     'latent_dim': 2,
     'num_filter': 32,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 100,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.26, # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': False,
     'decoder_use_batch_norm': False,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 1000,
     'fid_samples': 10000,
     },
 7: {'model_name': "AVAE",
     'dataset_name': "MNIST_dim_2",
     'batch_size': 100,
     'epochs': 50,
     'latent_dim': 2,
     'num_filter': 64,
     'kde_samples': 5000, #2000, #10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 10,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.29, #0.34, #0.26, #0.72,  # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 200, #1000,
     'fid_samples': 10000,
     },
 },