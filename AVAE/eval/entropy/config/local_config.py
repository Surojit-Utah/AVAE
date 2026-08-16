configurations = \
{0: {'model_name': "AVAE",
     'dataset_name': "CIFAR10",
     'batch_size': 100,
     'epochs': 100,
     'latent_dim': 128,
     'num_filter': 128,
     'kde_samples': 10000,
     'update_q_iter_count': 10,         # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1,    # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 5,                     # "patience window for the LR schedular"
     'factor': 0.5,                     # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 1.12,             # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-01,      # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'fid_samples': 10000,
     'entropy_samples': 10000,
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
    'fid_samples': 10000,
    'entropy_samples': 10000,
    },
 2: {'model_name': "AVAE",
     'dataset_name': "MNIST",
     'batch_size': 100,
     'epochs': 50,
     'latent_dim': 16,
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
     'ori_bandwidth': 0.72,  # "KDE bandwidth"\
     'max_cdf_epsilon': 1 - 1e-04,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'fid_samples': 10000,
     'entropy_samples': 10000,
     },
 8: {'model_name': "AVAE",
     'dataset_name': "CIFAR10_Fixed_KDE",
     'batch_size': 100,
     'epochs': 100,
     'latent_dim': 128,
     'num_filter': 128,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1000,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 10,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 1.12,
     'max_cdf_epsilon': 1 - 1e-01,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 200,
     'fid_samples': 10000,
     'entropy_samples': 10000,
     },
 9: {'model_name': "AVAE",
     'dataset_name': "MNIST_Fixed_KDE",
     'batch_size': 100,
     'epochs': 50,
     'latent_dim': 16,
     'num_filter': 64,
     'kde_samples': 10000,
     'update_q_iter_count': 10,  # "iterations to update the KDE samples"
     'update_KDE_epoch_fraction': 1000,  # "epochs to shuffle KDE samples with training data"
     'print_every_epoch': 1,
     'save_every_epoch': 10,
     'dec_reg_strength': 0,
     'learning_rate': 5e-04,
     'patience': 5,  # "patience window for the LR schedular"
     'factor': 0.5,  # "reduce the learning rate by 0.5 beyond the patience window"
     'ori_bandwidth': 0.72,  # "KDE bandwidth"
     'max_cdf_epsilon': 1 - 1e-06,  # used for computing the epsilon in KL divergence
     'encoder_use_batch_norm': True,
     'decoder_use_batch_norm': True,
     'train_data_noise': False,
     'train_from_checkpoint': False,
     'print_model_summary': False,
     'conv_kernel_initializer_method': 'he_normal',
     'sigma_init_val': 1000,
     'fid_samples': 10000,
     'entropy_samples': 10000,
     },
 },
