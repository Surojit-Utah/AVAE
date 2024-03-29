import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.optimizers import Adam
from lr_schedular.reduce_lr_plateau import CustomReduceLRoP
from data.dataloader_CelebA import dataloader_celeba
from data.dataloader_CIFAR10 import dataloader_cifar10
from data.dataloader_MNIST import dataloader_mnist
from data.dataloader_DSprites import dataloader_dsprites
from data.dataloader_Shapes3D import dataloader_shapes3d
from config.local_config import configurations
from models import ae_model_CelebA, ae_model_CIFAR10, ae_model_MNIST, ae_model_DSprites, ae_model_Shapes3D
from train import trainer
import os
import numpy as np
import random
import nvidia_smi
import argparse
import socket
TARGET_FLOAT_EPS = 1e-30


def select_GPU(min_gpu_mem_frac=0.9):
    hostname = socket.gethostname()
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    for device_index in range(device_count):
        if device_index==1 and 'blackjack' in hostname:
            continue
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        print("Total memory:", info.total)
        print("Free memory:", info.free)
        print("Used memory:", info.used)

        if info.free > min_gpu_mem_frac*(info.total):
            use_gpu = device_index
            os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)
            break
    nvidia_smi.nvmlShutdown()

    # Allow memory growth for the selected GPU
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    return use_gpu, info.free


def set_seed(seed=0):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(seed)


def main():

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_id = args.run_id
    set_seed(run_id-1)
    use_gpu, mem_free = select_GPU()
    print("Selected GPU for training : " + str(use_gpu) + " with available memory : " + str(mem_free//(1024*1024*1024)))

    config = configurations[0][args.config_id]

    ########################
    # Some Network Constants
    ########################
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    batch_size                  = config['batch_size']
    latent_dim                  = config['latent_dim']
    num_filter                  = config['num_filter']
    epochs                      = config['epochs']
    print_every_epoch           = config['print_every_epoch']
    save_every_epoch            = config['save_every_epoch']
    kde_samples                 = config['kde_samples']
    update_q_iter_count         = config['update_q_iter_count']
    update_KDE_epoch_fraction   = config['update_KDE_epoch_fraction']
    dec_reg_strength            = config['dec_reg_strength']
    learning_rate               = config['learning_rate']
    patience                    = config['patience']
    factor                      = config['factor']
    max_cdf_epsilon             = config['max_cdf_epsilon']
    encoder_use_batch_norm      = config['encoder_use_batch_norm']
    decoder_use_batch_norm      = config['decoder_use_batch_norm']
    train_data_noise            = config['train_data_noise']
    train_from_checkpoint       = config['train_from_checkpoint']
    print_model_summary         = config['print_model_summary']
    conv_kernel_initializer_method = config['conv_kernel_initializer_method']
    sigma_init_val              = config['sigma_init_val']
    ori_bandwidth               = config['ori_bandwidth']
    alpha                       = np.sqrt(1/(1+ori_bandwidth**2)).astype(np.float32)
    bandwidth                   = ori_bandwidth*alpha
    Guassian_Prior_Std_Dev      = np.sqrt(1 - bandwidth**2).astype(np.float32)
    save_model_epochs           = [epoch_num for epoch_num in range(0, epochs, 50)][1:]


    ############################
    # Autoencoder model
    # Optimizer and LR schedular
    # Checkpoint
    ############################
    if 'MNIST' in dataset_name:
        encoder = ae_model_MNIST.Encoder(latent_dim=latent_dim, num_filter=num_filter, conv_kernel_initializer_method=conv_kernel_initializer_method)
        decoder = ae_model_MNIST.Decoder(latent_dim=latent_dim, num_filter=num_filter, reg_strength=dec_reg_strength, conv_kernel_initializer_method=conv_kernel_initializer_method)
        dataloader_obj = dataloader_mnist(dataset_name, kde_samples, batch_size)
        x_train, x_val = dataloader_obj.split_train_n_val_data()

    elif dataset_name=='CelebA':
        encoder = ae_model_CelebA.Encoder(latent_dim=latent_dim, num_filter=num_filter, conv_kernel_initializer_method=conv_kernel_initializer_method)
        decoder = ae_model_CelebA.Decoder(latent_dim=latent_dim, num_filter=num_filter, reg_strength=dec_reg_strength, conv_kernel_initializer_method=conv_kernel_initializer_method)
        dataloader_obj = dataloader_celeba(dataset_name, kde_samples, batch_size)
        x_train, x_val = dataloader_obj.split_train_n_val_data()

    elif dataset_name=='CIFAR10':
        encoder = ae_model_CIFAR10.Encoder(latent_dim=latent_dim, num_filter=num_filter, conv_kernel_initializer_method=conv_kernel_initializer_method)
        decoder = ae_model_CIFAR10.Decoder(latent_dim=latent_dim, num_filter=num_filter, reg_strength=dec_reg_strength, conv_kernel_initializer_method=conv_kernel_initializer_method)
        dataloader_obj = dataloader_cifar10(dataset_name, kde_samples, batch_size)
        x_train, x_val = dataloader_obj.split_train_n_val_data()

    elif dataset_name=='DSprites':
        encoder = ae_model_DSprites.Encoder(latent_dim=latent_dim, num_filter=num_filter, conv_kernel_initializer_method=conv_kernel_initializer_method)
        decoder = ae_model_DSprites.Decoder(latent_dim=latent_dim, num_filter=num_filter, reg_strength=dec_reg_strength, conv_kernel_initializer_method=conv_kernel_initializer_method)
        dataloader_obj = dataloader_dsprites(dataset_name, kde_samples, batch_size)
        x_train, x_val = dataloader_obj.split_train_n_val_data()

    elif 'Shapes3D' in dataset_name:
        encoder = ae_model_Shapes3D.Encoder(latent_dim=latent_dim, num_filter=num_filter, conv_kernel_initializer_method=conv_kernel_initializer_method)
        decoder = ae_model_Shapes3D.Decoder(latent_dim=latent_dim, num_filter=num_filter, reg_strength=dec_reg_strength, conv_kernel_initializer_method=conv_kernel_initializer_method)
        dataloader_obj = dataloader_shapes3d(dataset_name, kde_samples, batch_size)
        x_train, x_val = dataloader_obj.split_train_n_val_data()

    # Optmizers for the encoder and decoder
    optimizer = Adam(learning_rate)
    reduce_rl_plateau = CustomReduceLRoP(patience=patience,
                                         factor=factor,
                                         verbose=1,
                                         optim_lr=optimizer.learning_rate)

    ###############
    # Model summary
    ###############
    if print_model_summary:
        encoder.build(input_shape=(100, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        encoder.summary()
        decoder.build(input_shape=(100, latent_dim))
        decoder.summary()
        print("Press a key to continue....")
        input()

    model_trainer_obj = trainer.Train(run_id, encoder, decoder, optimizer, reduce_rl_plateau, encoder_use_batch_norm, decoder_use_batch_norm, train_data_noise, train_from_checkpoint,
    epochs, update_KDE_epoch_fraction, update_q_iter_count, dataloader_obj, latent_dim, bandwidth, max_cdf_epsilon, print_every_epoch, save_every_epoch, save_model_epochs, dataset_name, sigma_init_val)

    ###########################
    # File Pointers for Logging
    ###########################
    file_name = os.path.join(model_trainer_obj.spec_model_dir, "Experimental_SetUp_Run#_" + str(run_id) + ".txt")
    exp_spec_file_ptr = open(file_name, "w")
    exp_spec_file_ptr.write("Experimental SetUp for Run#: " + str(run_id) + "\n")
    if train_from_checkpoint:
        exp_spec_file_ptr.write("Using pre-trained parameters of Run ID: " + str(model_trainer_obj.load_run_id) + "\n")
    exp_spec_file_ptr.write("Seed for initialization        : " + str(run_id - 1) + "\n")
    exp_spec_file_ptr.write("Target Epsilon                 : " + str(TARGET_FLOAT_EPS) + "\n")
    exp_spec_file_ptr.write("Encoded Epsilon                : " + str(model_trainer_obj.ENCODED_FLOAT_EPS) + "\n")
    exp_spec_file_ptr.write("Total Data Samples             : " + str(x_train.shape[0]) + "\n")
    exp_spec_file_ptr.write("Validation Samples             : " + str(x_val.shape[0]) + "\n")
    exp_spec_file_ptr.write("Lag Samples                    : " + str(kde_samples) + "\n")
    exp_spec_file_ptr.write("Epochs for KDE update          : " + str(update_KDE_epoch_fraction) + "\n")
    exp_spec_file_ptr.write("Batch Size                     : " + str(batch_size) + "\n")
    exp_spec_file_ptr.write("Noise Dimension                : " + str(latent_dim) + "\n")
    exp_spec_file_ptr.write("Learning Rate                  : " + str(optimizer.get_config()['learning_rate']) + "\n")
    exp_spec_file_ptr.write("Patience for learning rate     : " + str(patience) + "\n")
    exp_spec_file_ptr.write("Learning rate reducing factor  : " + str(factor) + "\n")
    exp_spec_file_ptr.write("Original bandwidth             : " + str(ori_bandwidth) + "\n")
    exp_spec_file_ptr.write("Bandwidth                      : " + str(bandwidth) + "\n")
    exp_spec_file_ptr.write("Update Q Iter Count            : " + str(update_q_iter_count) + "\n")
    exp_spec_file_ptr.write("Number of Epochs               : " + str(epochs) + "\n")
    exp_spec_file_ptr.write("Encoder Batch Normalization    : " + str(encoder_use_batch_norm) + "\n")
    exp_spec_file_ptr.write("Decoder Batch Normalization    : " + str(decoder_use_batch_norm) + "\n")
    exp_spec_file_ptr.write("Decoder Param Regularization   : " + str(dec_reg_strength) + "\n")
    exp_spec_file_ptr.write("Guassian_Prior_Std_Dev         : " + str(Guassian_Prior_Std_Dev) + "\n")
    exp_spec_file_ptr.write("Learn data noise               : " + str(train_data_noise) + "\n")
    exp_spec_file_ptr.write("Sigma2 init value              : " + str(model_trainer_obj.sigma_init_val) + "\n")
    exp_spec_file_ptr.write("Initial channel depth          : " + str(num_filter) + "\n")
    exp_spec_file_ptr.write("Conv kernel initialization     : " + conv_kernel_initializer_method)
    exp_spec_file_ptr.flush()
    exp_spec_file_ptr.close()

    # train the model
    model_trainer_obj.train_model()
    print("Trained the model for " + str(model_trainer_obj.cur_epoch) + " epochs....")

main()