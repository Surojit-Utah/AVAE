import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import nvidia_smi
import os
import numpy as np
from gen_samples import get_data
from models import ae_model_CelebA, ae_model_CIFAR10, ae_model_MNIST


def select_GPU(min_gpu_mem_frac=0.5):
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    for device_index in range(device_count):
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


def get_encoded_data(dataset_name, run_id, basedir, latent_dim, use_encoder_batch_norm, num_filter, fid_samples, batch_size):
    use_gpu, mem_free = select_GPU()
    print("Selected GPU for training : " + str(use_gpu) + " with available memory : " + str(mem_free//(1024*1024*1024)))

    ###################
    # Autoencoder model
    ###################
    if 'MNIST' in dataset_name:
        encoder = ae_model_MNIST.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_MNIST.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    elif dataset_name=='CelebA':
        encoder = ae_model_CelebA.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CelebA.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    elif 'CIFAR10' in dataset_name:
        encoder = ae_model_CIFAR10.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CIFAR10.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    learning_rate = 5e-4
    optimizer = Adam(learning_rate)

    ####################
    # Checkpoint details
    ####################
    model_checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint_dir = os.path.join(basedir, 'Run_' + str(run_id), 'Models')
    status = model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.assert_existing_objects_matched()
    print("Loaded saved model parameters!!")

    latent_vectors = np.zeros((fid_samples, latent_dim))
    val_data = get_data.get_data_for_encoded_statistics(dataset_name, fid_samples)
    max_iter = fid_samples//batch_size
    print("Number of Iterations are : " + str(max_iter))
    for iter in range(max_iter):
        img_start = iter*batch_size
        img_end = img_start+batch_size
        minibatch = val_data[img_start:img_end]
        latent_vectors[img_start:img_end] = encoder(minibatch, use_batch_norm=use_encoder_batch_norm, training=False)

    return latent_vectors


def get_statistics_encoded_data(dataset_name, run_id, basedir, latent_dim, use_encoder_batch_norm, num_filter, fid_samples, batch_size, latent_vectors=None):

    if latent_vectors==None:
        latent_vectors = get_encoded_data(dataset_name, run_id, basedir, latent_dim, use_encoder_batch_norm, num_filter, fid_samples, batch_size)

    Mean_vector = np.mean(latent_vectors, axis=0)
    Covariance_Matrix = np.cov(latent_vectors.T)

    Round_Covariance_Matrix = np.round(Covariance_Matrix, decimals=3)
    diagonal_values = list()
    abs_off_diagonal_values = list()
    for (j,i),label in np.ndenumerate(Round_Covariance_Matrix):
        if i==j:
            diagonal_values.append(label)
        else:
            abs_off_diagonal_values.append(np.absolute(label))

    diagonal_values_array = np.asarray(diagonal_values)
    abs_off_diagonal_values_array = np.asarray(abs_off_diagonal_values)

    latent_stat = (Mean_vector, Round_Covariance_Matrix, diagonal_values_array, abs_off_diagonal_values_array)
    return latent_vectors, latent_stat