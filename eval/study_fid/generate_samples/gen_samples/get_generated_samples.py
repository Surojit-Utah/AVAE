import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import nvidia_smi
import os
import numpy as np
from models import ae_model_CelebA, ae_model_CIFAR10, ae_model_MNIST
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nvidia_smi


# Utility functions
def show_images(images, batch_size):
    row_cnt, col_cnt = int(np.sqrt(batch_size)), int(np.sqrt(batch_size))
    images = images[:(row_cnt*col_cnt)]
    fig = plt.figure(figsize=(40, 40))
    gs = gridspec.GridSpec(row_cnt, col_cnt)
    gs.update(wspace=0.1, hspace=0.1)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img, vmin=0, vmax=255)

    return fig

def select_GPU(min_gpu_mem_frac=0.9):
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


def get_generated_data(dataset_name, run_id, basedir, latent_dim, use_decoder_batch_norm, num_filter, latent_vectors, batch_size):

    ###################
    # Autoencoder model
    ###################
    if dataset_name=='MNIST':
        encoder = ae_model_MNIST.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_MNIST.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        gen_image_shape = [32, 32, 3]
    elif dataset_name=='CelebA':
        encoder = ae_model_CelebA.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CelebA.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        gen_image_shape = [64, 64, 3]
    elif dataset_name=='CIFAR10':
        encoder = ae_model_CIFAR10.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CIFAR10.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        gen_image_shape = [32, 32, 3]
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

    fid_samples = latent_vectors.shape[0]
    gen_images_array = np.zeros((fid_samples, gen_image_shape[0], gen_image_shape[1], gen_image_shape[2]), dtype=np.uint8)
    max_iter = fid_samples//batch_size
    print("Number of Iterations are : " + str(max_iter))
    for iter in range(max_iter):
        start_index = iter * batch_size
        end_index = start_index + batch_size
        sampled_z = latent_vectors[start_index:end_index]
        gen_images = decoder(sampled_z, use_batch_norm=use_decoder_batch_norm, training=False)
        if dataset_name == 'MNIST':
            gen_images = tf.image.grayscale_to_rgb(gen_images, name=None)
        if 'DSprites' in dataset_name:
            gen_images = tf.math.sigmoid(gen_images)
            gen_images = tf.image.grayscale_to_rgb(gen_images, name=None)
        if dataset_name == 'CelebA':
            gen_images = (gen_images + 1) / 2
        gen_images = gen_images.numpy()
        gen_images = (gen_images*255).astype(np.uint8)
        gen_images_array[start_index:end_index] = gen_images

    return gen_images_array