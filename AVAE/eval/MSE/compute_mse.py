import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from data import get_data
from models import ae_model_CelebA, ae_model_CIFAR10, ae_model_MNIST
from config.local_config import configurations
import argparse
import nvidia_smi
import socket
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


@tf.function
def autoencoder_loss(x, x_hat):

    batch_size = tf.shape(x)[0]
    inputs = tf.reshape(x, (batch_size, -1))
    x_hat = tf.reshape(x_hat, (batch_size, -1))
    reconstruction = tf.reduce_mean(tf.reduce_sum((inputs - x_hat)**2, 1))

    return reconstruction


def show_combined_images(images, trans_images, row_cnt, col_cnt):
    fig = plt.figure()
    grid_spec = gridspec.GridSpec(ncols=col_cnt, nrows=row_cnt, figure=fig)
    grid_spec.update(wspace=0.05, hspace=0.05)

    for i in range(row_cnt):
        for j in range(0, col_cnt, 2):
            img_index = i * row_cnt + (j // 2)

            img = images[img_index, :, :, :]
            trans_img = trans_images[img_index, :, :, :]

            img = (img * 255.0).astype(np.uint8)
            trans_img = (trans_img * 255.0).astype(np.uint8)

            # Clipping the Range [0, 255]
            img = np.clip(img, 0, 255)
            trans_img = np.clip(trans_img, 0, 255)

            ax = fig.add_subplot(grid_spec[i, j])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(img, vmin=0, vmax=255)

            ax = fig.add_subplot(grid_spec[i, j + 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(trans_img, vmin=0, vmax=255)

    return fig


if __name__=="__main__":

    use_gpu, mem_free = select_GPU()
    print("Selected GPU for training : " + str(use_gpu) + " with available memory : " + str(mem_free//(1024*1024*1024)))

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = configurations[0][args.config_id]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    batch_size                  = config['batch_size']
    latent_dim                  = config['latent_dim']
    num_filter                  = config['num_filter']
    encoder_use_batch_norm      = config['encoder_use_batch_norm']
    decoder_use_batch_norm      = config['decoder_use_batch_norm']
    num_eval_data               = config['num_eval_data']
    basedir                     = os.path.join('..', '..', 'logs', dataset_name)
    eval_ids                    = np.arange(1, 6, 1).tolist()

    save_a_batch = False
    log_dir = os.path.join('logs', dataset_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    mse_error_array = np.zeros(len(eval_ids))
    mse_error_per_pixels_array = np.zeros(len(eval_ids))
    for run_id in eval_ids:
        # load checkpoint
        if 'MNIST' in dataset_name:
            encoder = ae_model_MNIST.Encoder(latent_dim=latent_dim, num_filter=num_filter)
            decoder = ae_model_MNIST.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        elif 'CelebA' in dataset_name:
            encoder = ae_model_CelebA.Encoder(latent_dim=latent_dim, num_filter=num_filter)
            decoder = ae_model_CelebA.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        elif 'CIFAR10' in dataset_name:
            encoder = ae_model_CIFAR10.Encoder(latent_dim=latent_dim, num_filter=num_filter)
            decoder = ae_model_CIFAR10.Decoder(latent_dim=latent_dim, num_filter=num_filter)

        learning_rate = 5e-4
        optimizer = Adam(learning_rate)
        checkpoint_dir = os.path.join(basedir, 'Run_' + str(run_id), 'Models')
        model_checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        status = model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_existing_objects_matched()
        print("Loaded saved model parameters!!")

        input_data = get_data.get_data_for_mse_error(dataset_name, num_eval_data)
        max_iter = num_eval_data//batch_size
        avg_recons_loss = 0
        for iter_index in range(max_iter):
            start_index = iter_index*batch_size
            end_index = iter_index*batch_size + batch_size
            input_batch = input_data[start_index:end_index]

            # reconstruction of data
            encoder_output = encoder(input_batch, use_batch_norm=encoder_use_batch_norm, training=False)
            decoder_output = decoder(encoder_output, use_batch_norm=decoder_use_batch_norm, training=False)

            # mse error
            avg_recons_loss += autoencoder_loss(input_batch, decoder_output).numpy()
        avg_recons_loss /= max_iter
        mse_error_array[run_id-1] = avg_recons_loss
        num_pixels = input_data.shape[1]*input_data.shape[2]*input_data.shape[3]
        mse_error_per_pixels_array[run_id-1] = avg_recons_loss/num_pixels
        print('Run : ' + str(run_id) + ' MSE            : ' + str(np.around(mse_error_array[run_id-1], 2)))
        print('Run : ' + str(run_id) + ' MSE per pixel  : ' + str(np.around(mse_error_per_pixels_array[run_id-1], 3)))

        # save a batch of reconstructed data
        if save_a_batch==False:
            row_cnt = col_cnt = int(np.sqrt(batch_size))
            fig = show_combined_images(input_batch, decoder_output.numpy(), row_cnt, col_cnt * 2)
            reconstructed_image_path = os.path.join(log_dir, 'recons_example.png')
            plt.savefig(reconstructed_image_path)
            plt.close(plt.gcf())
            save_a_batch = True

    log_filepath = os.path.join(log_dir, 'mse_error.txt')
    log_fileptr = open(log_filepath, 'w')
    log_fileptr.write(str(np.around(mse_error_array, 2))+'\n')
    log_fileptr.write(str(np.around(mse_error_per_pixels_array, 4))+'\n\n')
    log_fileptr.write('MSE stat             : ' + str(np.around(np.mean(mse_error_array), 2)) + ' \u00B1 '
                      + str(np.around(np.std(mse_error_array), 2)) + '\n')
    log_fileptr.write('MSE per pixel stat   : ' + str(np.around(np.mean(mse_error_per_pixels_array), 4)) + ' \u00B1 '
                      + str(np.around(np.std(mse_error_per_pixels_array), 4)))
    log_fileptr.flush()
    log_fileptr.close()
    # save the numpy array
    np_filepath = os.path.join(log_dir, 'mse_error.npy')
    np.save(np_filepath, mse_error_array)
    np_filepath = os.path.join(log_dir, 'mse_error_per_pixels_array.npy')
    np.save(np_filepath, mse_error_per_pixels_array)