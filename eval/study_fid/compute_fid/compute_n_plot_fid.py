import os
import numpy as np
import argparse
from config.local_config import configurations
import get_fid
import nvidia_smi
import tensorflow as tf


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


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--gen_type", type=str, required=True, help="Can be one of the following options: generation, reconstruction")
    args = parser.parse_args()

    latent_dim = args.latent_dim
    mode = args.gen_type
    config = configurations[0][args.config_id]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    fid_samples                 = config['fid_samples']
    eval_ids                    = np.arange(1, 2, 1).tolist()
    fidstat_basedir             = 'fid_stats'
    if dataset_name=='MNIST':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_mnist.npz')
        gen_samples_np = False
    if dataset_name=='CelebA':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_celeba.npz')
        gen_samples_np = True
    if dataset_name=='CIFAR10':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_cifar10_train.npz')
        gen_samples_np = True
    if 'DSprites' in dataset_name:
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_dsprites.npz')
        gen_samples_np = False

    use_gpu, mem_free = select_GPU()
    print("Selected GPU for training : " + str(use_gpu) + " with available memory : " + str(mem_free//(1024*1024*1024)))

    fid_scores_stat = np.zeros(len(eval_ids))
    log_dir = os.path.join('logs', dataset_name, mode, 'Dim_'+str(latent_dim))
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'fid_stat.txt')
    if os.path.isfile(log_filepath):
        log_fileptr = open(log_filepath, 'a')
    else:
        log_fileptr = open(log_filepath, 'w')


    run_index = 0
    for run_id in eval_ids:

        generated_image_array_dir = os.path.join('..', 'generate_samples', 'logs', dataset_name, mode, 'Dim_'+str(latent_dim), 'run_id_' + str(run_id), 'images')

        if gen_samples_np:
            generated_image_array_path = os.path.join(generated_image_array_dir, "generated_images.npy")
            generated_image = np.load(generated_image_array_path).astype(np.float32)
            assert generated_image.shape[0]==fid_samples, "Number of generated samples are not sufficient...."
        else:
            generated_image = generated_image_array_dir

        # compute the FID score for each annulus
        model_fid_score = get_fid.calculate_fid_given_paths(generated_image, fid_stat_path, None, gen_samples_np=gen_samples_np)

        # save the fid score
        fid_scores_stat[run_index] = model_fid_score
        run_index += 1
        log_fileptr.write('FID score for run ID ' + str(run_id) + ' is ' + str(np.around(model_fid_score, 2)) + '\n')
        log_fileptr.flush()

    # fid score statistics
    avg_fid_score = np.mean(fid_scores_stat)
    stddev_fid_score = np.std(fid_scores_stat)

    # save the fid score statistics
    log_fileptr.write('Average of the FID stat over ' + str(len(eval_ids)) + ' models....' + '\n')
    log_fileptr.write(str(np.around(avg_fid_score, 2)) + ' \u00B1 ' + str(np.around(stddev_fid_score, 2)))
    log_fileptr.flush()
    log_fileptr.close()

    np_save_path = os.path.join(log_dir, 'fid_scores_stat.npy')
    np.save(np_save_path, fid_scores_stat)