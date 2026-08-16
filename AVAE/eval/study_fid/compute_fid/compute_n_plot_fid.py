import os
import numpy as np
import argparse
from config.local_config import configurations
import get_fid


def set_seed(seed=0):
    np.random.seed(seed)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--gen_type", type=str, required=True, help="Can be one of the following options: generation, interpolation, reconstruction")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    mode = args.gen_type
    config = configurations[0][args.config_id]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    fid_samples                 = config['fid_samples']
    eval_ids                    = np.arange(1, 6, 1).tolist()
    
    # Use relative path to fid_stats_data in eval/study_fid/
    fidstat_basedir = os.path.join(os.path.dirname(__file__), '..', 'fid_stats_data')
    if not os.path.exists(fidstat_basedir):
        raise ValueError(
            f"FID statistics directory not found at: {os.path.abspath(fidstat_basedir)}\n"
            "Expected location: eval/study_fid/fid_stats_data/\n"
            "Please ensure fid_stats_mnist.npz, fid_stats_celeba.npz, and fid_stats_cifar10_train.npz are present."
        )
    
    if 'MNIST' in dataset_name:
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_mnist.npz')
        gen_samples_np = False
    if dataset_name=='CelebA':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_celeba.npz')
        gen_samples_np = True
    if 'CIFAR10' in dataset_name:
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_cifar10_train.npz')
        gen_samples_np = True
    if 'DSprites' in dataset_name:
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_dsprites.npz')
        gen_samples_np = False

    fid_scores = np.zeros(len(eval_ids))
    fid_score_index = 0
    for run_id in eval_ids:

        # save few generated samples from each annulus
        generated_image_array_dir = os.path.join('..', 'generate_samples', 'logs', dataset_name, mode, 'run_id_' + str(run_id))
        if gen_samples_np:
            generated_image_array_path = os.path.join(generated_image_array_dir, "generated_images.npy")
            generated_image = np.load(generated_image_array_path).astype(np.float32)
            assert generated_image.shape[0]==fid_samples, "Number of generated samples are not sufficient...."
        else:
            generated_image = generated_image_array_dir

        # compute the FID score for each annulus
        model_fid_score = get_fid.calculate_fid_given_paths(generated_image, fid_stat_path, None, gen_samples_np=gen_samples_np)
        fid_scores[fid_score_index] = model_fid_score
        fid_score_index += 1
        print(fid_scores)
    print(fid_scores)
    # fid score statistics
    avg_fid_score = np.mean(fid_scores)
    stddev_fid_score = np.std(fid_scores)

    # save the fid score statistics
    log_dir = os.path.join('logs', dataset_name, mode)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'fid_stat.txt')
    log_fileptr = open(log_filepath, 'w')
    log_fileptr.write(str(fid_scores) + '\n')
    log_fileptr.write('Average of the FID stat over ' + str(len(eval_ids)) + ' models....' + '\n')
    log_fileptr.write(str(np.around(avg_fid_score, 2)) + ' \u00B1 ' + str(np.around(stddev_fid_score, 2)))
    log_fileptr.flush()
    log_fileptr.close()