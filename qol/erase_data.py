import argparse
import os
import shutil
import sys
sys.path.append('..')
from config.local_config import configurations


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--config_id", type=int, required=True)
    args = parser.parse_args()
    run_id = args.run_id
    config = configurations[0][args.config_id]
    dataset_name = config['dataset_name']
    latent_dim = config['latent_dim']

    # Experimental specifications
    spec_model_dir = os.path.join("..", "logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Exp_Spec")
    shutil.rmtree(spec_model_dir)
    print("Removed the exp spec file....")

    # Generated Images
    generated_image_dir = os.path.join("..", "logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Output", "Generated_Images")
    shutil.rmtree(generated_image_dir)

    # Reconstructed Images
    reconstructed_image_dir = os.path.join("..", "logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Output", "Reconstructed_Images")
    shutil.rmtree(reconstructed_image_dir)

    # Projection in the Latent Space
    latent_repr_image_dir = os.path.join("..", "logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Output", "Projection")
    shutil.rmtree(latent_repr_image_dir)
    print("Removed the output directories....")

    # Model Saving Directory
    save_model_dir = os.path.join("..", "logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Models")
    shutil.rmtree(save_model_dir)
    print("Removed the saved model directory....")

    # tensorboard log
    tb_dir = os.path.join("..", "logs", dataset_name, "Dim_"+str(latent_dim), "Run_" + str(run_id), "tf_logs")
    shutil.rmtree(tb_dir)
    print("Removed the tensorboard data....")