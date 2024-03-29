import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Utility functions
def show_images(images, row_cnt, col_cnt):

    images = images[:(row_cnt*col_cnt)]
    fig = plt.figure(figsize=(40, 40))
    gs = gridspec.GridSpec(row_cnt, col_cnt)
    gs.update(wspace=0.1, hspace=0.1)

    for i, img in enumerate(images):

        img = (img * 255.0).astype(np.uint8)

        # Clipping the Range [0, 255]
        img = np.clip(img, 0, 255)

        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img, vmin=0, vmax=255)

    return


def show_combined_images(images, trans_images, row_cnt, col_cnt):

    fig = plt.figure()
    grid_spec = gridspec.GridSpec(ncols=col_cnt, nrows=row_cnt, figure=fig)
    grid_spec.update(wspace=0.05, hspace=0.05)

    for i in range(row_cnt):
        for j in range(0, col_cnt, 2):

            img_index = i*row_cnt + (j // 2)

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
            plt.imshow(img, vmin=0, vmax= 255)

            ax = fig.add_subplot(grid_spec[i, j + 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(trans_img, vmin=0, vmax= 255)

    return fig


def show_interpolated_images(Interpolated_Images, row_cnt, col_cnt):

    fig = plt.figure(figsize=(4*col_cnt, 4*row_cnt))
    grid_spec = gridspec.GridSpec(ncols=col_cnt, nrows=row_cnt, figure=fig)
    grid_spec.update(wspace=0.05, hspace=0.05)

    for i in range(row_cnt):
        dim_images = Interpolated_Images[(i*col_cnt):((i+1)*col_cnt)]
        for j in range(col_cnt):
            img = dim_images[j]
            img = (img * 255.0).astype(np.uint8)
            # Clipping the Range [0, 255]
            img = np.clip(img, 0, 255)

            ax = fig.add_subplot(grid_spec[i, j])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(img, vmin=0, vmax=255)

    return fig


def splom(array, latent_image_path):

    # dimension = array.shape[1]
    dimension = min(30, array.shape[1])
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig, axs = plt.subplots(dimension, dimension, figsize=(30, 30))
    for i in range(dimension):
        for j in range(dimension):
            axs[i, j].scatter(array[:, i], array[:, j], s=0.2)
            axs[i, j].set_xlim(left=-3, right=3)
            axs[i, j].set_ylim(bottom=-3, top=3)

    plt.savefig(latent_image_path)
    plt.close(plt.gcf())


def diag_axis_splom(array, latent_image_path, max_sigma):

    dimension = array.shape[1]
    cols = min(dimension, 20)
    if (dimension % cols) > 0:
        rows = (dimension // cols) + 1
    else:
        rows = (dimension // cols)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig, axs = plt.subplots(rows, cols, squeeze=False)
    fig.set_size_inches(cols, rows)

    row_index = 0
    col_index = 0
    for i in range(dimension):
        # scatter plot
        axs[row_index, col_index].scatter(array[:, i], array[:, i], s=0.2)
        axs[row_index, col_index].set_xlim(left=-max_sigma, right=max_sigma)
        axs[row_index, col_index].set_ylim(bottom=-max_sigma, top=max_sigma)
        col_index += 1
        # adjust the row and col index
        if col_index == cols:
            col_index = 0
            row_index += 1

    plt.savefig(latent_image_path)
    plt.close(plt.gcf())


def create_log_directory(dataset_name, latent_dim, run_id):

    # Experimental specifications
    spec_model_dir = os.path.join("logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Exp_Spec")
    if not os.path.isdir(spec_model_dir):
        os.makedirs(spec_model_dir)
    else:
        print("Directory Already Exists!!")
        print("Rename the EXISTING Directory!!")
        input()
        os.makedirs(spec_model_dir)

    # Generated Images
    generated_image_dir = os.path.join("logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Output", "Generated_Images")
    if not os.path.isdir(generated_image_dir):
        os.makedirs(generated_image_dir)
    else:
        print("Directory Already Exists!!")
        print("Rename the EXISTING Directory!!")
        input()
        os.makedirs(generated_image_dir)

    # Reconstructed Images
    reconstructed_image_dir = os.path.join("logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Output", "Reconstructed_Images")
    if not os.path.isdir(reconstructed_image_dir):
        os.makedirs(reconstructed_image_dir)
    else:
        print("Directory Already Exists!!")
        print("Rename the EXISTING Directory!!")
        input()
        os.makedirs(reconstructed_image_dir)

    # Projection in the Latent Space
    latent_repr_image_dir = os.path.join("logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Output", "Projection")
    if not os.path.isdir(latent_repr_image_dir):
        os.makedirs(latent_repr_image_dir)
    else:
        print("Directory Already Exists!!")
        print("Rename the EXISTING Directory!!")
        input()
        os.makedirs(latent_repr_image_dir)

    # Model Saving Directory
    save_model_dir = os.path.join("logs", dataset_name, "Dim_"+str(latent_dim), "Run_"+str(run_id), "Models")
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)
    else:
        print("Directory Already Exists!!")
        print("Rename the EXISTING Directory!!")
        input()
        os.makedirs(save_model_dir)

    return spec_model_dir, generated_image_dir, reconstructed_image_dir, latent_repr_image_dir, save_model_dir