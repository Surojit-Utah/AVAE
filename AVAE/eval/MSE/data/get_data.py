import tensorflow.keras as keras
import os
import numpy as np
import copy


def set_seed(seed=0):
    np.random.seed(seed)


def split_train_n_val_data(imgs_train, train_data_count, val_data_count):

    x_train = imgs_train[:train_data_count]
    x_val = imgs_train[train_data_count:train_data_count+val_data_count]

    print("Train Image Stats:")
    print("Number : ", x_train.shape[0])
    print("Min    : ", np.min(x_train))
    print("Max    : ", np.max(x_train))

    print("Validation Image Stats:")
    print("Number : ", x_val.shape[0])
    print("Min    : ", np.min(x_val))
    print("Max    : ", np.max(x_val))

    return x_train, x_val


def load_data(dataset_name):

    if 'CelebA' in dataset_name:
        data_dir = '/home/sci/surojit/Research/Invertibile_GAN/Celeb_A'
        data_path = os.path.join(data_dir, 'train_images_npy.npy')
        imgs_train = copy.deepcopy(np.load(data_path))
        imgs_train = (imgs_train / 255.0).astype(np.float32)
        imgs_train = 2*imgs_train - 1

    if 'CIFAR10' in dataset_name:
        (imgs_train, _), (_, _) = keras.datasets.cifar10.load_data()
        imgs_train = (imgs_train / 255.0).astype(np.float32)

    if 'MNIST' in dataset_name:
        (ori_imgs_train, _), (_, _) = keras.datasets.mnist.load_data()
        ori_imgs_train = np.expand_dims(ori_imgs_train, axis=-1)
        imgs_train = np.zeros((ori_imgs_train.shape[0], 32, 32, 1)).astype(np.float32)
        imgs_train[:, 2:30, 2:30, :] = ori_imgs_train.astype(np.float32)
        imgs_train = (imgs_train / 255.0).astype(np.float32)

    return imgs_train


def get_data_for_mse_error(dataset_name, fid_samples):

    imgs_train = load_data(dataset_name)
    if 'CelebA' in dataset_name:
        train_data_count = 162770
        val_data_count = 19867
    elif 'CIFAR10' in dataset_name:
        train_data_count = 40000
        val_data_count = 10000
    elif 'MNIST' in dataset_name:
        train_data_count = 40000
        val_data_count = 10000

    # Split the train and validation dataset
    x_train, x_val = split_train_n_val_data(imgs_train, train_data_count, val_data_count)
    x_val = x_val[:fid_samples]

    return x_val