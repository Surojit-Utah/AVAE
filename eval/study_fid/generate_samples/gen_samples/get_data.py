import tensorflow.keras as keras
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
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


def sample_kde_n_train_data(x_train, kde_samples):

    # Sample train and T-Stat indices from the training data
    set_seed()
    TStat_samples_indices = np.random.choice(x_train.shape[0], kde_samples, replace=False).tolist()
    Train_samples_indices = list(set(np.arange(x_train.shape[0]).tolist()).difference(set(TStat_samples_indices)))

    # Test the overlap between the training data and the T-Stat samples
    train_lag_indices_overlap = set(Train_samples_indices).intersection(set(TStat_samples_indices))
    assert len(train_lag_indices_overlap) is 0, "There is overlap between the training and T-Stat samples"

    # Data to be used in training GENs
    x_train_sgd = x_train[Train_samples_indices]
    x_train_kde = x_train[TStat_samples_indices]

    print("Input Image Stats:")
    print("Number : ", x_train_sgd.shape[0])
    print("Min    : ", np.min(x_train_sgd))
    print("Max    : ", np.max(x_train_sgd))

    print("T-Stat Image Stats:")
    print("Number : ", x_train_kde.shape[0])
    print("Min    : ", np.min(x_train_kde))
    print("Max    : ", np.max(x_train_kde))

    return x_train_sgd, x_train_kde


def load_data(dataset_name):

    if dataset_name == 'CelebA':
        data_dir = ''
        data_path = os.path.join(data_dir, 'train_images_npy.npy')
        imgs_train = copy.deepcopy(np.load(data_path))
        imgs_train = (imgs_train / 255.0).astype(np.float32)
        imgs_train = 2*imgs_train - 1

    if dataset_name == 'CIFAR10':
        (imgs_train, _), (_, _) = keras.datasets.cifar10.load_data()
        imgs_train = (imgs_train / 255.0).astype(np.float32)

    if dataset_name == 'MNIST':
        (ori_imgs_train, _), (_, _) = keras.datasets.mnist.load_data()
        ori_imgs_train = np.expand_dims(ori_imgs_train, axis=-1)
        imgs_train = np.zeros((ori_imgs_train.shape[0], 32, 32, 1)).astype(np.float32)
        imgs_train[:, 2:30, 2:30, :] = ori_imgs_train.astype(np.float32)
        imgs_train = (imgs_train / 255.0).astype(np.float32)

    return imgs_train


def get_data_for_encoded_statistics(dataset_name, fid_samples, mode):

    imgs_train = load_data(dataset_name)
    if dataset_name == 'CelebA':
        train_data_count = 162770
        val_data_count = 19867
    elif dataset_name == 'CIFAR10':
        train_data_count = 40000
        val_data_count = 10000
    elif dataset_name == 'MNIST':
        train_data_count = 40000
        val_data_count = 10000

    # Split the train and validation dataset
    x_train, x_val = split_train_n_val_data(imgs_train, train_data_count, val_data_count)
    if mode=='generation':
        data = x_train[:fid_samples]
    else:
        data = x_val[:fid_samples]

    return data