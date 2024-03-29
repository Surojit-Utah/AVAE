import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import tensorflow.keras as keras
import copy
import os


'''
MNIST training on 2D latent space
'''
# select a subset of digits from the training data
# sel_digits = [1, 4, 7, 9]
# sel_indices = set()
# for sel_digit in sel_digits:
#     sel_indices = sel_indices.union(set(np.where(ori_label_train==sel_digit)[0].tolist()))
# sel_indices = list(sel_indices)
# ori_imgs_train = ori_imgs_train[sel_indices]
# self.val_data_count = (2000//batch_size)*batch_size

class dataloader_mnist():
    def __init__(self, dataset_name, kde_samples, batch_size=100):

        if 'MNIST' in dataset_name:
            (ori_imgs_train, _), (_, _) = keras.datasets.mnist.load_data()
            ori_imgs_train = np.expand_dims(ori_imgs_train, axis=-1)
            self.imgs_train = np.zeros((ori_imgs_train.shape[0], 32, 32, 1)).astype(np.float32)
            self.imgs_train[:, 2:30, 2:30, :] = ori_imgs_train.astype(np.float32)
            self.imgs_train = (self.imgs_train/255.0).astype(np.float32)
            self.val_data_count = (10000//batch_size)*batch_size
            self.train_data_count = ((self.imgs_train.shape[0] - self.val_data_count)//batch_size)*batch_size

        self.kde_samples = kde_samples
        self.batch_size = batch_size
        self.train_dataset = None
        self.kde_dataset = None
        self.val_dataset = None


    def create_dataset(self, data, batch_size):
        return tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)


    def split_train_n_val_data(self):

        self.x_train = self.imgs_train[:self.train_data_count]
        self.x_val = self.imgs_train[self.train_data_count:self.train_data_count+self.val_data_count]

        print("Train Image Stats:")
        print("Number : ", self.x_train.shape[0])
        print("Min    : ", np.min(self.x_train))
        print("Max    : ", np.max(self.x_train))

        print("Validation Image Stats:")
        print("Number : ", self.x_val.shape[0])
        print("Min    : ", np.min(self.x_val))
        print("Max    : ", np.max(self.x_val))

        return self.x_train, self.x_val


    def sample_kde_n_train_data(self):

        # Sample train and KDE indices from the training data
        KDE_samples_indices = np.random.choice(self.x_train.shape[0], self.kde_samples, replace=False).tolist()
        Train_samples_indices = list(set(np.arange(self.x_train.shape[0]).tolist()).difference(set(KDE_samples_indices)))

        # Test the overlap between the training data and the KDE samples
        train_lag_indices_overlap = set(Train_samples_indices).intersection(set(KDE_samples_indices))
        assert len(train_lag_indices_overlap) is 0, "There is overlap between the training and KDE samples"

        # Data to be used in training GENs
        x_train_sgd = self.x_train[Train_samples_indices]
        x_train_kde = self.x_train[KDE_samples_indices]

        print("Input Image Stats:")
        print("Number : ", x_train_sgd.shape[0])
        print("Min    : ", np.min(x_train_sgd))
        print("Max    : ", np.max(x_train_sgd))

        print("KDE Image Stats:")
        print("Number : ", x_train_kde.shape[0])
        print("Min    : ", np.min(x_train_kde))
        print("Max    : ", np.max(x_train_kde))

        return x_train_sgd, x_train_kde


    def create_kde_n_train_dataset(self):

        x_train_sgd, x_train_kde = self.sample_kde_n_train_data()
        self.train_dataset = self.create_dataset(x_train_sgd, self.batch_size)
        self.kde_dataset = self.create_dataset(x_train_kde, self.batch_size)
        return self.train_dataset, self.kde_dataset

    def create_val_dataset(self):

        val_dataset = self.create_dataset(self.x_val, self.batch_size)
        return val_dataset