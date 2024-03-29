import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import tensorflow.keras as keras
import sys
sys.path.append("..")
from data.datagenerator import DataGenerator
import copy
import os
import gc

class dataloader_dsprites():
    def __init__(self, dataset_name, kde_samples, batch_size=100):

        if dataset_name=='DSprites':
            self.data_dir = '/home/sci/surojit/Research/Data/DSprites'
            self.data_path = os.path.join(self.data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
            self.imgs_train = copy.deepcopy(np.load(self.data_path)['imgs'])
            self.imgs_train = np.expand_dims(self.imgs_train, axis=-1).astype(np.float32)
            self.val_data_count = (20000//batch_size)*batch_size
            self.train_data_count = ((self.imgs_train.shape[0] - self.val_data_count)//batch_size)*batch_size

        self.kde_samples = kde_samples
        self.batch_size = batch_size
        self.train_dataset = None
        self.kde_dataset = None
        self.val_dataset = None


    def create_dataset(self, data_generator, output_types=tf.float32, output_shapes=tf.TensorShape([64, 64, 1]), batch_size=100):
        return tf.data.Dataset.from_generator(data_generator, output_types=output_types, output_shapes=output_shapes).batch(batch_size)


    def split_train_n_val_data(self):

        val_indices = np.random.choice(self.imgs_train.shape[0], self.val_data_count, replace=False).tolist()
        train_indices = set(np.arange(self.imgs_train.shape[0]).tolist()).difference(set(val_indices))
        train_indices = list(train_indices)

        # Test the overlap between the training data and the KDE samples
        train_lag_indices_overlap = set(train_indices).intersection(set(val_indices))
        assert len(train_lag_indices_overlap) is 0, "There is overlap between the training and KDE samples"

        self.x_train = self.imgs_train[train_indices]
        self.x_val = self.imgs_train[val_indices]

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

        if self.train_dataset is not None or self.kde_dataset is not None:
            del self.train_dataset
            del self.train_sgd_generator

            del self.kde_dataset
            del self.train_kde_generator

            """
            https://github.com/tensorflow/tensorflow/issues/37505
            https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-issue-in-keras-model-training-e703907a6501
            """
            keras.backend.clear_session()
            gc.collect()
            print("Garbage collection is done....")

        x_train_sgd, x_train_kde = self.sample_kde_n_train_data()
        self.train_sgd_generator = DataGenerator(x_train_sgd, shuffle=True)
        self.train_kde_generator = DataGenerator(x_train_kde, shuffle=False)
        self.train_dataset = self.create_dataset(self.train_sgd_generator, output_shapes=tf.TensorShape(self.imgs_train.shape[1:]), batch_size=self.batch_size)
        self.kde_dataset = self.create_dataset(self.train_kde_generator, output_shapes=tf.TensorShape(self.imgs_train.shape[1:]), batch_size=self.batch_size)

        return self.train_dataset, self.kde_dataset

    def create_val_dataset(self):

        validation_generator = DataGenerator(self.x_val, shuffle=False)
        val_dataset = self.create_dataset(validation_generator, output_shapes=tf.TensorShape(self.imgs_train.shape[1:]), batch_size=self.batch_size)

        return val_dataset