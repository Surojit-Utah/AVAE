import numpy as np


class DataGenerator():
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.indices = np.arange(self.data.shape[0])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        # data preprocessing
        image = self.data[id]
        return image

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __call__(self):
        for i in self.indices:
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()