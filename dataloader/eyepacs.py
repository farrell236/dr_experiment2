import cv2
import datetime
import os

import numpy as np
import pandas as pd

from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, csv, data_root, batch_size=6, resize=256, n_channels=3, transform=None, shuffle=True):

        csv_file = pd.read_csv(csv)

        'Initialization'
        self.list_IDs = csv_file['name'].values
        self.list_labels = csv_file['level'].values
        self.data_root = data_root
        self.batch_size = batch_size
        self.resize = resize
        self.n_channels = n_channels
        self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.resize, self.resize, self.n_channels))
        y = np.empty((self.batch_size, 1), dtype=np.uint8)

        # Generate data
        for i, idx in enumerate(indexes):

            # Update random seed
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
            seed = np.random.get_state()[1][0]

            # Load sample
            image = cv2.imread(os.path.join(self.data_root, self.list_IDs[idx])+'.jpeg')

            # Resize Images
            image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)

            # Data augmentation and transformation
            if self.transform:
                image = self.transform['image'].random_transform(image, seed=seed)

            # Store sample
            X[i, ...] = image
            y[i, 0] = self.list_labels[idx]

        return X, y



if __name__ == '__main__':

    train_csv = 'data/EyePACS/train_all_df.csv'
    train_root = 'preprocessing/train_C0_R1024x1024'
    test_csv = 'data/EyePACS/test_public_df.csv'
    test_root = 'preprocessing/test_C0_R1024x1024'

    res = 512

    train_generator = DataGenerator(train_csv, train_root, batch_size=8, resize=res, transform=None)
    valid_generator = DataGenerator(test_csv, test_root, batch_size=4, resize=res, transform=None, shuffle=False)

    data_transforms = {
        'image': keras.preprocessing.image.ImageDataGenerator(
            rotation_range=45,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant', cval=0.0,
            brightness_range=[0.5, 1.5]
        ),
        'label': keras.preprocessing.image.ImageDataGenerator(
            rotation_range=45,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant', cval=0.0,
        ),
    }

    dataloader = DataGenerator(train_csv, train_root, resize=1024, transform=None)

    batch = dataloader.__getitem__(1)

    for i in range(6):
        cv2.imwrite('imgdump/image{}.png'.format(i), batch[0][i])

    for i in range(6):
        print(i, batch[1][i])

    a=1

