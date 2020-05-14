import cv2
import datetime
import os

import numpy as np
import pandas as pd

from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, csv, data_root, batch_size=6, resize=None, rnd_crop=None, n_channels=3, transform=None, shuffle=True, preload=True):

        csv_file = pd.read_csv(csv)

        'Initialization'
        self.list_IDs = csv_file['image'].values
        self.list_labels = csv_file['label'].values
        self.data_root = data_root
        self.batch_size = batch_size
        self.resize = resize
        self.rnd_crop = rnd_crop
        self.n_channels = n_channels
        self.transform = transform
        self.shuffle = shuffle
        self.preload = preload
        self.dim = rnd_crop or resize or 1024

        if preload:
            self.images = []
            self.labels = []
            for i in range(len(self.list_IDs)):
                self.images.append(cv2.imread(os.path.join(self.data_root, self.list_IDs[i])))
                self.labels.append(cv2.imread(os.path.join(self.data_root, self.list_labels[i])))

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self._data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def randomCrop(self, img, mask, width, height):
        'Apply random crop on the image and mask pair'
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = np.random.randint(0, img.shape[1] - width)
        y = np.random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width]
        mask = mask[y:y + height, x:x + width]
        return img, mask

    def _data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim, self.dim, 1), dtype=np.uint8)

        # Generate data
        for i, idx in enumerate(indexes):

            # Update random seed
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
            seed = np.random.get_state()[1][0]

            # Load sample
            if self.preload:
                image = np.copy(self.images[idx])
                label = np.copy(self.labels[idx])
            else:
                image = cv2.imread(os.path.join(self.data_root, self.list_IDs[idx]))
                label = cv2.imread(os.path.join(self.data_root, self.list_labels[idx]))

            # Resize Images
            if self.resize != None:
                image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)

            # Random Crop Images
            if self.rnd_crop != None:
                image, label = self.randomCrop(image, label, self.rnd_crop, self.rnd_crop)

            # Data augmentation and transformation
            if self.transform:
                image = self.transform['image'].random_transform(image, seed=seed)
                label = self.transform['label'].random_transform(label, seed=seed)

            # Store sample
            X[i, ...] = image
            y[i, :, :, 0] = label[..., 0]

        return X, y



if __name__ == '__main__':

    train_csv = 'data/IDRID/train_list.csv'
    data_root = 'data/IDRID/IDRID_train'

    # train_csv = 'data/DRIVE/train_list.csv'
    # data_root = 'data/DRIVE/DRIVE_train'

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

    dataloader = DataGenerator(train_csv, data_root, resize=1024, transform=None)

    batch = dataloader.__getitem__(1)

    for i in range(6):
        cv2.imwrite('imgdump/image{}.png'.format(i), batch[0][i])
        cv2.imwrite('imgdump/label{}.png'.format(i), batch[1][i]*255)

    for i in range(6):
        print(i, batch[0][i,0,0,0])

    a=1

