import os
import cv2
import tqdm

import numpy as np
import tensorflow as tf

from tensorflow import keras

from dataloader.idrid_drive import DataGenerator
from networks.UNet2D_v2 import unet
from networks.UNet2D import R2AttU_Net
from networks.BCDU_Net import BCDU_net_D3

# train_csv = 'data/IDRID/train_list.csv'
# train_root = 'data/IDRID/IDRID_train'
# test_csv = 'data/IDRID/test_list.csv'
# test_root = 'data/IDRID/IDRID_test'

train_csv = 'data/DRIVE/train_list.csv'
train_root = 'data/DRIVE/DRIVE_train'
test_csv = 'data/DRIVE/test_list.csv'
test_root = 'data/DRIVE/DRIVE_test'


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

train_generator = DataGenerator(train_csv, train_root, batch_size=5, rnd_crop=256, transform=data_transforms)
valid_generator = DataGenerator(test_csv, test_root, batch_size=1, transform=None, shuffle=False)


# Standard unet
inputs = keras.layers.Input(shape=(None, None, 3))
outputs = unet(inputs=inputs, output_ch=1, activation='sigmoid')
unet_model = keras.models.Model(inputs=inputs, outputs=outputs)
unet_model.summary()

# AttnUnet
# unet_model = R2AttU_Net(output_ch=1, t=3)

# BCDU-Net
# inputs = keras.layers.Input(shape=(256, 256, 3))
# outputs = BCDU_net_D3(inputs=inputs, output_ch=1, N=256)
# unet_model = keras.models.Model(inputs=inputs, outputs=outputs)
# unet_model.summary()


unet_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                   loss='binary_crossentropy',
                   metrics=['binary_accuracy'])

a=1

# callbacks

model_path = 'models/drive_segmentation_2/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

history = keras.callbacks.History()

def scheduler(epoch):
  if epoch < 100:
    return 0.0001
  else:
    return 0.0001 * np.math.exp(0.2 * (100 - epoch))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint = keras.callbacks.ModelCheckpoint(
    # filepath=model_path + '/weights-improvement-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5',
    filepath=model_path + '/weights-improvement-{epoch:02d}.hdf5',
    # monitor='val_loss',
    verbose=1,
    save_freq=2000,
    save_best_only=False)

earlystopping = keras.callbacks.EarlyStopping(patience=8, verbose=1)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.1,
                                                 patience=5, cooldown=3, min_lr=1e-6, verbose=1)

csv_logger = keras.callbacks.CSVLogger(model_path + 'training.log')

callbacks_list = [history, csv_logger, checkpoint]

a=1

unet_model.fit_generator(train_generator,
                         # validation_data=valid_generator,
                         steps_per_epoch=len(train_generator),
                         # validation_steps=len(valid_generator),
                         # use_multiprocessing=True, workers=8,
                         epochs=1000,
                         callbacks=callbacks_list)

unet_model.save(model_path + 'model.h5')

a=1

####### evaluate

y_pred = []
y_true = []

for i in tqdm.tqdm(range(len(valid_generator))):
    X, y = valid_generator.__getitem__(i)
    y_pred.append(unet_model.predict(X))
    y_true.append(y)

y_pred = np.concatenate(y_pred, axis=0)
y_true = np.concatenate(y_true, axis=0)

y_pred = np.uint8(y_pred > 0.5)
# y_pred = np.argmax(y_pred, axis=-1)

for i in tqdm.tqdm(range(y_pred.shape[0])):
    cv2.imwrite('imgdump/{}_pred.png'.format(i), y_pred[i, ...]*255)
    cv2.imwrite('imgdump/{}_true.png'.format(i), y_true[i, ...]*255)

a=1
