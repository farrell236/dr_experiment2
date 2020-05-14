# This file is based on the BCDU-Net by Azad et. al
# https://github.com/rezazad68/BCDU-Net

import numpy as np

from tensorflow import keras


def BCDU_net_D3(inputs, output_ch=1, N=256, activation='sigmoid'):

    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = keras.layers.Dropout(0.5)(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_1 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4_1 = keras.layers.Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4_1)
    conv4_2 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_2)
    conv4_2 = keras.layers.Dropout(0.5)(conv4_2)
    # D3
    merge_dense = keras.layers.concatenate([conv4_2, drop4_1], axis=3)
    conv4_3 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_dense)
    conv4_3 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_3)
    drop4_3 = keras.layers.Dropout(0.5)(conv4_3)
    up6 = keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_3)
    up6 = keras.layers.BatchNormalization(axis=3)(up6)
    up6 = keras.layers.Activation('relu')(up6)

    x1 = keras.layers.Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(drop3)
    x2 = keras.layers.Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
    merge6 = keras.layers.concatenate([x1, x2], axis=1)
    merge6 = keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge6)

    conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    up7 = keras.layers.BatchNormalization(axis=3)(up7)
    up7 = keras.layers.Activation('relu')(up7)

    x1 = keras.layers.Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
    x2 = keras.layers.Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
    merge7 = keras.layers.concatenate([x1, x2], axis=1)
    merge7 = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge7)

    conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = keras.layers.BatchNormalization(axis=3)(up8)
    up8 = keras.layers.Activation('relu')(up8)

    x1 = keras.layers.Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = keras.layers.Reshape(target_shape=(1, N, N, 64))(up8)
    merge8 = keras.layers.concatenate([x1, x2], axis=1)
    merge8 = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge8)

    conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = keras.layers.Conv2D(output_ch, 1, activation=activation)(conv8)

    return conv9


def BCDU_net_D1(inputs, output_ch=1, N=256, activation='sigmoid'):

    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = keras.layers.Dropout(0.5)(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_1 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4_1 = keras.layers.Dropout(0.5)(conv4_1)

    up6 = keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_1)
    up6 = keras.layers.BatchNormalization(axis=3)(up6)
    up6 = keras.layers.Activation('relu')(up6)

    x1 = keras.layers.Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(drop3)
    x2 = keras.layers.Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
    merge6 = keras.layers.concatenate([x1, x2], axis=1)
    merge6 = keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge6)

    conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    up7 = keras.layers.BatchNormalization(axis=3)(up7)
    up7 = keras.layers.Activation('relu')(up7)

    x1 = keras.layers.Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
    x2 = keras.layers.Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
    merge7 = keras.layers.concatenate([x1, x2], axis=1)
    merge7 = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge7)

    conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = keras.layers.BatchNormalization(axis=3)(up8)
    up8 = keras.layers.Activation('relu')(up8)

    x1 = keras.layers.Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = keras.layers.Reshape(target_shape=(1, N, N, 64))(up8)
    merge8 = keras.layers.concatenate([x1, x2], axis=1)
    merge8 = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge8)

    conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = keras.layers.Conv2D(output_ch, 1, activation=activation)(conv8)

    return conv9


