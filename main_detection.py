import cv2

import numpy as np
import tensorflow as tf

from tensorflow import keras

from dataloader.eyepacs import DataGenerator

train_csv = 'data/EyePACS/train_all_df.csv'
train_root = 'preprocessing/train_C0_R1024x1024'
test_csv = 'data/EyePACS/test_public_df.csv'
test_root = 'preprocessing/test_C0_R1024x1024'

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

res = 512

train_generator = DataGenerator(train_csv, train_root, batch_size=24, resize=res, transform=data_transforms)
valid_generator = DataGenerator(test_csv, test_root, batch_size=8, resize=res, transform=None, shuffle=False)

a=1

# this could also be the output a different Keras model or layer
input_tensor = keras.layers.Input(shape=(res, res, 3))  # this assumes K.image_data_format() == 'channels_last'

# create the base pre-trained model
base_model = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)

# add a global spatial average pooling layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = keras.layers.Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = keras.layers.Dense(5, activation='softmax')(x)

# this is the model we will train
model = keras.models.Model(inputs=input_tensor, outputs=predictions)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='dr_detection.h5',
    monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True)

model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

a=1


model.fit_generator(train_generator,
                    validation_data=valid_generator,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(valid_generator),
                    # use_multiprocessing=True, workers=8,
                    epochs=20,
                    callbacks=[checkpoint])


a=1

# y_pred = []
# y_true = []
#
# for i in range(len(valid_generator)):
#     X, y = valid_generator.__getitem__(i)
#     y_pred.append(np.argmax(np.uint8(model.predict(X) > 0.5), axis=-1))
#     y_true.append(np.squeeze(y))
#     print(i)
#
# y_pred = np.concatenate(y_pred)
# y_true = np.concatenate(y_true)




a=1



