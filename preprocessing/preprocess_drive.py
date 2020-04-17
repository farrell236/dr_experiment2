import cv2
import glob
import imageio
import os
import tqdm

import numpy as np

from preprocess_retina import _border_pad, _get_retina_bb

def autodir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


################################################################################
## Pre-processing parameters
################################################################################

dataset = 'TEST'
data_root = 'DRIVE/'
res = 1024


################################################################################
## Constants
################################################################################

if dataset == 'TRAIN':
    images      = range(21,41)
    img_root    = data_root + '/training/images'
    msk_root    = data_root + '/training/1st_manual'
    out_folder  = 'DRIVE_train'
    f           = 'training'
elif dataset == 'TEST':
    images      = range(1,21)
    img_root    = data_root + '/test/images'
    out_folder  = 'DRIVE_test'
    f           = 'test'
else:
    print('Invalid mode')
    exit(0)

autodir(out_folder)

for i in tqdm.tqdm(images):

    # Read in image
    image = cv2.imread(img_root + '/{:02d}_{}.tif'.format(i,f))

    # Get retina bounding box
    x, y, w, h = _get_retina_bb(image)

    # Crop image to bbox
    image = image[y:y+h,x:x+w]

    # pad image to square
    image = _border_pad(image)

    # resize image to final shape
    image = cv2.resize(image, (res, res))

    # save image
    cv2.imwrite(out_folder + '/{:02d}_{}.png'.format(i,f), np.uint8(image))

    # process mask
    if dataset == 'TRAIN':
        mask = imageio.imread(msk_root + '/{:02d}_manual1.gif'.format(i))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask[y:y + h, x:x + w]
        mask = _border_pad(mask)
        mask = cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_folder + '/{:02d}_manual1.png'.format(i), mask)




