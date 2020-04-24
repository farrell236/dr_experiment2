import cv2
import glob
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
data_root = '/vol/vipdata/data/retina/IDRID/A. Segmentation'
res = 1024
normalize = 0
clip = 0

################################################################################
## Constants
################################################################################

if dataset == 'TRAIN':
    images = range(1, 55)
    img_root = data_root + '/1. Original Images/a. Training Set'
    msk_root = data_root + '/2. All Segmentation Groundtruths/a. Training Set'
    idx = np.arange(1, 55)
    out_folder = 'IDRID_train'
elif dataset == 'TEST':
    images = range(55, 82)
    img_root = data_root + '/1. Original Images/b. Testing Set'
    msk_root = data_root + '/2. All Segmentation Groundtruths/b. Testing Set'
    idx = np.arange(55, 82)
    out_folder = 'IDRID_test'
else:
    print('Invalid mode')
    exit(0)

autodir(out_folder)
autodir(out_folder + '/images')
autodir(out_folder + '/masks')
autodir(out_folder + '/masks/onehot')
autodir(out_folder + '/masks/rgb')

mask_shape = (2848, 4288)  # default size for all IDRID images

colour_class = {
    'retina': (255, 255, 255),
    'MA': (242, 80, 34),
    'HE': (127, 186, 0),
    'EX': (0, 164, 239),
    'SE': (255, 185, 0),
    'OD': (115, 115, 115),
    'BG': (0, 0, 0),
}

_retina = np.zeros((2848, 4288, 3)).astype('uint8')
_retina[:] = colour_class['retina']
_BG = np.zeros((2848, 4288, 3)).astype('uint8')
_BG[:] = colour_class['BG']
_OD = np.zeros((2848, 4288, 3)).astype('uint8')
_OD[:] = colour_class['OD']
_MA = np.zeros((2848, 4288, 3)).astype('uint8')
_MA[:] = colour_class['MA']
_SE = np.zeros((2848, 4288, 3)).astype('uint8')
_SE[:] = colour_class['SE']
_HE = np.zeros((2848, 4288, 3)).astype('uint8')
_HE[:] = colour_class['HE']
_EX = np.zeros((2848, 4288, 3)).astype('uint8')
_EX[:] = colour_class['EX']

b = np.zeros((res, res, 3))
c = np.clip(clip, 0, 1)
cv2.circle(b, (res // 2, res // 2), int(res // 2 * c), (1, 1, 1), -1, 8, 0)

for i in tqdm.tqdm(idx):

    ###########################################################################
    ## Read and pre-processing images
    ###########################################################################

    # Read in image
    image = cv2.imread(img_root + '/IDRiD_{:02d}.jpg'.format(i))

    # Get retina bounding box
    x, y, w, h, retina = _get_retina_bb(image)
    retina = np.uint8(retina > 0)

    # Crop image to bbox
    image = image[y:y + h, x:x + w]

    # pad image to square
    image = _border_pad(image)

    # resize image to final shape
    image = cv2.resize(image, (res, res))

    # If normalise colour
    if normalize:
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), res * 0.01), -4, 128)

    # If clip retina boundary
    if clip:
        image = image * b

    # save image
    cv2.imwrite(out_folder + '/images/IDRiD_{:02d}.png'.format(i), image)

    ###########################################################################
    ## Read and pre-processing mask (n.b. not all subjects have a mask)
    ###########################################################################

    try:
        MA = np.uint8(cv2.imread(
            msk_root + '/1. Microaneurysms/IDRiD_{:02d}_MA.tif'.format(i), 0) > 0)
    except:
        MA = np.zeros(mask_shape).astype('uint8')

    try:
        HE = np.uint8(cv2.imread(
            msk_root + '/2. Haemorrhages/IDRiD_{:02d}_HE.tif'.format(i), 0) > 0)
    except:
        HE = np.zeros(mask_shape).astype('uint8')

    try:
        EX = np.uint8(cv2.imread(
            msk_root + '/3. Hard Exudates/IDRiD_{:02d}_EX.tif'.format(i), 0) > 0)
    except:
        EX = np.zeros(mask_shape).astype('uint8')

    try:
        SE = np.uint8(cv2.imread(
            msk_root + '/4. Soft Exudates/IDRiD_{:02d}_SE.tif'.format(i), 0) > 0)
    except:
        SE = np.zeros(mask_shape).astype('uint8')

    try:
        OD = np.uint8(cv2.imread(
            msk_root + '/5. Optic Disc/IDRiD_{:02d}_OD.tif'.format(i), 0) > 0)
    except:
        OD = np.zeros(mask_shape).astype('uint8')

    ###########################################################################
    ## Some Masks overlap in pixels, make unique
    ###########################################################################

    # Pixel Priority:
    # EX -> HE -> SE -> MA -> OD -> retina
    EX_HE = np.uint8((EX + HE) > 0)  # cumulative mask (EX + HE)
    EX_HE_SE = np.uint8((EX + HE + SE) > 0)  # cumulative mask (EX + HE + SE)
    EX_HE_SE_MA = np.uint8((EX + HE + SE + MA) > 0)  # cumulative mask (EX + HE + SE + MA)
    EX_HE_SE_MA_OD = np.uint8((EX + HE + SE + MA + OD) > 0)  # cumulative mask (EX + HE + SE + MA + OD)
    EX_HE_SE_MA_OD_retina = np.uint8((EX + HE + SE + MA + OD + retina) > 0)
    BG = 1 - EX_HE_SE_MA_OD_retina

    # BG = BG #* 255
    retina = retina * (1 - EX_HE_SE_MA_OD)  # *255
    OD = OD * (1 - EX_HE_SE_MA)  # * 255
    MA = MA * (1 - EX_HE_SE)  # * 255
    SE = SE * (1 - EX_HE)  # * 255
    HE = HE * (1 - EX)  # * 255
    # EX = EX #* 255

    # sanity check, the sum of all mask should be 1 to ensure no overlap
    # t=EX+HE+SE+MA+OD+BG
    # print(i, t.max())

    ###########################################################################
    ## Make mask for one-hot
    ###########################################################################

    one_hot = np.stack((BG, EX, HE, SE, MA, OD, retina), axis=-1)
    one_hot_vec = np.argmax(one_hot, axis=-1)
    one_hot_vec = one_hot_vec[y:y + h, x:x + w]
    one_hot_vec = np.expand_dims(one_hot_vec, axis=-1)
    one_hot_vec = _border_pad(one_hot_vec)
    one_hot_vec = cv2.resize(one_hot_vec, (res, res), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(out_folder + '/masks/onehot/IDRiD_{:02d}.png'.format(i), one_hot_vec)

    ###########################################################################
    ## Make mask for RGB
    ###########################################################################

    retina = cv2.bitwise_and(_retina, _retina, mask=retina)
    BG = cv2.bitwise_and(_BG, _BG, mask=BG)
    OD = cv2.bitwise_and(_OD, _OD, mask=OD)
    MA = cv2.bitwise_and(_MA, _MA, mask=MA)
    SE = cv2.bitwise_and(_SE, _SE, mask=SE)
    HE = cv2.bitwise_and(_HE, _HE, mask=HE)
    EX = cv2.bitwise_and(_EX, _EX, mask=EX)

    mask = EX + HE + SE + MA + OD + BG + retina
    mask = mask[y:y + h, x:x + w]
    mask = _border_pad(mask)
    mask = cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(out_folder + '/masks/rgb/IDRiD_{:02d}.png'.format(i), mask)






