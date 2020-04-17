import argparse
import cv2
import glob
import os
import tqdm

import numpy as np


def _border_pad(image, long_side=None):
    h, w, _ = image.shape

    if long_side==None: long_side=max(h,w)

    l_pad = (long_side - w) // 2
    r_pad = (long_side - w) // 2
    t_pad = (long_side - h) // 2
    b_pad = (long_side - h) // 2
    if w % 2 != 0: r_pad = r_pad + 1
    if h % 2 != 0: b_pad = b_pad + 1

    image = np.pad(
        image,
        ((t_pad, b_pad),
         (l_pad, r_pad),
         (0,0)),
        'constant')

    return image


def _get_retina_bb(image_in):

    # make image greyscale and normalise
    image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # calculate threshold perform threshold
    threshold = np.mean(image)/3-7
    ret, image = cv2.threshold(image, max(0,threshold), 255, cv2.THRESH_BINARY)

    # median filter, erode and dilate to remove noise and holes
    image = cv2.medianBlur(image, 25)
    image = cv2.erode(image, None, iterations=2)
    image = cv2.dilate(image, None, iterations=2)

    # find mask contour
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # Get bounding box from mask contour
    x,y,w,h = cv2.boundingRect(c)

    return x, y, w, h


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess Retina Fundus Images')
    parser.add_argument('-i', '--input_dir', default=None, required=True,
                        help='path to the folder that contain source images')
    parser.add_argument('-o', '--output_dir', default=None, required=True,
                        help='path to the output directory')
    parser.add_argument('-r', '--resolution', default=1024, type=int, required=True,
                        help='resolution of output images (r x r)')
    parser.add_argument('-n', '--normalize', dest='normalize', action='store_true',
                        help='colour normalize images')
    parser.add_argument('-c', '--clip', default=None, type=float,
                        help='boundary clip the images by factor (0 ~ 1]')
    args = parser.parse_args()

    images = sorted(glob.glob(args.input_dir + '/*'))
    
    res = args.resolution

    # Create clipping mask
    if args.clip:
        b = np.zeros((res,res,3))
        c = np.clip(args.clip, 0, 1)
        cv2.circle(b,(res//2,res//2),int(res//2*c),(1,1,1),-1,8,0)

    # Create output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Main processing loop
    for i in tqdm.tqdm(range(len(images))):
    
        # Read in image
        image = cv2.imread(images[i])
    
        # Get retina bounding box
        x, y, w, h = _get_retina_bb(image)
    
        # Crop image to bbox
        image = image[y:y+h,x:x+w]
    
        # Pad image to square
        image = _border_pad(image)
    
        # Resize image to final shape
        image = cv2.resize(image, (res, res))
    
        # If normalise colour
        if args.normalize:
            image = cv2.addWeighted(image,4,cv2.GaussianBlur(image,(0,0),res*0.01),-4,128)

        # If clip retina boundary
        if args.clip:
            image = image * b
    
        # Save image
        cv2.imwrite(args.output_dir+'/'+os.path.basename(images[i]),image)
