"""
Prepares a directory of grayscale flipbooks for upload to Zooniverse:

1. Resizes the flipbook in (h, w) to be square with a given edge length (--size)

2. Optionally, normalizes the contrast of the flipbook with histogram
equalization and rescaling from 25 to 230 (--contrast)

3. Saves each image in the flipbook as a separate jpg. The original name
of the flipbook has a suffix '_{zindex}.jpg' added. The zindex is the position
of the image in the flipbook. E.g. the first image will be called 'flipbook_name_0.jpg' and
the third image in the flipbook will be 'flipbook_name_2.jpg'

"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage import io
from skimage import transform
from skimage.exposure import equalize_hist, rescale_intensity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', type=str, help='Directory containing tif images to prepare')
    parser.add_argument('savedir', type=str, help='Directory in which to save processed jpgs')
    parser.add_argument('--size', type=int, default=480, help='Square dimension of resized image')
    parser.add_argument('--contrast', action='store_true', help='Whether to equalize and rescale image contrast')
    args = parser.parse_args()
    
    imdir = args.imdir
    savedir = args.savedir
    size = args.size
    contrast = args.contrast
    
    # glob all the images
    fpaths = glob(os.path.join(imdir, '*.tif*'))
    print(f'Found {len(fpaths)} .tif flipbooks to prepare.')
    
    # create savedir if is doesn't exist
    os.makedirs(savedir, exist_ok=True)
    
    # process images
    for fp in tqdm(fpaths):
        # extract the fname
        fname = os.path.basename(fp)
        stack = io.imread(fp)
        
        for i, image in enumerate(stack):
            # resize the image
            image = transform.resize(image, (size, size), preserve_range=True)
            
            # fix the contrast
            if contrast:
                image = rescale_intensity(image, in_range=(0, 255), out_range=(25, 230))
                
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # save the jpg
            out_fname = os.path.join(savedir, fname.replace('.tif', f'_{i}.jpg'))
            io.imsave(out_fname, image, quality=100, check_contrast=False)