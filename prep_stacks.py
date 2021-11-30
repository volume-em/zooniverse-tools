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
    parser.add_argument('imdir', type=str)
    parser.add_argument('savedir', type=str)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--contrast', action='store_true')
    args = parser.parse_args()
    
    imdir = args.imdir
    savedir = args.savedir
    size = args.size
    contrast = args.contrast
    
    # glob all the images
    fpaths = glob(os.path.join(imdir, '*.tif'))
    print(f'Found {len(fpaths)} .tif stacks to prepare.')
    
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
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # fix the contrast
            if contrast:
                image = rescale_intensity(image, in_range=(0, 255), out_range=(25, 230))
                
            # save the jpg
            out_fname = os.path.join(savedir, fname.replace('.tif', f'_{i}.jpg'))
            io.imsave(out_fname, image, quality=100, check_contrast=False)