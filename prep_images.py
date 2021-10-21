import os, cv2
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage import io
from skimage.exposure import equalize_hist

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
    fpaths = glob(os.path.join(imdir, '*.tiff'))
    print(f'Found {len(fpaths)} .tiff images to prepare.')
    
    # create savedir if is doesn't exist
    os.makedirs(savedir, exist_ok=True)
    
    # process images
    for fp in tqdm(fpaths):
        # extract the fname
        fname = os.path.basename(fp)
        image = cv2.imread(fp, 0)
        
        # resize the image
        image = cv2.resize(image, (size, size))

        # fix the contrast
        if contrast:
            image = equalize_hist(image)
            image -= image.min()
            image /= image.max()
            image *= 255
            image = image.astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # save the jpg
        out_fname = os.path.join(savedir, fname.replace('.tiff', '.jpg'))
        io.imsave(out_fname, image, quality=100, check_contrast=False)