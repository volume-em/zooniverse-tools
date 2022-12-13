"""
Saves images and masks from stacks into a directory.

"""
import os
import argparse
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure

ORDERED_SPLIT_STRS = [
    '-ROI-',
    '-LOC-2d-',
    '-LOC-'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image stack .tif file')
    parser.add_argument('mask', type=str, help='Path to mask stack .tif file')
    parser.add_argument('attributes', type=str, help='Path to consensus_attributes.csv file')
    parser.add_argument('save_dir', type=str, help='Directory in which to save segmentations')
    parser.add_argument('--ignore', type=int, nargs='+', help='Stack indices to skip.')
    parser.add_argument('--no-groups', action='store_true', 
                        help='If flag is passed images will not be organized into subdirectories')
    args = parser.parse_args()
    
    # read in the volumes and attributes
    image = io.imread(args.image)
    mask = io.imread(args.mask)
    if image.shape != mask.shape:
        raise Exception(f'Image and mask must have the same shape!')
    
    ignore = args.ignore if args.ignore is not None else []
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.no_groups:
        os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'masks'), exist_ok=True)
    
    if image.ndim == 4: # stack of flipbooks
        span = image.shape[1] # flipbook length
    elif image.ndim == 3: # stack of grayscale images
        span = 0
    else:
        raise Exception(f'Image is expected to be 3d or 4d! Got {image.ndim}')
    
    attrs_df = pd.read_csv(args.attributes)
    if len(image) != len(attrs_df):
        raise Exception(f'Image and csv must have the same length!')
    
    for attr_idx, im_attrs in attrs_df.iterrows():
        fname = '.'.join(im_attrs['image_name'].split('.')[:-1])
        
        stack_idx = im_attrs['stack_index']
        midpt = span // 2 # midpoint index of flipbook
        
        # skip images in ignore
        if stack_idx in ignore:
            continue
        
        # get the real dimensions before padding
        if 'height' in im_attrs:
            h = int(im_attrs['height'])
            w = int(im_attrs['width'])
        else:
            # calculate padding from image if missing in csv
            im = image[stack_idx]
            if im.ndim == 3:
                im = im[midpt]
                
            h = np.any(im, axis=1).nonzero()[0][-1] + 1
            w = np.any(im, axis=0).nonzero()[0][-1] + 1
        
        im = image[stack_idx, :h, :w]
        msk = mask[stack_idx, :h, :w].astype(np.int32)
        
        # slice out the middle image from flipbook
        if im.ndim == 3:
            im = im[midpt]
            msk = msk[midpt]
            
        # save the image to the correct subdirectory
        # such that it is grouped with images from the
        # same source dataset
        dataset_name = fname
        for split_str in ORDERED_SPLIT_STRS:
            if split_str in fname:
                dataset_name = fname.split(split_str)[0]
                break
                
        if args.no_groups:
            fdir = args.save_dir
        else:
            fdir = os.path.join(args.save_dir, dataset_name)
            os.makedirs(fdir, exist_ok=True)
            os.makedirs(os.path.join(fdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(fdir, 'masks'), exist_ok=True)
        
        io.imsave(os.path.join(fdir, f'images/{fname}.tiff'), im, check_contrast=False)
        io.imsave(os.path.join(fdir, f'masks/{fname}.tiff'), msk, check_contrast=False)