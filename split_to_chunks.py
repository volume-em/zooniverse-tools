import PIL
import os
import argparse
import scipy
import skimage
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from PIL import Image, ImageSequence
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('im_file', type=str, 
                        help='Path to zooniverse batch images')
    parser.add_argument('mask_file', type=str, 
                        help='Directory from which to load masks')
    parser.add_argument('csv_file', type=str, 
                        help='Directory in which to save splitchunked masks')
    parser.add_argument('save_dir', type=str, 
                        help='Directory in which to save splitchunked masks')
    parser.add_argument('--cs', type=int, default=42, 
                        help='chunk size')
    
    args = parser.parse_args()
    

    imf = args.im_file
    segf = args.mask_file
    sdir = args.save_dir
    chunk_size = args.cs
    csvf = args.csv_file
    
    im = io.imread(imf, plugin="tifffile") #or tifffile.imread(imf)
    seg = io.imread(segf, plugin="tifffile")
    attr_csv = pd.read_csv(csvf)

    start = 0
    stop = len(attr_csv)
    step = chunk_size
    remain = stop - (stop % step)
    sindices = range(start, stop, step) 
    eindices = range(step, stop + step, step)
    batch_name = imf.split('_')[0]
    
    print(os.path.join(sdir, f'splitimages.tif'))
    for flipbook_s,flipbook_e in zip(sindices, eindices):
        # each flipbook is 6 images (5 + 1 padding)
        # convert from flipbook index to image stack indices
        flipbook_e = min(stop, flipbook_e)
        
        image_stack_s = flipbook_s * 6
        image_stack_e = flipbook_e * 6
        fbs_str, fbe_str = str(flipbook_s).zfill(6), str(flipbook_e).zfill(6)
                
        impath = os.path.join(sdir, f'{batch_name}_splitimages{fbs_str}-{fbe_str}.tif')
        segpath = os.path.join(sdir, f'{batch_name}_splitlabels{fbs_str}-{fbe_str}.tif')
        csvpath = os.path.join(sdir, f'{batch_name}_attr{fbs_str}-{fbe_str}.csv')
        io.imsave(impath, im[image_stack_s:image_stack_e], check_contrast=False)
        io.imsave(segpath, seg[image_stack_s:image_stack_e], check_contrast=False)
        chunk_csv = attr_csv[flipbook_s:flipbook_e]
        chunk_csv['start'] = chunk_csv['start'] - image_stack_s
        chunk_csv['end'] = chunk_csv['end'] - image_stack_s
        chunk_csv.to_csv(csvpath, index=False)
        
