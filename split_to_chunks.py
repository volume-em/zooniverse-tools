import os
import argparse
import pandas as pd
from glob import glob
from skimage import io

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
    parser.add_argument('--flipbook', action='store_true', 
                        help='Whether images in stack are of flipbooks.')
    parser.add_argument('--cs', type=int, default=42, 
                        help='chunk size')
    
    args = parser.parse_args()
    imf = args.im_file
    segf = args.mask_file
    sdir = args.save_dir
    chunk_size = args.cs
    csvf = args.csv_file

    os.makedirs(sdir, exist_ok=True)
    
    im = io.imread(imf)
    seg = io.imread(segf)
    attr_csv = pd.read_csv(csvf)

    start = 0
    stop = len(attr_csv)
    step = chunk_size
    sindices = range(start, stop, step) 
    eindices = range(step, stop + step, step)
    batch_name = os.path.basename(imf).split('_')[0]
    
    for flipbook_s,flipbook_e in zip(sindices, eindices):
        # each flipbook is 6 images (5 + 1 padding)
        # convert from flipbook index to image stack indices
        flipbook_e = min(stop, flipbook_e)
        
        if args.flipbook:
            image_stack_s = flipbook_s * 6
            image_stack_e = flipbook_e * 6
        else:
            image_stack_s = flipbook_s
            image_stack_e = flipbook_e
            
        fbs_str, fbe_str = str(flipbook_s).zfill(4), str(flipbook_e).zfill(4)
                
        impath = os.path.join(sdir, f'{batch_name}_chunk_{fbs_str}-{fbe_str}.tif')
        segpath = os.path.join(sdir, f'{batch_name}_chunk_{fbs_str}-{fbe_str}_labels.tif')
        csvpath = os.path.join(sdir, f'{batch_name}_attr_chunk_{fbs_str}-{fbe_str}.csv')
        io.imsave(impath, im[image_stack_s:image_stack_e], check_contrast=False)
        io.imsave(segpath, seg[image_stack_s:image_stack_e], check_contrast=False)

        # process the csv file
        chunk_csv = attr_csv[flipbook_s:flipbook_e]
        chunk_csv['start'] = chunk_csv['start'] - image_stack_s
        chunk_csv['end'] = chunk_csv['end'] - image_stack_s
        chunk_csv.to_csv(csvpath, index=False)
        
