"""
Script to anonymize the names of tiff images that may contain
sensitive info like the name of investigators/collaborators.

Arguments:
-----------
imdir: A directory of .tif(f) images to anonymize.

Results:
--------
1. Renames all .tif(f) images with a new anonymous name
2. Saves a .csv file with 2 columns: the old name and the new name.

"""

import os
import random
import string
import argparse
import pandas as pd
from glob import glob

ORDERED_SPLIT_STRS = [
    '-ROI-',
    '-LOC-2d-',
    '-LOC-'
]

def make_random_prefix(size=20):
    """
    Creates a random alphanumeric prefix with length "size".
    """
    # printing letters
    letters = string.ascii_letters
    digits = string.digits
    
    pstr = []
    for _ in range(size):
        if random.random() < 0.5:
            pstr.append(random.choice(letters))
        else:
            pstr.append(random.choice(digits))
            
    return ''.join(pstr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', type=str, help="Directory containing .tif(f) image files")
    args = parser.parse_args()
    
    imdir = args.imdir
    
    # glob all the tiffs
    impaths = glob(os.path.join(imdir, '*.tif*'))
    
    if not impaths:
        raise Exception(f'Directory {imdir} if expected to contain .tif(f) images! None found.')
    
    # prefix is everything before split-str
    # if split-str isn't in the filename then
    # each image will have a unique random prefix
    # (this is bad if you want to group images by source dataset)
    prefixes = {}
    for fp in impaths:
        prefix = '.'.join(os.path.basename(fp).split('.')[:-1])
        for split_str in ORDERED_SPLIT_STRS:
            if split_str in prefix:
                prefix = prefix.split(split_str)[0]
                break
            
        if prefix not in prefixes:
            random_prefix = make_random_prefix(20)
            prefixes[prefix] = random_prefix
        else:
            random_prefix = prefixes[prefix]
        
        # move the image to a new name, no copying!
        new_path = fp.replace(prefix, random_prefix)
        os.rename(fp, new_path)
        
    # store the prefixes and original names
    # for reference or de-anonymization later
    prefix_df = {'prefix': [], 'random_prefix': []}
    for prefix,random_prefix in prefixes.items():
        prefix_df['prefix'].append(prefix)
        prefix_df['random_prefix'].append(random_prefix)
        
    pd.DataFrame.from_dict(prefix_df).to_csv(os.path.join(imdir, 'prefix_conversion.csv'), index=False)