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
    
    # prefix is everything before LOC
    # if -LOC- isn't in the name then the
    # whole image name will be anonymized
    prefixes = [os.path.basename(fp).split('-LOC-')[0] for fp in impaths]
    
    prefixes = {}
    used_strings = []
    for fp in impaths:
        prefix = os.path.basename(fp).split('-LOC-')[0]
        if prefix not in prefixes:
            random_prefix = make_random_prefix(20)
            assert random_prefix not in used_strings # virtually 0 chance
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