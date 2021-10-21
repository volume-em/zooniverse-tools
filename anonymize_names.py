import os
import random
import string
import argparse
import pandas as pd
from glob import glob

def make_random_prefix(size=20):
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
    parser.add_argument('imdir', type=str)
    args = parser.parse_args()
    
    imdir = args.imdir
    
    # glob all the jpgs
    impaths = glob(os.path.join(imdir, '*.tiff'))
    
    # prefix is everything before LOC
    prefixes = [os.path.basename(fp).split('-LOC-')[0] for fp in impaths]
    
    prefixes = {}
    used_strings = []
    for fp in impaths:
        prefix = os.path.basename(fp).split('-LOC-')[0]
        if prefix not in prefixes:
            random_prefix = make_random_prefix(20)
            assert random_prefix not in used_strings
            prefixes[prefix] = random_prefix
        else:
            random_prefix = prefixes[prefix]
        
        new_path = fp.replace(prefix, random_prefix)
        os.rename(fp, new_path)
        
    prefix_df = {'prefix': [], 'random_prefix': []}
    for prefix,random_prefix in prefixes.items():
        prefix_df['prefix'].append(prefix)
        prefix_df['random_prefix'].append(random_prefix)
        
    pd.DataFrame.from_dict(prefix_df).to_csv(os.path.join(imdir, 'prefix_conversion.csv'), index=False)