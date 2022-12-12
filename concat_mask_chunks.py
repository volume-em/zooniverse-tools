"""
Takes proofread labelmap chunks and concatentates
them into a single stack. It's assumed that the corresponding
image stack and consensus_attributes.csv were retained from
the create_proofreading_stack.py script. During proofreading
only the labelmaps change.

Arguments:
-----------
fdir: Directory containing the proofread labelmap chunks
chunk_str: A glob string to select the correct image files. Commonly
this should be just be "masks" based on the naming convention in 
split_to_chunks.py. Selected chunks are automatically ordered based on
the '{chunk}_{s}-{e}' part of the filename from split_to_chunks.py

"""

import os
import argparse
import numpy as np
from glob import glob
from skimage import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fdir', type=str)
    parser.add_argument('chunk_str', type=str, default='masks')
    args = parser.parse_args()

    fpaths = glob(os.path.join(args.fdir, f'*{args.chunk_str}*.tif'))
    fpaths = sorted(fpaths)

    stack = []
    for fp in fpaths:
        print('Loading', fp)
        image = io.imread(fp)
        stack.append(image)

    stack = np.concatenate(stack, axis=0)
    print(f'Created stack of size {stack.shape}')

    # save the stack 
    out_name = os.path.basename(fpaths[0]).split('_chunk')[0]
    out_name += '_cs_masks_proof.tif'
    out_fpath = os.path.join(os.path.dirname(fpaths[0]), out_name)
    io.imsave(out_fpath, stack, check_contrast=False)
