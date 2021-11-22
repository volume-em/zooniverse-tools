import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from skimage import io
from skimage import measure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fdir', type=str)
    parser.add_argument('chunk_str', type=str)
    parser.add_argument('--mask', action='store_true')
    args = parser.parse_args()

    fpaths = glob(os.path.join(args.fdir, f'*{args.chunk_str}*'))
    fpaths = sorted(fpaths)#, key=lambda fp: int(fp.split('labels')[-1].split('_')[0].split('-')[0]))
    #fpaths = list(filter(lambda x: 'label' not in x, fpaths))
    #fpaths = list(filter(lambda x: 'mask' not in x, fpaths))

    stack = []
    for fp in fpaths:
        print('Loading', fp)
        image = io.imread(fp).astype(np.uint8)
        print(image.shape, image.dtype, image.max(), image.min())
        stack.append(image)

    stack = np.concatenate(stack, axis=0)
    print(stack.dtype, stack.max(), stack.min())
    print(f'Created stack of size {stack.shape}')

    out_name = os.path.basename(fpaths[0]).split('_')[0]
    
    # now label connected components
    if args.mask:
        for index in range(2, len(stack), 6):
            stack[index] = measure.label(stack[index]).astype(stack.dtype)

        out_name += '_cs_masks_proofed.tif'

    else:
        out_name += '_images.tif'

    out_fpath = os.path.join(os.path.dirname(fpaths[0]), out_name)

    io.imsave(out_fpath, stack, check_contrast=False)
