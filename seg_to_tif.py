import os
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from skimage import io

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fdir', type=str)
	parser.add_argument('im_str', type=str)
	args = parser.parse_args()

	impaths = sorted(glob(os.path.join(args.fdir, f'*{args.im_str}*')))
	segpaths = sorted(glob(os.path.join(args.fdir, f'*.seg.nrrd')))

	for imp, segp in zip(impaths, segpaths):
		print(imp, segp)

		im = sitk.ReadImage(imp)
		seg = sitk.ReadImage(segp)

		resamp = sitk.ResampleImageFilter()
		resamp.SetReferenceImage(im)
		resamp.SetInterpolator(sitk.sitkNearestNeighbor)
		seg = resamp.Execute(seg)
		seg = sitk.GetArrayFromImage(seg)

		# add empty background channel at zero
		seg = np.concatenate([np.zeros_like(seg[..., :1]), seg], axis=-1)
		seg = seg.argmax(-1)

		if seg.shape[-1] > 255:
			raise Exception('Segmentation has too many labels to be converted to 8-bit!')

		seg = seg.astype(np.uint8)
		io.imsave(segp.replace('.seg.nrrd', '.tif'), seg, check_contrast=False)
