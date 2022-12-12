"""
This scripts takes an annotation_csv downloaded from Zooniverse and
local image source directories and returns a single stack of
all images/flipbooks and correspond consensus annotations.

Arguments:
-----------
annotation_csv: Path to the Zooniverse downloaded annotations.

image_dir: Directory that contains the source images uploaded
to Zooniverse. This should be the wdir from upload_images.ipynb.

save_dir: Directory in which to save results.

default_size: It's assumed that all images uploaded to Zooniverse
have the same (h, w) dimensions and are square. This parameter should
be set to this size. Instance segmentations will be incorrectly positioned
if this parameter is set incorrectly.

flipbook: Flag denoting that the annotations/images are for flipbooks.

flipbook-n: Number of images contained in a single flipbook. 
Should typically be an odd number. Default of 5.

"""

import os
import argparse
import csv
import cv2
import json
import pandas as pd
import numpy as np
from glob import glob
from skimage import io
from tqdm import tqdm
from metrics import average_precision
from helpers import *
from aggregation import mask_aggregation, aggregated_instance_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_csv', type=str,
                        help='Path to zooniverse classification export csv')
    parser.add_argument('image_dir', type=str,
                        help='Directory from which to load images')
    parser.add_argument('save_dir', type=str,
                        help='Directory in which to save consensus masks')
    parser.add_argument('default_size', type=int,
                        help='Expected default square dimension of each image uploaded to Zooniverse')
    parser.add_argument('--flipbook', action='store_true',
                        help='Whether images to create proofreading stack are flipbooks or 2D images.')
    parser.add_argument('--flipbook-n', type='int', default=5,
                        help='If flipbook is True, this is the number of images per flipbook.')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    print('Loading annotation csv...')
    # necessary to handle overflow of long csv columns
    csv.field_size_limit(256<<12)
    csv.field_size_limit()
    results_df = pd.read_csv(args.annotation_csv)
    print(f'Loaded annotation csv with {len(results_df)} rows.')

    # convert the metadata fields to parsable json strings
    results_df['metadata_json'] = [json.loads(q) for q in results_df.metadata]
    results_df['annotations_json'] = [json.loads(q) for q in results_df.annotations]
    results_df['subject_data_json'] = [json.loads(q) for q in results_df.subject_data]

    def parse_annotation(annotation):
        """Helper function for extracting polygons and confidence"""
        confidence = None
        for task in annotation:
            # annotation is task T0 in our workflow
            # with a confidence score question
            if task['task'] == 'T0':
                # if not labeled objects, annotation is blank list
                if task['value']:
                    polygons = [
                        polygon_to_array(value['points']) for value in task['value']
                    ]
                else:
                    # empty mask
                    polygons = []

            else:
                confidence = int(task['value'][:1])

        if confidence is None:
            print(f'No confidence score found for image')
            confidence = 0

        return polygons, confidence

    # extracts all polygon annotations from
    # the results csv and organizes them by the subject_id
    # keys are image names and values are attributes
    subject_annotations = {}
    for i, row in results_df.iterrows():
        # subject id and annotated image name
        subject_id = list(row['subject_data_json'].keys())[0]
        # image name key is middle slice in flipbook or 0
        key_id = f'Image {args.flipbook_n // 2}' if args.flipbook else 'Image 0'
        image_name = row['subject_data_json'][subject_id][key_id]

        # subject height and width
        w, h = (args.default_size, args.default_size)

        # reload or create new subject dict
        if image_name in subject_annotations:
            subject_dict = subject_annotations[image_name]
        else:
            subject_dict = {'id': subject_id, 'shape': (w, h),
                            'confidences': [], 'polygons': []}

        annotation = row['annotations_json']
        metadata = row['metadata_json']

        polygons, confidence = parse_annotation(annotation)
        subject_dict['polygons'].append(polygons)
        subject_dict['confidences'].append(confidence)

        # update the dict
        subject_annotations[image_name] = subject_dict

    # convert from polygons to consensus annotations
    print(f'Reconstructing consensus annotations...')
    consensus_attrs = {}
    for imname, subject_dict in tqdm(subject_annotations.items(), total=len(subject_annotations)):
        subject_id = subject_dict['id']
        image_shape = subject_dict['shape']

        # create instance segmentations from polygons
        masks = []
        for polyset in subject_dict['polygons']:
            masks.append(poly2segmentation(polyset, image_shape))

        # create consensus instance segmentation
        instance_scores = mask_aggregation(masks)
        instance_seg = aggregated_instance_segmentation(instance_scores, vote_thr=0.5)

        # compute average precision between each
        # mask and the consensus, this measures strength
        # of the consensus annotations
        scores = [average_precision(instance_seg, mask, 0.50, False)[0] for mask in masks]

        median_confidence = np.median(subject_dict['confidences'])
        consensus_strength = np.mean(scores)
        consensus_attrs[imname] = {
            'id': subject_id, 'median_confidence': median_confidence,
            'consensus_strength': consensus_strength,
            'seg': instance_seg
        }

    # sort the stacks from lowest to highest consensus strength
    consensus_attrs = sorted(consensus_attrs.items(), key=lambda x: x[1]['consensus_strength'])

    image_stack = []
    mask_stack = []

    consensus_df = pd.DataFrame(columns=[
        'start', 'end', 'image_name', 'zooniverse_id', 
        'median_confidence', 'consensus_strength'
    ])

    idx = 0
    for x in consensus_attrs:
        imname, attrs = x
        # convert from imname to stack name
        if args.flipbook:
            # remove the _zloc from imname
            stack_fname = '_'.join(imname.split('_')[:-1]) + '.tif'
        else:
            stack_fname = imname.replace('.jpg', '.tiff')
            
        # image is either (flipbook_n,)
        image = io.imread(os.path.join(args.image_dir, stack_fname))
        if args.flipbook:
            n, h, w = image.shape
            if n != args.flipbook_n:
                raise Exception(
                    f'Given flipbook-n {args.flipbook_n} does not match length of loaded flipbook {n}'
                )
        else:
            h, w = image.shape

        # resize the mask to the image's h and w
        mask = attrs['seg']

        # cv2 flips height and width
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if args.flipbook:
            flipbook_mask = np.zeros_like(image)
            flipbook_mask[args.flipbook_n // 2] = mask # middle image in stack is the one annotated
            mask = flipbook_mask

        image_stack.append(image)
        mask_stack.append(mask)

        consensus_df = consensus_df.append({
            'stack_index': idx,
            'image_name': stack_fname, 'zooniverse_id': attrs['id'],
            'median_confidence': attrs['median_confidence'],
            'consensus_strength': attrs['consensus_strength'],
            'height': h, 'width': w
        }, ignore_index=True)

        idx += 1

    # save the consensus attributes as a csv
    batch_name = '-'.join(args.annotation_csv.split('-')[:-1])
    attr_fpath = os.path.join(args.save_dir, f'{batch_name}_consensus_attributes.csv')
    consensus_df.to_csv(attr_fpath, index=False)

    # find (h, w) dimensions of largest image
    if args.flipbook:
        max_h = max([img.shape[1] for img in image_stack])
        max_w = max([img.shape[2] for img in image_stack])
    else:
        max_h = max([img.shape[0] for img in image_stack])
        max_w = max([img.shape[1] for img in image_stack])

    # pad all images and masks to the maximum size
    for ix, (img, msk) in enumerate(zip(image_stack, mask_stack)):
        if args.flipbook:
            image_stack[ix] = pad_flipbook(img, (max_h, max_w))
            mask_stack[ix] = pad_flipbook(msk, (max_h, max_w))
        else:
            image_stack[ix] = pad_image(img, (max_h, max_w))
            mask_stack[ix] = pad_image(msk, (max_h, max_w))

    # stack into 3d or 4d images
    image_stack = np.stack(image_stack, axis=0).astype(np.uint8)
    mask_stack = np.stack(mask_stack, axis=0)

    io.imsave(os.path.join(args.save_dir, f'{batch_name}_images.tif'), image_stack, check_contrast=False)
    io.imsave(os.path.join(args.save_dir, f'{batch_name}_cs_masks.tif'), mask_stack, check_contrast=False)
