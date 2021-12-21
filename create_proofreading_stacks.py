import os
import argparse
import csv
import cv2
import json
import pickle
import pandas as pd
import numpy as np
from glob import glob
from skimage import io
from skimage import transform
from skimage import measure
from tqdm import tqdm
from multiprocessing import Pool
from metrics import *
from helpers import pad_flipbook, pad_image
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
    parser.add_argument('--use-gt', action='store_true',
                        help='Whether to save gold standard annotations over consensus')

    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    print('Loading annotation csv...')
    # necessary to handle overflow of long csv columns
    csv.field_size_limit(256<<12)
    csv.field_size_limit()
    results_df = pd.read_csv(args.annotation_csv)
    print(f'Loading annotation csv with {len(results_df)} rows.')

    # convert the metadata fields to parsable json strings
    results_df['metadata_json'] = [json.loads(q) for q in results_df.metadata]
    results_df['annotations_json'] = [json.loads(q) for q in results_df.annotations]
    results_df['subject_data_json'] = [json.loads(q) for q in results_df.subject_data]

    def parse_annotation(annotation):
        """Helper function for extracting polygons and confidence"""
        confidence = None
        for task in annotation:
            # annotation is task T0
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
            print([task['task'] for task in annotation])
            print(f'No confidence found for image, using 1!')
            confidence = 2


        return polygons, confidence

    # keys are image names and values are attributes
    subject_annotations = {}

    for i, row in results_df.iterrows():
        # subject id and annotated image name
        subject_id = list(row['subject_data_json'].keys())[0]
        key_id = 'Image 2' if args.flipbook else 'Image 0'
        image_name = row['subject_data_json'][subject_id][key_id]

        # subject height and width
        w, h = (args.default_size, args.default_size)

        # reload or create new subject dict
        if image_name in subject_annotations:
            subject_dict = subject_annotations[image_name]
        else:
            subject_dict = {'id': subject_id, 'shape': (w, h),
                            'confidences': [], 'polygons': [],
                            'is_gt': []}

        annotation = row['annotations_json']
        metadata = row['metadata_json']

        polygons, confidence = parse_annotation(annotation)
        subject_dict['is_gt'].append(row['gold_standard'] is True)
        #subject_dict['is_gt'].append(row['user_name'] == 'conradry')
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
        is_gt = subject_dict['is_gt']

        # create instance segmentations from polygons
        masks = []
        using_gt = False
        for gt, polyset in zip(is_gt, subject_dict['polygons']):
            if gt and args.use_gt:
                masks = [poly2segmentation(polyset, image_shape)]
                using_gt = True
                break
            else:
                masks.append(poly2segmentation(polyset, image_shape))

        # create consensus instance segmentation
        instance_scores = mask_aggregation(masks)
        instance_seg = aggregated_instance_segmentation(instance_scores, vote_thr=0.5)

        # compute average precision between each
        # mask and the consensus, this measures strength
        # of the consensus annotations
        if using_gt:
            scores = [10]
        else:
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

    consensus_df = pd.DataFrame(columns=['start', 'end', 'image_name', 'zooniverse_id', 'median_confidence', 'consensus_strength'])

    idx = 0
    for x in consensus_attrs:
        imname, attrs = x
        # convert from imname to stack name
        if args.flipbook:
            stack_fname = '_'.join(imname.split('_')[:-1]) + '.tif'
        else:
            stack_fname = imname.replace('.jpg', '.tiff')

        # SPECIFIC FOR BATCH 1
        #loc_str = stack_fname.split('-LOC-5stack-')[-1]
        #axis = loc_str.split('_')[-3][:1]
        #zindex = loc_str.split('_')[-1].split('.tif')[0].zfill(4)
        #yindex = loc_str.split('_')[-3][1:]
        #xindex = loc_str.split('_')[-2]
        #stack_fname = stack_fname.split('-LOC-')[0] + f'-LOC-5stack-{axis}_{zindex}_{yindex}_{xindex}.tiff'

        #try:
        image = io.imread(os.path.join(args.image_dir, stack_fname))
        #except:
        #    print(f'Failed to load {os.path.join(args.image_dir, stack_fname)}.')
        #    continue

        if args.flipbook:
            h, w = image.shape[1:]
            # add an empty padding slice to flipbook
            image = np.concatenate([image, np.zeros_like(image)[:1]], axis=0)
        else:
            h, w = image.shape


        # resize the mask to the image's h and w
        mask = attrs['seg']

        # cv2 flips height and width
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if args.flipbook:
            flipbook_mask = np.zeros_like(image)
            flipbook_mask[2] = mask # 3rd image in stack is the one annotated
            mask = flipbook_mask

        image_stack.append(image)
        mask_stack.append(mask)

        if args.flipbook:
            start = idx * 6
            end = (idx + 1) * 6 - 1
        else:
            start = idx
            end = idx

        consensus_df = consensus_df.append({
            'start': start, 'end': end,
            'image_name': stack_fname, 'zooniverse_id': attrs['id'],
            'median_confidence': attrs['median_confidence'],
            'consensus_strength': attrs['consensus_strength'],
            'height': h, 'width': w
        }, ignore_index=True)

        idx += 1

    # save the consensus attributes as a list
    batch_name = '-'.join(args.annotation_csv.split('-')[:-1])
    attr_fpath = os.path.join(args.save_dir, f'{batch_name}_consensus_attributes.csv')
    consensus_df.to_csv(attr_fpath, index=False)

    # find dimensions of largest image
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

    if args.flipbook:
        image_stack = np.concatenate(image_stack, axis=0).astype(np.uint8)
        mask_stack = np.concatenate(mask_stack, axis=0)
    else:
        image_stack = np.stack(image_stack, axis=0).astype(np.uint8)
        mask_stack = np.stack(mask_stack, axis=0)

    io.imsave(os.path.join(args.save_dir, f'{batch_name}_images.tif'), image_stack, check_contrast=False)
    io.imsave(os.path.join(args.save_dir, f'{batch_name}_cs_masks.tif'), mask_stack, check_contrast=False)
