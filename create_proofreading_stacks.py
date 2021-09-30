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
from helpers import pad_flipbook
from aggregation import mask_aggregation, aggregated_instance_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_csv', type=str, 
                        help='Path to zooniverse classification export csv')
    parser.add_argument('image_dir', type=str, 
                        help='Directory from which to load images')
    parser.add_argument('save_dir', type=str, 
                        help='Directory in which to save consensus masks')
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
                
        return polygons, confidence
    
    # keys are image names and values are attributes
    subject_annotations = {}

    for i, row in results_df.iterrows():
        # subject id and annotated image name
        subject_id = list(row['subject_data_json'].keys())[0]
        image_name = row['subject_data_json'][subject_id]['Image 2']
        
        # subject height and width
        image_dims = row['metadata_json']['subject_dimensions'][2]
        
        if image_dims is None:
            print(f'Missing image dimensions for {image_name} using (480, 480).')
            w, h = (480, 480)
        else:
            w, h = image_dims['naturalWidth'], image_dims['naturalHeight']
        
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
        # mask and the consensus, this measure strength
        # of the consensus annotations
        if using_gt:
            scores = [10]
        else:
            scores = [average_precision(instance_seg, mask, 0.50, False)[0] for mask in masks]

        avg_confidence = np.mean(subject_dict['confidences'])
        consensus_strength = np.mean(scores)
        consensus_attrs[imname] = {
            'id': subject_id, 'avg_confidence': avg_confidence, 
            'consensus_strength': consensus_strength,
            'seg': instance_seg
        }
        
    # sort the stacks from lowest to highest consensus strength
    consensus_attrs = sorted(consensus_attrs.items(), key=lambda x: x[1]['consensus_strength'])

    for i, (k, v) in enumerate(consensus_attrs):
    	print(i, i * 6, k, v['consensus_strength'])

    """
    consensus_attrs = {k: v for k,v in consensus_attrs}
    

    image_stack = []
    mask_stack = []
    for imname, attrs in consensus_attrs.items():
        # convert from imname to stack name
        stack_fname = '_'.join(imname.split('_')[:-1]) + '.tif'
        
        try:
            flipbook = io.imread(os.path.join(args.image_dir, stack_fname))
        except:
            print(f'Failed to load {os.path.join(args.image_dir, stack_fname)}.')
            continue
            
        # add an empty padding slice to flipbook
        flipbook = np.concatenate([flipbook, np.zeros_like(flipbook)[:1]], axis=0)
        
        # resize the mask to the flipbook's h and w
        mask = attrs['seg']
        
        # cv2 flips height and width
        mask = cv2.resize(mask, tuple(flipbook.shape[1:][::-1]), interpolation=cv2.INTER_NEAREST)
        
        flipbook_mask = np.zeros_like(flipbook)
        flipbook_mask[2] = mask # 3rd image in stack is the one annotated
        
        image_stack.append(flipbook)
        mask_stack.append(flipbook_mask)
        
    for ix, (name, v) in enumerate(consensus_attrs.items()):
        print(ix, name, v['consensus_strength'])
        
    # find dimensions of largest image
    max_h = max(img.shape[1] for img in image_stack)
    max_w = max(img.shape[2] for img in image_stack)
    
    # pad all images and masks to the maximum size
    for ix, (img, msk) in enumerate(zip(image_stack, mask_stack)):
        image_stack[ix] = pad_flipbook(img, (max_h, max_w))
        mask_stack[ix] = pad_flipbook(msk, (max_h, max_w))
        
    image_stack = np.concatenate(image_stack, axis=0).astype(np.uint8)
    mask_stack = np.concatenate(mask_stack, axis=0)
    
    # save the consensus attributes and image and mask stacks
    attr_fpath = os.path.join(args.save_dir, 'consensus_attributes.pkl')
    with open(attr_fpath, 'wb') as handle:
        pickle.dump(consensus_attrs, handle)
    
    """
    #batch_name = '-'.join(args.annotation_csv.split('-')[:-1])
    #io.imsave(os.path.join(args.save_dir, f'{batch_name}_images.tif'), image_stack, check_contrast=False)
    #io.imsave(os.path.join(args.save_dir, f'{batch_name}_cs_masks.tif'), mask_stack, check_contrast=False)
