import PIL
import os
import argparse
import scipy
import skimage
import numpy as np
from glob import glob
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageSequence
import cv2
import pickle
from os import listdir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', type=str)
    parser.add_argument('consensus_file', type=str)
    parser.add_argument('excel_file', type=str)
    parser.add_argument('img_index', type=int)
    args = parser.parse_args()
    
    imdir = args.imdir
    consensusdir=args.consensus_file
    exceldir=args.excel_file
    overall_image_index=args.img_index
    
chunk_list = os.listdir(imdir)
chunk_list.sort()

def find_chunk_length(f):
  s, e = f.split('images')[-1].split('.tif')[0].split('-')
  return int(e) - int(s)

chunk_lengths = [find_chunk_length(f) for f in chunk_list]
chunk_size=chunk_lengths[0]

file_index = overall_image_index // chunk_size
get_file_name = chunk_list[file_index]

# get specific image
specific_image_index = overall_image_index % chunk_size

image=Image.open(imdir +"/"+ get_file_name)


with open(consensusdir, 'rb') as infile:
    consensus_dict = pickle.load(infile)

for k in list(consensus_dict.keys()):
    del consensus_dict[k]['seg']


consensus_df = {
    'stack_start': [],
    'stack_end': [],
    'image_name': [],
    'Sample UID': [],
    **{k: [] for k in list(consensus_dict.values())[0]}    
}

for index, (image_name, columns) in enumerate(consensus_dict.items()):
    consensus_df['stack_start'].append(index * 6)
    consensus_df['stack_end'].append(index * 6 + 6)
    consensus_df['image_name'].append(image_name)

    if '-ROI-' in image_name:
        dataset_uid = image_name.split('-ROI-')[0]
    elif '-LOC-' in image_name:
        dataset_uid = image_name.split('-LOC-')[0]
    else:
        dataset_uid = image_name[:len('.jpg')]

    if dataset_uid[-3:-1] == 'BV':
        dataset_uid = dataset_uid[:-3]

    if dataset_uid[-3:] == 'inv':
        dataset_uid = dataset_uid[:-3]
    
    consensus_df['Sample UID'].append(dataset_uid)
  
    for k, v in columns.items():
        consensus_df[k].append(v)

consensus_df = pd.DataFrame.from_dict(consensus_df)

metadata_df = pd.read_excel(exceldir, sheet_name='ExternalMetadata')

consensus_df['Sample UID'] = consensus_df['Sample UID'].astype('str')
metadata_df['Sample UID'] = metadata_df['Sample UID'].astype('str')

final_df = consensus_df.merge(metadata_df, how="outer", on=['Sample UID'])

final_df=final_df[final_df['stack_start'] >= 0]

final_df.to_excel(f'{file_index}index_updated_consensus.xlsx')
