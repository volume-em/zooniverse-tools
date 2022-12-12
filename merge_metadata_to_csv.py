"""
Merges consensus creation metadata with dataset metadata.
This is useful to analyze if the quality of user generated
annotations correlates with image metadata. The expected
metadata_excel should like something like cem_mitolab_metadata.xlsx file
available on EMPIAR: https://www.ebi.ac.uk/empiar/EMPIAR-11037/
"""

import os
import argparse
import numpy as np
from glob import glob
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('consensus_csv', type=str)
    parser.add_argument('metadata_excel', type=str)
    args = parser.parse_args()
    
    consensus_csv = args.consensus_csv
    metadata_excel = args.metadata_excel
    
    consensus_df = pd.read_csv(consensus_csv)
    external_metadata_df = pd.read_excel(metadata_excel, sheet_name='ExternalMetadata')
    internal_metadata_df = pd.read_excel(metadata_excel, sheet_name='InternalMetadata')
    # drop CMM directory column if exists
    if 'CMM Directory' in internal_metadata_df:
        internal_metadata_df = internal_metadata_df.drop(columns=['CMM Directory'])

    # concat internal and external metadata sheets
    metadata_df = pd.concat([internal_metadata_df, external_metadata_df])

    # convert from image name to equivalent Sample UID
    # as it would appear in the metadata spreadsheet
    sample_uids = []
    for image_name in consensus_df['image_name'].values:
        if '-ROI-' in image_name:
            dataset_uid = image_name.split('-ROI-')[0]
        elif '-LOC-' in image_name:
            dataset_uid = image_name.split('-LOC-')[0]
        elif image_name.endswith('.jpg'):
            dataset_uid = image_name[:len('.jpg')]
        elif image_name.endswith('.tiff'):
            dataset_uid = image_name[:len('.tiff')]
        else:
            raise Exception(f'Failed to extract sample uid from {image_name}')
        
        sample_uids.append(dataset_uid)

    consensus_df['Sample UID'] = np.array(sample_uids)

    # make sure both Sample UID columns are string type
    consensus_df['Sample UID'] = consensus_df['Sample UID'].astype('str')
    metadata_df['Sample UID'] = metadata_df['Sample UID'].astype('str')

    merged_df = consensus_df.merge(metadata_df, how="outer", on=['Sample UID'])
    merged_df = merged_df[merged_df['start'] >= 0]
    merged_df = merged_df.sort_values(by='start')

    savedir = os.path.dirname(consensus_csv)
    consensus_csv_name = os.path.basename(consensus_csv)[:-len('.csv')]
    merged_df.to_csv(os.path.join(savedir, f'{consensus_csv_name}_with_metadata.csv'), index=False)