""" Process raw pcd files and save as rooms and blocks as npy files.

Author: Umamaheswaran Raman Kumar 2024
"""

import os
import glob
import numpy as np
import pandas as pd
import argparse
import pickle
import yaml
from itertools import  combinations
from pyntcloud import PyntCloud

from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file path')
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    META_PATH = config['DATA']['META_FILE']
    DATA_PATH = config['DATA']['RAW_DIR']
    DST_PATH = config['DATA']['PROCESSED_DIR']
    
    # Create train, val and test folders
    if len(config['TRAIN']['HOUSES']) > 0:
        TRAIN_ROOM_PATH = os.path.join(DST_PATH, 'train', 'rooms')
        if not os.path.exists(TRAIN_ROOM_PATH): os.makedirs(TRAIN_ROOM_PATH)
        TRAIN_BLOCK_PATH = os.path.join(DST_PATH, 'train', 'blocks')
        if not os.path.exists(TRAIN_BLOCK_PATH): os.makedirs(TRAIN_BLOCK_PATH)
    if len(config['VAL']['HOUSES']) > 0:
        VAL_ROOM_PATH = os.path.join(DST_PATH, 'val', 'rooms')
        if not os.path.exists(VAL_ROOM_PATH): os.makedirs(VAL_ROOM_PATH)
        VAL_BLOCK_PATH = os.path.join(DST_PATH, 'val', 'blocks')
        if not os.path.exists(VAL_BLOCK_PATH): os.makedirs(VAL_BLOCK_PATH)
    if len(config['TEST']['HOUSES']) > 0:
        TEST_ROOM_PATH = os.path.join(DST_PATH, 'test', 'rooms')
        if not os.path.exists(TEST_ROOM_PATH): os.makedirs(TEST_ROOM_PATH)
        TEST_BLOCK_PATH = os.path.join(DST_PATH, 'test', 'blocks')
        if not os.path.exists(TEST_BLOCK_PATH): os.makedirs(TEST_BLOCK_PATH)
    
    # Create meta data file
    DST_META_PATH = os.path.join(DST_PATH, 'meta')
    if not os.path.exists(DST_META_PATH): os.makedirs(DST_META_PATH)    
    PKL_MAP_FILE = os.path.join(DST_META_PATH, 'map.pkl')

    # Metadata file mapping
    CLASS_LABELS = list([x.rstrip().split()[0]
                   for x in open(META_PATH)])
    COLOR2LABEL = {tuple(map(int, x.rstrip().split()[1:4])): x.rstrip().split()[0]
                   for x in open(META_PATH)}
    LABEL2CLASS = {cls: i for i, cls in enumerate(CLASS_LABELS)}

    CLASS2COLOR = {}
    for color, label in COLOR2LABEL.items():
        class_val = LABEL2CLASS[label]
        CLASS2COLOR[class_val] = color

    with open(PKL_MAP_FILE, 'wb') as f:
        pickle.dump([CLASS_LABELS, LABEL2CLASS, CLASS2COLOR], f, pickle.HIGHEST_PROTOCOL)
    
    # Process house folders
    for folder in os.listdir(DATA_PATH):
        print('\nProcessing folder : ', folder)

        # Process rooms
        for file_path in glob.glob(DATA_PATH+'/'+folder+'/*.pcd'):
            in_filename = os.path.split(file_path)[1]
            print(in_filename)
            f_split = in_filename.split('.')
            f_split = [f for f in f_split if f not in ('obj', 'groundtruth', 'pcd')]
            out_filename = folder+'_floor'+f_split[0] + '_'+''.join(f_split[1:])+'.npy'
            if folder in config['TRAIN']['HOUSES']:
                room_out_filepath = os.path.join(TRAIN_ROOM_PATH, out_filename)
            elif folder in config['VAL']['HOUSES']:
                room_out_filepath = os.path.join(VAL_ROOM_PATH, out_filename)
            elif folder in config['TEST']['HOUSES']:
                room_out_filepath = os.path.join(TEST_ROOM_PATH, out_filename)
            else:
                continue
            point_cloud = PyntCloud.from_file(file_path)

            # Add GT labels
            points = pd.DataFrame(point_cloud.points[['x','y','z','red','green','blue']])
            points = points.assign(label=-1)
            for (r,g,b), label in COLOR2LABEL.items():
                class_value = LABEL2CLASS[label]
                points.loc[(points['red']==r) & (points['green']==g) & (points['blue']==b), 
                        'label'] = class_value
            points = points[['x','y','z','label']]

            # Transform points from cms to meters and save as npy files
            points['x'] = points['x'].div(100)
            points['y'] = points['y'].div(100)
            points['z'] = points['z'].div(100)
            points = points.to_numpy()
            np.save(room_out_filepath, points)
    
            # Split rooms to blocks
            if folder in config['TRAIN']['HOUSES']:
                blocks_list = room2blocks(points, block_size=config['BLOCK']['SIZE'], 
                                                stride=config['BLOCK']['STRIDE'], 
                                                min_npts=config['BLOCK']['MIN_NPTS'])
                block_out_folderpath = TRAIN_BLOCK_PATH
            elif folder in config['VAL']['HOUSES']:
                blocks_list = room2blocksamples(points, block_size=config['BLOCK']['SIZE'], 
                                                stride=config['BLOCK']['STRIDE'], 
                                                min_npts=config['BLOCK']['MIN_NPTS'],
                                                sample_num_point=config['BLOCK']['NPTS'])
                block_out_folderpath = VAL_BLOCK_PATH
            elif folder in config['TEST']['HOUSES']:
                blocks_list = room2blocksamples(points, block_size=config['BLOCK']['SIZE'], 
                                                stride=config['BLOCK']['STRIDE'], 
                                                min_npts=config['BLOCK']['MIN_NPTS'],
                                                sample_num_point=config['BLOCK']['NPTS'])
                block_out_folderpath = TEST_BLOCK_PATH
            print('{0} is split into {1} blocks.'.format(out_filename, len(blocks_list)))

            # Save blocks
            for i, block_data in enumerate(blocks_list):
                block_filename = out_filename[:-4] + '_block_' + str(i) + '.npy'
                np.save(os.path.join(block_out_folderpath, block_filename), block_data)

                # === End for loop classes ===
            # === End for loop blocks ===
        # === End for loop rooms ===
    # === End for loop houses ===

