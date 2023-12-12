# General utilities
import os
from tqdm import tqdm
from time import time
import gc
import numpy as np
import pandas as pd

# CV/ML
import torch

# 3D reconstruction
import pycolmap
import cv2

from utils import get_image_pairs_shortlist, detect_features, match_features, import_into_colmap, match_loftr, create_camera
import h5py



TYPE_NAME_DICT = {"urban" : ["kyiv-puppet-theater"]}
LOCAL_FEATURE = 'LoFTR'

# Switch to CPU or Cuda if not have a mac torch.device('cpu')
device=torch.device('mps')


src = "path/to/data/image-matching-challenge-2023"
feature_root = os.getcwd()  # Root path
DISK_weights_path = os.path.join(feature_root, "database/weights/outdoor_ot_DISK.ckpt")
loftr_weights_path = os.path.join(feature_root, "database/weights/outdoor_ds_loftr.ckpt")
keypoints_file = os.path.join(feature_root, "version_alpha_1/outputs/analysis/keypoints.h5")
matches_file = os.path.join(feature_root, "version_alpha_1/outputs/analysis/matches.h5")
train_label_path = "path/to/data/image-matching-challenge-2023/train/train_labels.csv"



out_results = {}
timings = {"shortlisting":[],
           "feature_detection": [],
           "feature_matching":[],
           "RANSAC": [],
           "Reconstruction": []}



gc.collect()
datasets = [scene for scene in TYPE_NAME_DICT]

for dataset in datasets:
    print(dataset)
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in TYPE_NAME_DICT[dataset]:
        print(scene)
        img_dir = f'{src}/train/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue
        # Wrap the meaty part in a try-except block.
        try:
            out_results[dataset][scene] = {}
            img_fnames = []
            for filename in os.listdir(img_dir):
                img_fnames.append(f"{src}/train/{dataset}/{scene}/images/{filename}")
            print (f"Got {len(img_fnames)} images")
            feature_dir = f'{feature_root}/featureout/{dataset}_{scene}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
            t=time()
            #### Hardcoding to remove
            img_fnames = img_fnames[:len(img_fnames)//4]
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                  sim_th = 0.7321971, # should be strict
                                  min_pairs =37, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = 30,
                                  device=device)
            t=time() -t 
            timings['shortlisting'].append(t)
            print (f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
            gc.collect()
            t=time()
            if LOCAL_FEATURE != 'LoFTR':
                detect_features(img_fnames, LOCAL_FEATURE,
                                8192,
                                feature_dir=feature_dir,
                                upright=True,
                                device=torch.device('cpu'),
                                resize_small_edge_to=800,
                                DISK_weights_path=DISK_weights_path
                               )
                gc.collect()
                t=time() -t 
                timings['feature_detection'].append(t)
                print(f'Features detected in  {t:.4f} sec')
                t=time()
                idx_matches = match_features(img_fnames, index_pairs, feature_dir=feature_dir,device=torch.device('cpu'))
            else:
                match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, device=device, state_dict_path=loftr_weights_path, resize_to_=(600, 800))
            t=time() -t 
            timings['feature_matching'].append(t)
            print(f'Features matched in  {t:.4f} sec')
            database_path = f'{feature_dir}/colmap.db'
            if os.path.isfile(database_path):
                os.remove(database_path)
            gc.collect()
            import_into_colmap(img_dir, feature_dir=feature_dir,database_path=database_path)
            output_path = f'{feature_dir}/colmap_rec_{LOCAL_FEATURE}'

            t=time()
            pycolmap.match_exhaustive(database_path)   ### Try without it 
            t=time() - t 
            timings['RANSAC'].append(t)
            print(f'RANSAC in  {t:.4f} sec')

            t=time()
            # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 2
            mapper_options.min_num_matches = 1
            mapper_options.ba_local_max_num_iterations = 500
            mapper_options.ba_global_max_num_iterations = 500
            os.makedirs(output_path, exist_ok=True)
            maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, options=mapper_options)
            print(maps)
            for k, im in maps[0].images.items():
                print(im.rotmat())
                np.array(im.tvec)