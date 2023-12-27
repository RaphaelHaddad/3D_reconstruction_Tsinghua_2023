import argparse
import os

from typing import Any, Dict, Tuple
from kp_imc2023.config.paths import DataPaths
# General utilities
import os
import torch
import gc
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'




from utils import get_image_pairs_shortlist, detect_features, match_features, import_into_colmap, match_loftr, txt_to_index_pairs, config_SG, match_dkmv3 \
, match_sg, remove_files, register_keypoints, register_matches, merge_matches_keypoints



TYPE_NAME_DICT = {"heritage" : ["cyprus"]}
device=torch.device('mps')
src = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/data/image-matching-challenge-2023"
feature_root = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/test_methods/exemple_submission_2"
DISK_weights_path = "test_methods/example_submission/weights/outdoor_ot.ckpt"
loftr_weights_path = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/data/weights/loftr/outdoor_ds.ckpt"
keypoints_file = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/test_methods/exemple_submission_2/featureout/urban_kyiv-puppet-theater/keypoints.h5"
matches_file = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/test_methods/exemple_submission_2/featureout/urban_kyiv-puppet-theater/matches.h5"
train_label_path = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/data/image-matching-challenge-2023/train/train_labels.csv"
pairs_txt_path = "/Users/raphaelhaddad/Documents/Tsinghua/Semester_1/ML/project/data/pairs.txt"

# ###### Image chosen for demonstration purpose ######
out_results = {}

TYPE_NAME_DICT = {"heritage" : ["cyprus"]}
LOCAL_FEATURE = "LoFTR" #LoFTR, DISK, DKMv3, LoFTR_SG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_results = {}
HARD_CODE_PAIR = True
CROPPING = False

gc.collect()

def match(
    paths: DataPaths,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Matches pairs of images and outputs keypoints.

    Though here, the whole matching process will be done with both a detector-free and a detector-based method
    that will be merged (first SP/SG & DKMv3).

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """
    datasets = [scene for scene in TYPE_NAME_DICT]
    
    for dataset in datasets:
        print(dataset)
        if dataset not in out_results:
            out_results[dataset] = {}
        for scene in TYPE_NAME_DICT[dataset]:
            print(scene)
            img_dir = f'{src}/train/{dataset}/{scene}/images'
            if CROPPING :
                img_dir = f'{src}/train/{dataset}/{scene}/images_cropped'
            if not os.path.exists(img_dir):
                continue
            # Wrap the meaty part in a try-except block.
            out_results[dataset][scene] = {}
            img_fnames = []
            if not HARD_CODE_PAIR :
                for filename in os.listdir(img_dir):
                    img_fnames.append(os.path.join(img_dir, filename))
            print(f"Got {len(img_fnames)} images")
            feature_dir = f'{feature_root}/featureout/{dataset}_{scene}_{LOCAL_FEATURE}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
        ### Change fnames to only include images mentionned in the pairs.txt
            if HARD_CODE_PAIR :
                index_pairs = txt_to_index_pairs(img_fnames, pairs_txt_path, src, dataset, scene)
            elif not HARD_CODE_PAIR :
                index_pairs = get_image_pairs_shortlist(img_fnames,
                                    sim_th = 0.7321971, # should be strict
                                    min_pairs = 10,
                                    exhaustive_if_less = 20,
                                    device=device)
            print (f"Got {len(img_fnames)} images")
            print (f'{len(index_pairs)}, pairs to match')
            gc.collect()
            remove_files([f'{feature_dir}/matches_loftr.h5'])
            if LOCAL_FEATURE == "DISK":
                detect_features(img_fnames, LOCAL_FEATURE,
                                    8192,
                                    feature_dir=feature_dir,
                                    upright=True,
                                    device=device,
                                    resize_small_edge_to=800,
                                    DISK_weights_path=DISK_weights_path
                                    )
                gc.collect()
                idx_matches = match_features(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
            if LOCAL_FEATURE == "LoFTR":
                out_match, unique_kpts = match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, \
                            device=device, state_dict_path=loftr_weights_path, resize_to_=(600, 800))
            if LOCAL_FEATURE == "DKMv3":
                out_match, unique_kpts = match_dkmv3(img_fnames, index_pairs, feature_dir=feature_dir, \
                            device=device, resize_to_=(600, 800), th_conf=0.7)
            if LOCAL_FEATURE == "LoFTR_SG":
                out_match0, unique_kpts0 = match_sg(img_fnames, index_pairs, feature_dir, \
                                device=device, resize_to_=(600, 800))
                out_match1, unique_kpts1 = match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, \
                                device=device, state_dict_path=loftr_weights_path, resize_to_=(600, 800))
                out_match, unique_kpts = merge_matches_keypoints(out_match0, unique_kpts0, out_match1, unique_kpts1)
            register_keypoints(feature_dir, out_match, unique_kpts)
            register_matches(feature_dir, out_match, unique_kpts)
            database_path = f'{feature_dir}/colmap.db'
            if os.path.isfile(database_path):
                os.remove(database_path)
            gc.collect()
            import_into_colmap(img_dir, feature_dir=feature_dir,database_path=database_path)
