import argparse
import os
import pickle

from typing import Any, Dict, Tuple

import numpy as np
from kp_imc23.config.paths import DataPaths
from .rotate import rotate_images 
from .pairs import compute_pairs 
from .crop import crop_images 
from kp_imc23.matching.loftr import loftr
from kp_imc23.matching.superglue import superglue

def concat_keypoints(keypoints1_pickle,keypoints2_pickle):
    keypoints1 = None
    keypoints2 = None

    if(keypoints1_pickle):
        with open(keypoints1_pickle, "rb") as file:
            keypoints1 = pickle.load(file)

    if(keypoints2_pickle):
        with open(keypoints2_pickle, "rb") as file:
            keypoints2 = pickle.load(file)


    concatenated = {}

    for image,v in keypoints1:
        for image_pair,pair_keypoints in v:
            kp0 = pair_keypoints["keypoints0"]
            kp1 = pair_keypoints["keypoints0"]

            kp00 = keypoints2[image][image_pair]["keypoints0"]
            kp11 = keypoints2[image][image_pair]["keypoints0"]

            concatenated[image][image_pair]["keypoints0"] = np.vstack((kp0,kp00)) 
            concatenated[image][image_pair]["keypoints1"] = np.vstack((kp1,kp11)) 
    
    return concatenated

def build_superlist(keypoints):
    superlist = {} 
    temp_counter = {}

    def get_super_pair(latest_super_pair):
        super_pair = None
        for image_pair,pair_keypoints in keypoints[latest_super_pair].items():
            kp = pair_keypoints["keypoints0"]
            if(image_pair not in superlist and (latest_super_pair not in superlist or len(kp) > temp_counter[latest_super_pair])):
                temp_counter[latest_super_pair] = len(kp)
                superlist[latest_super_pair] = image_pair
                super_pair = image_pair
        return super_pair
    
    latest_super_pair = list(keypoints.keys())[0]

    while(latest_super_pair is not None):
        next_super_pair = get_super_pair(latest_super_pair)
        superlist[latest_super_pair] = next_super_pair
        latest_super_pair = next_super_pair

    return superlist

def preprocess(
    paths: DataPaths,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Preprocess images and output rotated images, and computed pairs.

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """

    image_list = os.listdir(paths.input_dir_images)

    # rotate images
    rotate_images(paths.input_dir_images, image_list, paths.rotated_image_dir, paths.rotation_model_weights)

    # compute pairs 
    compute_pairs(paths.rotated_image_dir, image_list, paths.features_retrieval, paths.pairs_path)

    # TODO: run in parallel
    # extract important keypoints 
    superglue(paths.rotated_image_dir,paths.pairs_path, paths.superglue_keypoints_pickle)
    loftr(paths.rotated_image_dir,paths.pairs_path, paths.superglue_keypoints_pickle)
    
    # concat important keypoints
    keypoints = concat_keypoints(paths.superglue_keypoints_pickle,paths.loftr_keypoints_pickle)

    # crop images 
    crop_images(paths.rotated_image_dir, paths.pairs_path, paths.cropped_image_dir,keypoints)

    # build superlist
    superlist = build_superlist(keypoints)