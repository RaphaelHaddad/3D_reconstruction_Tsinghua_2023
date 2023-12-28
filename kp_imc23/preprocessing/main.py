import argparse
import os
import pickle

from typing import Any, Dict, Tuple

import numpy as np
from ...kp_imc23.config.paths import DataPaths
from .rotate import rotate_images 
from .pairs import compute_pairs 
from .crop import crop_images 
from ...kp_imc23.matching.loftr import loftr
from ...kp_imc23.matching.superglue import superglue
from .utils import concat_keypoints, build_superlist



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
    image_dir_used = paths.rotated_image_dir
    compute_pairs(image_dir_used, image_list, paths.features_retrieval, paths.pairs_path)

    # # TODO: run in parallel
    # # extract important keypoints 
    superglue(paths.rotated_image_dir,paths.pairs_path, paths.superglue_keypoints_pickle)
    loftr(paths.rotated_image_dir,paths.pairs_path ,paths.loftr_model_weights, paths.loftr_keypoints_pickle)
    
    # concat important keypoints
    keypoints = concat_keypoints(paths.superglue_keypoints_pickle,paths.loftr_keypoints_pickle)

    # crop images 
    chosen_dir_image = paths.input_dir_images
    crop_images(chosen_dir_image, paths.pairs_path, paths.cropped_image_dir,keypoints)

    # build superlist
    superlist = build_superlist(keypoints)

    return keypoints, image_dir_used