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
from .utils import concat_keypoints, build_superlist
from kp_imc23.external.hloc import extract_features, match_features

def preprocess(
    paths: DataPaths,
    image_list,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Preprocess images and output rotated images, and computed pairs.

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """

    # # rotate images
    rotate_images(paths.input_dir_images, image_list, paths.rotated_image_dir, paths.rotation_model_weights)

    # # compute pairs 
    compute_pairs(paths.rotated_image_dir, image_list, paths.features_retrieval, paths.pairs_path)
    # # # extract important keypoints 
    extract_features.main(
            conf= {
                'output': 'feats-disk',
                'model': {
                    'name': 'disk',
                    'max_keypoints': 5000,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1600,
                },
            },
            image_dir=paths.rotated_image_dir,
            image_list=image_list,
            feature_path=paths.features_path,
        )
    
    match_features.main(
            conf= {
                'output': 'matches-superpoint-lightglue',
                'model': {
                    'name': 'lightglue',
                    'features': 'superpoint',
                },
            },
            pairs=paths.pairs_path,
            features=paths.features_path,
            matches=paths.matches_path,
        )

    
    
    # concat important keypoints
    # keypoints = concat_keypoints(paths.superglue_keypoints_pickle,paths.loftr_keypoints_pickle)

    # crop images 
    # crop_images(chosen_dir_image, paths.pairs_path, paths.cropped_image_dir,sg_keypoints)

    # build superlist
    # superlist = build_superlist(keypoints)

    # return keypoints, image_dir_used
