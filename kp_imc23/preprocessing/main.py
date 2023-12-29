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
# from kp_imc23.external.hloc import extract_features, match_features
from kp_imc23.external.hloc import extract_features, match_features,reconstruction, match_dense

import gc

def preprocess(
    paths: DataPaths,
    image_list,
    args: argparse.Namespace,
    matcher = ["lightglue"], # "lightglue" or "loftr" or "dkm"
    num_pairs: int = 10,
) -> Tuple[Dict[str, Any], bool]:
    """Preprocess images and output rotated images, and computed pairs.

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """

    print(f"Image List: {image_list}")

    # # rotate images
    rotate_images(paths.input_dir_images, image_list, paths.rotated_image_dir, paths.rotation_model_weights)

    # # compute pairs 
    compute_pairs(paths.input_dir_images, image_list, paths.features_retrieval, paths.pairs_path, num_pairs=num_pairs)
    # # # extract important keypoints 
    extract_features.main(
            conf= {
                'output': 'feats-superpoint-n4096-rmax1600',
                'model': {
                    'name': 'superpoint',
                    'nms_radius': 4,
                    'max_keypoints': 4096,
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1600,
                    'resize_force': True,
                },
            },
            image_dir=paths.rotated_image_dir,
            image_list=image_list,
            feature_path=paths.features_path,
        )
    
    matchers_confs = {
        'lightglue': {
            'output': 'matches-superpoint-lightglue',
            'model': {
                'name': 'lightglue',
                'features': 'superpoint',
            }
        },
        'loftr': {
            "output": "matches-loftr",
            "model": {"name": "loftr", "weights": "outdoor"},
            "preprocessing": {"grayscale": True, "resize_max": 1024, "dfactor": 8},  # 1024,
            "max_error": 2,  # max error for assigned keypoints (in px)
            "cell_size": 8,  # size of quantization patch (max 1 kp/patch)
        },
        'dkm': {
            "output": "matches-dkm",
            "model": {"name": "dkm", "weights": "outdoor"},
        }
    }
    
    if "lightglue" in matcher:
        match_features.main(
                conf=matchers_confs["lightglue"],
                pairs=paths.pairs_path,
                features=paths.features_path,
                matches=paths.matches_path, 
        )
    if "loftr" in matcher:
        features, loc_matches = match_dense.main(
            conf=matchers_confs["loftr"],
            pairs=paths.pairs_path,
            image_dir=paths.rotated_image_dir,
            features=paths.features_path,
            matches=paths.matches_path, 
            max_kps=None,
            overwrite=False
        )
    if "dkm" in matcher:
        features, loc_matches = match_dense.main(
            conf=matchers_confs["dkm"],
            pairs=paths.pairs_path,
            image_dir=paths.rotated_image_dir,
            features=paths.features_path,
            matches=paths.matches_path, 
            max_kps=None,
            overwrite=False
        )
    
    
    # concat important keypoints
    # keypoints = concat_keypoints(paths.superglue_keypoints_pickle,paths.loftr_keypoints_pickle)

    # crop images 
    # crop_images(chosen_dir_image, paths.pairs_path, paths.cropped_image_dir,sg_keypoints)

    # build superlist
    # superlist = build_superlist(keypoints)

    # return keypoints, image_dir_used
