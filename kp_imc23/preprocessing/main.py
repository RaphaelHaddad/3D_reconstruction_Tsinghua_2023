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
from kp_imc23.external.hloc import extract_features, match_features,reconstruction

import gc
import pycolmap

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

    # # rotate images
    # rotate_images(paths.input_dir_images, image_list, paths.rotated_image_dir, paths.rotation_model_weights)

    # # compute pairs 
    image_dir_used = paths.input_dir_images
    # compute_pairs(image_dir_used, image_list, paths.features_retrieval, paths.pairs_path)

    # # # TODO: run in parallel
    # # # extract important keypoints 
    # sg_keypoints = superglue(paths.rotated_image_dir,paths.pairs_path, paths.superglue_keypoints_pickle)
    # loftr(paths.rotated_image_dir,paths.pairs_path ,paths.loftr_model_weights, paths.loftr_keypoints_pickle)
    extract_features.main(
            conf= {
                'output': 'feats-superpoint-n4096-rmax1600',
                'model': {
                    'name': 'superpoint',
                    'nms_radius': 3,
                    'max_keypoints': 4096,
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1600,
                    'resize_force': True,
                },
            },
            image_dir=paths.input_dir_images,
            image_list=image_list,
            feature_path=paths.features_path,
        )
    
    match_features.main(
            conf= {
                'output': 'matches-superglue-it5',
                'model': {
                    'name': 'superglue',
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 5,
                },
            },
            image_dir=paths.input_dir_images,
            image_list=image_list,
            feature_path=paths.matches_path,
        )

    # if paths.sfm_dir.exists() and not overwrite:
    #     try:
    #         sparse_model = pycolmap.Reconstruction(paths.sfm_dir)
    #         print(f"Sparse model already at {paths.sfm_dir}")
    #         return
    #     except ValueError:
    #         sparse_model = None

    # read images from rotated image dir if rotation wrapper is used
    camera_mode = pycolmap.CameraMode.AUTO
    
    print(f"Using images from {paths.input_dir_images}")
    print(f"Using pairs from {paths.pairs_path}")
    print(f"Using features from {paths.features_path}")
    print(f"Using matches from {paths.matches_path}")
    print(f"Using {camera_mode}")

    gc.collect()
    
    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.min_model_size = 6
    mapper_options.min_num_matches = 10

    sparse_model = reconstruction.main(
        sfm_dir=paths.sfm_dir,
        image_dir=paths.input_dir_images,
        image_list=image_list,
        pairs=paths.pairs_path,
        features=paths.features_path,
        matches=paths.matches_path,
        camera_mode=camera_mode,
        verbose=False,
        reference_model=None,
        mapper_options=mapper_options.todict(),
        # skip_geometric_verification=True,
    )

    sparse_model.write(paths.sfm_dir)

    gc.collect()
    # concat important keypoints
    # keypoints = concat_keypoints(paths.superglue_keypoints_pickle,paths.loftr_keypoints_pickle)

    # crop images 
    chosen_dir_image = paths.input_dir_images
    # crop_images(chosen_dir_image, paths.pairs_path, paths.cropped_image_dir,sg_keypoints)

    # build superlist
    # superlist = build_superlist(keypoints)

    # return keypoints, image_dir_used
