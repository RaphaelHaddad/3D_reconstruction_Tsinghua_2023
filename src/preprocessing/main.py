import argparse
import os

from typing import Any, Dict, Tuple
from src.config.paths import DataPaths
from .rotate import rotate_images 
from .pairs import compute_pairs 
from .crop import crop_images 

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

    # crop images 
    crop_images(paths.rotated_image_dir, paths.pairs_path, paths.cropped_image_dir)

