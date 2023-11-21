import argparse
import os

from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.config.paths import DataPaths
from .rotate import rotate_images 

def preprocess(
    paths: DataPaths,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Preprocess images in input_dir and save them to output_dir.

    Args:s
        input_dir (Path): Directory containing the a folder "images" with the images to preprocess.
        args (argparse.Namespace): Arguments.
    """

    image_list = os.listdir(paths.input_dir_images)

    # rotate image
    rotate_images(paths.input_dir_images, image_list,paths.rotated_image_dir, paths.rotation_model_weights)

