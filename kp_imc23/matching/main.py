from typing import Any, Dict, Tuple
from kp_imc23.config.paths import DataPaths
import argparse
import os
from .save_matches_keypoints import keypoints_to_out_match_unique_kpts, register_keypoints, register_matches, import_into_colmap
from .colmap import COLMAP_mapping, COLMAP_result_analysis
import pycolmap


def database_colmap_run(
    paths: DataPaths,
    dataset: str,
    scene: str,
    keypoints : Dict[str,Dict[Dict,Dict]],
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Save the data into database and run pycolmap on the database.

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """


    # Keypoints into unique keypoints and list of matches
    out_match, unique_kpts = keypoints_to_out_match_unique_kpts(keypoints, paths.matches_path)

    # Save keypoints and matches into database

    register_keypoints(paths.keypoints_final_path, unique_kpts)
    register_matches(paths.matches_final_path, out_match)

    # Import into database COLMAP
    image_dir_used = paths.input_dir_images
    import_into_colmap(paths.database_path, paths.keypoints_final_path, image_dir_used, paths.matches_final_path)

    # Match exhaustif of pycolmap for formatting
    pycolmap.match_exhaustive(paths.database_path)

    # Run COLMAP incremental mapping
    maps = COLMAP_mapping(paths.colmap_output, paths.database_path, image_dir_used)

    # Results analysis
    out_results = COLMAP_result_analysis(maps, dataset, scene)

    