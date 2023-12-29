from typing import Any, Dict, Tuple
from kp_imc23.config.paths import DataPaths
import argparse
import os
from .save_matches_keypoints import keypoints_to_out_match_unique_kpts, register_keypoints, register_matches, import_into_colmap, create_submission
from .colmap import COLMAP_mapping, COLMAP_result_analysis
try :
    import pycolmap
except : pass
from kp_imc23.external.hloc import reconstruction
import gc

def database_colmap_run(
    paths: DataPaths,
    image_list,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Save the data into database and run pycolmap on the database.

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """


    # # Keypoints into unique keypoints and list of matches
    # out_match, unique_kpts = keypoints_to_out_match_unique_kpts(keypoints, paths.matches_path)

    # # Save keypoints and matches into database

    # register_keypoints(paths.keypoints_final_path, unique_kpts)
    # register_matches(paths.matches_final_path, out_match)

    # # Import into database COLMAP
    # import_into_colmap(paths.database_path, paths.keypoints_final_path, image_dir_used, paths.matches_final_path)

    # # Match exhaustif of pycolmap for formatting
    # pycolmap.match_exhaustive(paths.database_path)

    # Run COLMAP incremental mapping
    # maps = COLMAP_mapping(paths.colmap_output, paths.database_path, image_dir_used)

    # Results analysis
    # out_results = COLMAP_result_analysis(maps, dataset, scene)

    # if paths.sfm_dir.exists() and not overwrite:
    #     try:
    #         sparse_model = pycolmap.Reconstruction(paths.sfm_dir)
    #         print(f"Sparse model already at {paths.sfm_dir}")
    #         return
    #     except ValueError:
    #         sparse_model = None

    # read images from rotated image dir if rotation wrapper is used
    camera_mode = pycolmap.CameraMode.AUTO
    
    print(f"Using images from {paths.rotated_image_dir}")
    print(f"Using pairs from {paths.pairs_path}")
    print(f"Using features from {paths.features_path}")
    print(f"Using matches from {paths.matches_path}")
    print(f"Using {camera_mode}")

    gc.collect()
    
    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.min_model_size = 3
    mapper_options.min_num_matches = 10
    mapper_options.ba_global_images_ratio = 1.1   #(default: 1.1)
    mapper_options.ba_global_points_ratio = 1.1   #(default: 1.1)
    mapper_options.ba_global_images_freq = 1500   #(default: 500)
    mapper_options.ba_global_points_freq = 350000 #(default: 250000)

    print(image_list)
    sparse_model = reconstruction.main(
        sfm_dir=paths.sfm_dir,
        image_dir=paths.rotated_image_dir,
        image_list=image_list,
        pairs=paths.pairs_path,
        features=paths.features_path,
        matches=paths.matches_path,
        camera_mode=camera_mode,
        verbose=False,
        mapper_options=mapper_options.todict(),
        # skip_geometric_verification=True,
    )

    sparse_model.write(paths.sfm_dir)

    gc.collect()
    # Create submission
    # image_list = os.listdir(image_dir_used)
    # create_submission(out_results, image_list, paths.submission_path,dataset,scene)

    