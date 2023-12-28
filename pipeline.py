import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from kp_imc23.config.paths import DataPaths
import gdown
import os
import wget

from kp_imc23.preprocessing.main import preprocess
from kp_imc23.matching.main import database_colmap_run
from kp_imc23.pipeline import configurate

if __name__ == '__main__':

    # data_dir = Path("./")
    # output_dir = Path("./output")
    # dataset = "heritage"
    # scene = "cyprus"
    # mode = "train"
    dataset, scene = "heritage", "cyprus"
    paths = configurate(
        data_dir="./",
        output_dir="./output",
        dataset=dataset, 
        scene=scene,
        mode="train"
    )

    # preprocess images
    keypoints, image_dir_used = preprocess(paths,args=None)

    # Database
    database_colmap_run(paths, image_dir_used, dataset, scene, keypoints, args=None)

    