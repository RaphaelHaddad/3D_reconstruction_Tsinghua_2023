import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from kp_imc23.config.paths import DataPaths
import gdown
import os

from kp_imc23.preprocessing.main import preprocess
from kp_imc23.matching.main import database_colmap_run

def configurate(data_dir, output_dir, dataset, scene, mode):
    """Configurate paths for the computation results.
    Also downloads weights if they are not here.
    """
    # download weights
    diodModelPath = "./weights/model-vit-ang-loss.h5"
    if not os.path.exists(diodModelPath):
        print("Downloading weights...")
        gdown.download("https://drive.google.com/u/0/uc?id=1sdmPmaDhivdHPfn9M9vAkTbiprbPq94e&export=download", diodModelPath, quiet=False)

    paths = DataPaths(Path(data_dir), Path(output_dir), dataset, scene, mode)
    return paths

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

    