import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from kp_imc2023.config.paths import DataPaths
import gdown
import os

from kp_imc2023.preprocessing.main import preprocess

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

    paths = configurate(
        data_dir="./",
        output_dir="./output",
        dataset="heritage", 
        scene="cyprus",
        mode="train"
    )

    # preprocess images
    preprocess(paths,args=None)