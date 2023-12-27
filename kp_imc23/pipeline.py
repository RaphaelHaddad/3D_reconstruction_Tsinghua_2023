import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from kp_imc23.config.paths import DataPaths
import gdown
import os

from kp_imc23.preprocessing.main import preprocess

def configurate(data_dir, output_dir, dataset, scene, mode):
    """Configurate paths for the computation results.
    Also downloads weights if they are not here.
    """
    # download weights of dioad
    diodModelPath = "./weights/model-vit-ang-loss.h5"
    if not os.path.exists(diodModelPath):
        print("Downloading weights...")
        gdown.download("https://drive.google.com/u/0/uc?id=1sdmPmaDhivdHPfn9M9vAkTbiprbPq94e&export=download", diodModelPath, quiet=False)

    # download weights of superglue
    superGlueWeights = ["superglue_indoor.pth", "superglue_outdoor.pth", "superglue_v1.pth"]
    ids = ["1cGa3BG_6guARq37cpkxGt5-w2ZRlh5yn",
        "1gpO6DO4ddJtLh5LdYDvP8uJAM4LYaw-I",
        "1wcAzAhwwn47JG0iXYewdbXq0SHPSTA-Z"
    ]

    if not os.path.exists("kp_imc23/external/superglue/models/weights/"):
        os.makedirs("kp_imc23/external/superglue/models/weights/")
    for file, id in zip(superGlueWeights, ids):
        path = f"./kp_imc23/external/superglue/models/weights/{file}"
        if not os.path.exists(path):
            print(f"Downloading weights {file}...")
            gdown.download(f"https://drive.google.com/u/0/uc?id={id}&export=download", path, quiet=False)

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