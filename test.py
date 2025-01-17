import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from .config.paths import DataPaths
import gdown
import os
import wget

from kp_imc23.preprocessing.main import preprocess
from kp_imc23.matching.main import database_colmap_run

def configurate(data_dir, output_dir, dataset, scene, mode):
    """Configurate paths for the computation results.
    Also downloads weights if they are not here.
    """
    # download weights of dioad
    diodModelPath = "./weights/model-vit-ang-loss.h5"
    if not os.path.exists("./weights/"):
        os.makedirs("./weights/")
    if not os.path.exists(diodModelPath):
        print("Downloading weights...")
        gdown.download("https://drive.google.com/u/0/uc?id=1sdmPmaDhivdHPfn9M9vAkTbiprbPq94e&export=download", diodModelPath, quiet=False)

    # download weights of superglue
    superGlueWeights = ["superglue_indoor.pth", "superglue_outdoor.pth", "superpoint_v1.pth"]
    ids = ["1cGa3BG_6guARq37cpkxGt5-w2ZRlh5yn",
        "1gpO6DO4ddJtLh5LdYDvP8uJAM4LYaw-I",
        "1wcAzAhwwn47JG0iXYewdbXq0SHPSTA-Z"
    ]

    for file, id in zip(superGlueWeights, ids):
        path = f"./weights/{file}"
        if not os.path.exists(path):
            print(f"Downloading weights {file}...")
            gdown.download(f"https://drive.google.com/u/0/uc?id={id}&export=download", path, quiet=False)

    # download weights of loftr
    path = f"./weights/outdoor_ds.ckpt"
    if not os.path.exists(path):
        id = "1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY"
        print(f"Downloading weights {path}...")
        gdown.download(f"https://drive.google.com/u/0/uc?id={id}&export=download", path, quiet=False)

    

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
    # keypoints, image_dir_used = preprocess(paths,args=None)
    keypoints = []
    image_dir_used = paths.rotated_image_dir
    # Database
    database_colmap_run(paths, image_dir_used, dataset, scene, keypoints, args=None)

    