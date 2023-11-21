import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.config.paths import DataPaths

from src.preprocessing.main import preprocess 


if __name__ == '__main__':

    data_dir = Path("./")
    output_dir = Path("./output")
    dataset = "heritage"
    scene = "cyprus"
    mode = "train"

    paths = DataPaths(data_dir,output_dir,dataset,scene,mode)

    # rotate image
    preprocess(paths,args=None)

