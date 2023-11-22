
from src.external.hloc import (
    extract_features,
    pairs_from_retrieval,
    
)
from pathlib import Path

def compute_pairs(image_dir:Path,img_list,features_retrieval_path,pairs_path) -> None: 
    print("Compute pairs")

    extract_features.main(
        conf=extract_features.confs["netvlad"],
        image_dir=image_dir,
        image_list=img_list,
        feature_path=features_retrieval_path,
    )

    pairs_from_retrieval.main(
        descriptors=features_retrieval_path,
        num_matched=10,
        output=pairs_path,
    )   

    return pairs_path