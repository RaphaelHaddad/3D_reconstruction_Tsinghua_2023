
from kp_imc23.external.hloc import (
    extract_features,
    pairs_from_retrieval,
    
)
from pathlib import Path

def get_pairs(path):
    with open(path, "r") as f:
        # initialize an empty list to store the pairs
        pairs = []
        # loop through each line of the file
        for line in f:
            # split the line by whitespace and assign to a list
            pair = line.split()
            # append the list to another list
            pairs.append(pair)

        # print the list of pairs
        return pairs
    
def compute_pairs(image_dir:Path,img_list,features_retrieval_path,pairs_path, num_pairs=10) -> None: 
    print("Compute pairs")

    extract_features.main(
        conf=extract_features.confs["eigenplaces"],
        image_dir=image_dir,
        image_list=img_list,
        feature_path=features_retrieval_path,
    )

    pairs_from_retrieval.main(
        descriptors=features_retrieval_path,
        num_matched=num_pairs,
        output=pairs_path,
    )   

    return pairs_path