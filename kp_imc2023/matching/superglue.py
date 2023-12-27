import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from kp_imc2023.preprocessing.pairs import get_pairs
from kp_imc2023.external.superglue.models.matching import Matching
from kp_imc2023.external.superglue.models.utils import read_image

def extract_features_superglue(matching, config, inp0,inp1,device):
    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > 10
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    return mkpts0, mkpts1, conf,valid, matches

def scale_to_resized(mkpts0, mkpts1, scale1,scale2):
    ### scale to original im size because we used max_image_size
    # first point
    mkpts0[:, 0] = mkpts0[:, 0] / scale1[0]
    mkpts0[:, 1] = mkpts0[:, 1] / scale1[1]    
    # second point
    mkpts1[:, 0] = mkpts1[:, 0] / scale2[0]
    mkpts1[:, 1] = mkpts1[:, 1] / scale2[1]
    
    return mkpts0, mkpts1

def superglue(images_dir: Path,pairs_path,output_dir, resize = [1376,]):
    device =  'cpu'
    # device = torch.device('cuda')
    # resize = [[840,], [1024,], [1280,] ]

    pairs = get_pairs(pairs_path)

    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        },
    }
      
    matching = Matching(config).eval().to(device)
    keypoints = {}

    for image_0_name,image_1_name in tqdm(pairs, desc=f"Superglue {images_dir.name}", ncols=80):
        img0_path = images_dir / image_0_name
        
        img1_path = images_dir / image_1_name

        if(image_0_name not in keypoints):
            keypoints[image_0_name] = {}
        
        # for resize_value in resize:
      
        image0, inp0, scales0 = read_image(img0_path,device,resize,0,True)
        image1, inp1, scales1 = read_image(img1_path,device,resize,0,True)

        mkpts0, mkpts1, conf, valid, matches = extract_features_superglue(matching, config,inp0,inp1,device)
        
        keypoints[image_0_name][image_1_name] = {"keypoints0":mkpts0,"keypoints1":mkpts1}

    with open(output_dir, "wb") as file:
        pickle.dump(keypoints, file)