import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from kp_imc23.preprocessing.pairs import get_pairs
from kp_imc23.external.superglue.models.matching import Matching
from kp_imc23.external.superglue.models.utils import read_image
import torch
import kornia as K
import kornia.feature as KF

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

def loftr(images_dir: Path,pairs_path,weights_path,output_dir,resize = [1376,]):
    device =  'cpu'
    # device = torch.device('cuda')
    # resize = [[840,], [1024,], [1280,] ]

    pairs = get_pairs(pairs_path)

    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(weights_path)['state_dict'])
    matcher = matcher.to(device).eval()

    keypoints = {}

    for image_0_name,image_1_name in tqdm(pairs, desc=f"Loftr {images_dir.name}", ncols=80):
        
        img0_path = images_dir / image_0_name
        
        img1_path = images_dir / image_1_name

        if(image_0_name not in keypoints):
            keypoints[image_0_name] = {}
        
        # for resize_value in resize:
      
        image0, inp0, scales0 = read_image(img0_path,device,resize,0,True)
        image1, inp1, scales1 = read_image(img1_path,device,resize,0,True)

        with torch.inference_mode():
            input_dict = {"image0": inp0,"image1": inp1}
            correspondences = matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
        # mkpts0, mkpts1 = scale_to_resized(mkpts0,mkpts1,scales0,scales1)

        
        keypoints[image_0_name][image_1_name] = {"keypoints0":mkpts0,"keypoints1":mkpts1}

    with open(output_dir, "wb") as file:
        pickle.dump(keypoints, file)