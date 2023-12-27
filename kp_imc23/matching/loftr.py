import pickle
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from pathlib import Path

from kp_imc23.preprocessing.pairs import get_pairs
from kp_imc23.external.superglue.models.matching import Matching
from kp_imc23.external.superglue.models.utils import frame2tensor, make_matching_plot, read_image, get_torch_device
import torch
import kornia as K
import kornia.feature as KF

from kp_imc23.preprocessing.utils import split_images_into_regions

def make_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, conf):
    alpha = 0
    color = cm.jet(conf, alpha=alpha)

    conf_min = conf.min()
    conf_max = conf.max()
    text = [
        f'LoFTR',
        '# Matches (showing/total): {}/{}'.format(len(mkpts0), len(mkpts0)),
    ]
    small_text = [
        f'Showing matches from {0}:{2000}',
        # f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
        # 'Image Pair: {:06}:{:06}'.format(stem0, stem1),
    ]
    # Make the matching plot and save it to a file
    make_matching_plot(
        image0, image1, mkpts0, mkpts1, mkpts0, mkpts1, color,
        text, Path("./result.png"), True, 'Matches', small_text)
        
def scale_to_resized(mkpts0, mkpts1, scale1,scale2):
    ### scale to original im size because we used max_image_size
    # first point
    mkpts0[:, 0] = mkpts0[:, 0] / scale1[0]
    mkpts0[:, 1] = mkpts0[:, 1] / scale1[1]    
    # second point
    mkpts1[:, 0] = mkpts1[:, 0] / scale2[0]
    mkpts1[:, 1] = mkpts1[:, 1] / scale2[1]
    
    return mkpts0, mkpts1

def get_model(weights_path,device):

    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(weights_path)['state_dict'])
    matcher = matcher.to(device).eval()
    return matcher

def extract_features(model,inp0,inp1):
    with torch.inference_mode():
        input_dict = {"image0": inp0,"image1": inp1}
        correspondences = model(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()
    
    return mkpts0, mkpts1, conf

def loft_split_matching(model, image0, image1):
    # Split the images into regions using the specified method
    tiles1, tiles2, offsets = split_images_into_regions(image0, image1)
    # Initialize an empty list for storing the matches
    matches = []

    # Loop through the regions and match them using SuperGlue
        
    for i, (tile1, tile2) in enumerate(zip(tiles1, tiles2)):
        # Convert the images to tensors
        inp0 = frame2tensor(tile1, "cuda")
        inp1 = frame2tensor(tile2, "cuda")
        
        mkpts0, mkpts1, conf = extract_features(model,inp0,inp1)
        
        # Add the offset to the coordinates of the keypoints and matches
        x_offset, y_offset = offsets[i]
        # make_plot(tile1, tile2, mkpts0, mkpts1, mkpts0, mkpts1, conf, True)
        mkpts0[:, 0] += x_offset
        mkpts0[:, 1] += y_offset
        mkpts1[:, 0] += x_offset
        mkpts1[:, 1] += y_offset
        # Append the matches to the list
        matches.append(( mkpts0, mkpts1, conf))

    # Concatenate the matches from all regions
    mkpts0, mkpts1, conf = zip(*matches)
    mkpts0 = np.concatenate(mkpts0, axis=0)
    mkpts1 = np.concatenate(mkpts1, axis=0)
    conf = np.concatenate(conf, axis=0)

    # Make the final plot of the matched images
    # make_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, conf, True)
    return mkpts0, mkpts1, conf


def loftr(images_dir: Path,pairs_path,weights_path,output_dir,resize = [1376,], with_splitting = False):
    device =  get_torch_device()

    pairs = get_pairs(pairs_path)

    model = get_model(weights_path, device)

    keypoints = {}

    for image_0_name,image_1_name in tqdm(pairs, desc=f"Loftr {images_dir.name}", ncols=80):
        
        img0_path = images_dir / image_0_name
        
        img1_path = images_dir / image_1_name

        if(image_0_name not in keypoints):
            keypoints[image_0_name] = {}
        
        # for resize_value in resize:
      
        image0, inp0, scales0 = read_image(img0_path,device,resize,0,True)
        image1, inp1, scales1 = read_image(img1_path,device,resize,0,True)
    

        if(with_splitting):
            mkpts0, mkpts1 = loft_split_matching(model,image0,image1)
        else:
            mkpts0, mkpts1 = extract_features(model,inp0,inp1)
        # mkpts0, mkpts1 = scale_to_resized(mkpts0,mkpts1,scales0,scales1)

        keypoints[image_0_name][image_1_name] = {"keypoints0":mkpts0,"keypoints1":mkpts1}

    with open(output_dir, "wb") as file:
        pickle.dump(keypoints, file)