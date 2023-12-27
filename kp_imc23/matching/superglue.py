import pickle
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from pathlib import Path
from kp_imc23.preprocessing.pairs import get_pairs
from kp_imc23.external.superglue.models.matching import Matching
from kp_imc23.external.superglue.models.utils import frame2tensor, make_matching_plot, read_image, get_torch_device
from kp_imc23.preprocessing.utils import split_images_into_regions, set_torch_device


def get_model(device):
    
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
    return matching

def scale_to_resized(mkpts0, mkpts1, scale1,scale2):
    ### scale to original im size because we used max_image_size
    # first point
    mkpts0[:, 0] = mkpts0[:, 0] / scale1[0]
    mkpts0[:, 1] = mkpts0[:, 1] / scale1[1]    
    # second point
    mkpts1[:, 0] = mkpts1[:, 0] / scale2[0]
    mkpts1[:, 1] = mkpts1[:, 1] / scale2[1]
    
    return mkpts0, mkpts1


def make_plot(model, image0, image1, kpts0, kpts1, mkpts0, mkpts1, conf, valid):
    # Get the color map for the matches
    color = cm.jet(conf[valid])
    # # Define the text for the plot
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]

    # Display extra parameter info.
    k_thresh = model.superpoint.config['keypoint_threshold']
    m_thresh = model.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]

    make_matching_plot(
        image0, image1, mkpts0, mkpts1, mkpts0, mkpts1, color,
        text, Path("./result.png"), True, 'Matches', small_text)
        

def extract_features(model, inp0,inp1):
    # Perform the model.
    pred = model({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > 10
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    return mkpts0, mkpts1, conf,valid, matches, kpts0, kpts1



def extract_features_split_matching(model, image0,image1):
    # Split the images into regions using the specified method
    tiles1, tiles2, offsets = split_images_into_regions(image0, image1)
    # Get the SuperGlue model with the specified type
    # Initialize an empty list for storing the matches
    matches = []

    # Loop through the regions and match them using SuperGlue
        
    for i, (tile1, tile2) in enumerate(zip(tiles1, tiles2)):
        # Convert the images to tensors
        inp1 = frame2tensor(tile1, "cuda")
        inp2 = frame2tensor(tile2, "cuda")

        # Extract the features and matches using SuperGlue
        mkpts0, mkpts1, conf, valid, _, kpts0, kpts1 = extract_features(model, inp1, inp2)
        # Add the offset to the coordinates of the keypoints and matches
        x_offset, y_offset = offsets[i]
        # make_plot(tile1, tile2, mkpts0, mkpts1, mkpts0, mkpts1, conf, True)
        kpts0[:, 0] += x_offset
        kpts0[:, 1] += y_offset
        kpts1[:, 0] += x_offset
        kpts1[:, 1] += y_offset
        mkpts0[:, 0] += x_offset
        mkpts0[:, 1] += y_offset
        mkpts1[:, 0] += x_offset
        mkpts1[:, 1] += y_offset
        # Append the matches to the list
        matches.append((mkpts0, mkpts1, mkpts0, mkpts1, conf, valid))

    # Concatenate the matches from all regions
    kpts0, kpts1, mkpts0, mkpts1, conf, valid = zip(*matches)
    kpts0 = np.concatenate(kpts0, axis=0)
    kpts1 = np.concatenate(kpts1, axis=0)
    mkpts0 = np.concatenate(mkpts0, axis=0)
    mkpts1 = np.concatenate(mkpts1, axis=0)
    conf = np.concatenate(conf, axis=0)
    valid = np.concatenate(valid, axis=0)
    # make_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, conf, valid)

    return mkpts0, mkpts1, conf, valid, matches, kpts0, kpts1
    # Make the final plot of the matched images


def superglue(images_dir: Path,pairs_path,output_dir, resize = [1376,],with_splitting = False):
    device = set_torch_device()
    # device = torch.device('cuda')
    # resize = [[840,], [1024,], [1280,] ]

    pairs = get_pairs(pairs_path)
    model = get_model(device)
    keypoints = {}

    for image_0_name,image_1_name in tqdm(pairs, desc=f"Superglue {images_dir.name}", ncols=80):
        img0_path = images_dir / image_0_name
        
        img1_path = images_dir / image_1_name

        if(image_0_name not in keypoints):
            keypoints[image_0_name] = {}
        
        # for resize_value in resize:
      
        image0, inp0, scales0 = read_image(img0_path,device,resize,0,True)
        image1, inp1, scales1 = read_image(img1_path,device,resize,0,True)


        if(with_splitting):
            mkpts0, mkpts1, _,_, _, _, _ = extract_features_split_matching(img0_path,img1_path)
        else:
            mkpts0, mkpts1, _,_, _, _, _ = extract_features(model,inp0,inp1)
        
        keypoints[image_0_name][image_1_name] = {"keypoints0":mkpts0,"keypoints1":mkpts1}

    with open(output_dir, "wb") as file:
        pickle.dump(keypoints, file)
