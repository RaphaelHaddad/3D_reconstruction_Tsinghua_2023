import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.external.superglue.models.matching import Matching
from src.external.superglue.models.utils import read_image

from sklearn.cluster import DBSCAN

def crop_image(image, matching_points, eps, min_samples=5):
    # Convert matching points to numpy array
    matching_points = np.array(matching_points)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(matching_points)

    # Find the most dense cluster
    unique_labels, label_counts = np.unique(clustering.labels_, return_counts=True)
#     print(unique_labels)
#     print(label_counts)
    most_dense_cluster_label = unique_labels[np.argmax(label_counts)]
#     print(unique_labels)
    # Get the matching points in the most dense cluster
    cluster_points = matching_points[clustering.labels_ == most_dense_cluster_label]
#     print(cluster_points)
    # Calculate the minimum and maximum coordinates in the cluster
    min_x = int(np.min(cluster_points[:, 0]))
    max_x = int(np.max(cluster_points[:, 0]))
    min_y = int(np.min(cluster_points[:, 1]))
    max_y = int(np.max(cluster_points[:, 1]))
    # Crop the image based on the cluster coordinates
    cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]

    return cropped_image,min_x,min_y

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
    
def extract_features_superglue(config, inp0,inp1,device):
    matching = Matching(config).eval().to(device)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > 10
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    return mkpts0, mkpts1,kpts0,kpts1,conf,valid, matches

def scale_to_resized(mkpts0, mkpts1, scale1,scale2):
    ### scale to original im size because we used max_image_size
    # first point
    mkpts0[:, 0] = mkpts0[:, 0] / scale1[0]
    mkpts0[:, 1] = mkpts0[:, 1] / scale1[1]    
    # second point
    mkpts1[:, 0] = mkpts1[:, 0] / scale2[0]
    mkpts1[:, 1] = mkpts1[:, 1] / scale2[1]
    
    return mkpts0, mkpts1

def crop_images(images_dir: Path,pairs_path,output_dir):
    matches_per_image = {}

    device =  'cpu'
    resize = [1600, ]
    resize_float = True

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
            }
        }
    
    for image_0_name,image_1_name in tqdm(pairs, desc=f"Cropping {images_dir.name}", ncols=80):
        img0_path = images_dir / image_0_name
        
        img1_path = images_dir / image_1_name
        
        image0, inp0, scales0 = read_image(img0_path,device,resize,0,resize_float)
        image1, inp1, scales1 = read_image(img1_path,device,resize,0,resize_float)

        matches0, matches1,kpts0,kpts1,conf,valid, matches = extract_features_superglue(config,inp0,inp1,device)
        
        # mkpts11,mkpts22 = scale_to_resized(matches0, matches1, scales0,scales1)
        # mkpts1=mkpts11
        # mkpts2=mkpts22
        
        if( image_0_name not in matches_per_image or len(matches0) > matches_per_image[image_0_name]):
            print(f"Biggest matches found: {len(matches0)}")
            matches_per_image[image_0_name] = len(matches0)

            print(f"Crop image: {image_0_name}")
            crop_image0,_,_ = crop_image(image0, matches0, 20)
            cv2.imwrite(str(output_dir / image_0_name),crop_image0 )