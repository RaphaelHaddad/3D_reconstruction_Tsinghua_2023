import pickle
import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from pathlib import Path

from kp_imc2023.external.superglue.models.utils import read_image
from .pairs import get_pairs
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

def crop_images(images_dir: Path,pairs_path,output_dir,keypoints,resize = [1376,]):
    matches_per_image = {}

    pairs = get_pairs(pairs_path)
 

    for image_0_name,image_1_name in tqdm(pairs, desc=f"Cropping {images_dir.name}", ncols=80):
        img0_path = images_dir / image_0_name
        
        img1_path = images_dir / image_1_name

        image0, inp0, scales0 = read_image(img0_path,"cpu",resize,0,True)
        image1, inp1, scales1 = read_image(img1_path,"cpu",resize,0,True)

        current = keypoints[image_0_name][image_1_name]["keypoints0"]

        if( image_0_name not in matches_per_image or len(current) > matches_per_image[image_0_name]):
            print(f"Biggest matches found: {len(current)}")
            matches_per_image[image_0_name] = len(current)

            print(f"Crop image: {image_0_name}")
            crop_image0,_,_ = crop_image(image0, current, 20)
            cv2.imwrite(str(output_dir / image_0_name), crop_image0)
