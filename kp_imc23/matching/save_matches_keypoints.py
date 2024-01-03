import torch
from collections import defaultdict
import h5py
import numpy as np
from copy import deepcopy
from fastprogress import progress_bar
import os, gc
from .colmap import COLMAPDatabase, add_keypoints, add_matches
from pathlib import Path


def keypoints_to_out_match_unique_kpts(keypoints, matches_path):
    with h5py.File(matches_path, mode='w') as f_match:
        for filename1 in progress_bar(keypoints):
            for filename2 in keypoints[filename1]:
                mkpts0 = keypoints[filename1][filename2]['keypoints0']
                mkpts1 = keypoints[filename1][filename2]['keypoints1']               
                group  = f_match.require_group(filename1)
                group.create_dataset(filename2, data=np.concatenate([mkpts0, mkpts1], axis=1))

    out_match, unique_kpts = find_unique_pixels(matches_path)

    return out_match, unique_kpts



def register_keypoints(keypoints_path, unique_kpts) :
    with h5py.File(keypoints_path, mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1



def register_matches(match_path, out_match) :
    with h5py.File(match_path, mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match





def find_unique_pixels(matches_path) :
    # Let's find unique sg pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(matches_path, mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1] 
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                    ],
                                    axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    return out_match, unique_kpts



def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices



def import_into_colmap(database_path, keypoint_path, image_dir_used, matches_path):

    if os.path.isfile(database_path):
        os.remove(database_path)
        gc.collect()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, keypoint_path, image_dir_used, 'simple-radial', single_camera)
    add_matches(
        db,
        matches_path,
        fname_to_id,
    )

    db.commit()
    return


def arr_to_str(arr):
    return ';'.join(map(str, arr))



# Function to create a submission file.
def create_submission(out_results, image_list, csv_path, dataset, scene):
    # Check if the CSV file already exists
    file_exists = Path(csv_path).exists()

    # Open the CSV file in append mode
    with open(csv_path, 'a') as f:
        # If the file doesn't exist, write the header
        if not file_exists:
            f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')

        for image in image_list:
            path = f'{dataset}/{scene}/images/{image}'
            if path in out_results[dataset][scene]:
                R = out_results[dataset][scene][path]['R'].reshape(-1)
                T = out_results[dataset][scene][path]['t'].reshape(-1)
            else:
                R = np.eye(3).reshape(-1)
                T = np.zeros((3))
            f.write(f'{path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')
        print(f"Submission file written: {csv_path}")
    return out_results


def dumb_submission(image_list, csv_path, dataset, scene) :
    # Check if the CSV file already exists
    file_exists = Path(csv_path).exists()

    # Open the CSV file in append mode
    with open(csv_path, 'a') as f:
        # If the file doesn't exist, write the header
        if not file_exists:
            f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')

        for image in image_list:
            path = f'{dataset}/{scene}/images/{image}'
            R = np.eye(3).reshape(-1)
            T = np.zeros((3))
            f.write(f'{path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')
        print(f"Submission file written: {csv_path}")
