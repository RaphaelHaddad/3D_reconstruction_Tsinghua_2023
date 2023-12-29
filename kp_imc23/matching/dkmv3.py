import pickle
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ...kp_imc23.preprocessing.pairs import get_pairs
from ...kp_imc23.external.superglue.models.matching import Matching
from ...kp_imc23.external.superglue.models.utils import frame2tensor, make_matching_plot, read_image, get_torch_device
import torch
import kornia as K
import kornia.feature as KF

from ...kp_imc23.preprocessing.utils import split_images_into_regions
from ...kp_imc23.external.dkmv3.dkm import DKMv3_outdoor
import h5py
import cv2
from PIL import Image
from fastprogress import progress_bar
from collections import defaultdict, OrderedDict
from copy import deepcopy

def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices

def dkmv3(images_dir: Path, pairs_path, weights_path, output_dir,
                        feature_dir = '.featureout_loftr',\
                        device=torch.device('cuda'),\
                        resize_to_=(600, 800), th_conf=0.7, min_matches=10) :

    pairs = get_pairs(pairs_path)

    matcher = DKMv3_outdoor(device=device)

    # Optimisation for less dense matches and less computations
    matcher.sample_thresh = 0.2
    matcher.w_resized = 864
    matcher.h_resized = 1152
    matcher.upsample_preds = False

    keypoints = {}

    for name1,name2 in tqdm(pairs, desc=f"DKMv3 {images_dir.name}", ncols=80):
        fname1, fname2 = str(images_dir/name1), str(images_dir/name2)

        if(name1 not in keypoints):
            keypoints[name1] = {}
        
        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)

        img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).resize(resize_to_)
        img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).resize(resize_to_)

        dense_matches, dense_certainty = matcher.match(img1PIL, img2PIL)
        dense_certainty = dense_certainty.sqrt()
        dense_matches = dense_matches.reshape((-1, 4))
        dense_certainty = dense_certainty.reshape((-1,))

        # drop low confidence pairs
        dense_matches = dense_matches[ dense_certainty >= th_conf, :]
        dense_certainty = dense_certainty[ dense_certainty >= th_conf]

        mkpts0 = dense_matches[:, :2].cpu().numpy()
        mkpts1 = dense_matches[:, 2:].cpu().numpy()

        h, w, c = img1.shape
        mkpts0[:, 0] = ((mkpts0[:, 0] + 1)/2) * w
        mkpts0[:, 1] = ((mkpts0[:, 1] + 1)/2) * h

        h, w, c = img2.shape
        mkpts1[:, 0] = ((mkpts1[:, 0] + 1)/2) * w
        mkpts1[:, 1] = ((mkpts1[:, 1] + 1)/2) * h

        mkpts0 = mkpts0.astype(int).astype(np.float32)
        mkpts1 = mkpts1.astype(int).astype(np.float32)

        keypoints[name1][name2] = {"keypoints0":mkpts0,"keypoints1":mkpts1}

    with open(output_dir, "wb") as file:
        pickle.dump(keypoints, file)
    # # Let's find unique loftr pixels and group them together.
    # kpts = defaultdict(list)
    # match_indexes = defaultdict(dict)
    # total_kpts=defaultdict(int)
    # with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
    #     for k1 in f_match.keys():
    #         group  = f_match[k1]
    #         for k2 in group.keys():
    #             matches = group[k2][...]
    #             total_kpts[k1]
    #             kpts[k1].append(matches[:, :2])
    #             kpts[k2].append(matches[:, 2:])
    #             current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
    #             current_match[:, 0]+=total_kpts[k1]
    #             current_match[:, 1]+=total_kpts[k2]
    #             total_kpts[k1]+=len(matches)
    #             total_kpts[k2]+=len(matches)
    #             match_indexes[k1][k2]=current_match

    # for k in kpts.keys():
    #     kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    # unique_kpts = {}
    # unique_match_idxs = {}
    # out_match = defaultdict(dict)
    # for k in kpts.keys():
    #     uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
    #     unique_match_idxs[k] = uniq_reverse_idxs
    #     unique_kpts[k] = uniq_kps.numpy()
    # for k1, group in match_indexes.items():
    #     for k2, m in group.items():
    #         m2 = deepcopy(m)
    #         m2[:,0] = unique_match_idxs[k1][m2[:,0]]
    #         m2[:,1] = unique_match_idxs[k2][m2[:,1]]
    #         mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
    #                                 unique_kpts[k2][  m2[:,1]],
    #                                ],
    #                                axis=1)
    #         unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
    #         m2_semiclean = m2[unique_idxs_current]
    #         unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
    #         m2_semiclean = m2_semiclean[unique_idxs_current1]
    #         unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
    #         m2_semiclean2 = m2_semiclean[unique_idxs_current2]
    #         out_match[k1][k2] = m2_semiclean2.numpy()
    # with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
    #     for k, kpts1 in unique_kpts.items():
    #         f_kp[k] = kpts1

    # with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
    #     for k1, gr in out_match.items():
    #         group  = f_match.require_group(k1)
    #         for k2, match in gr.items():
    #             group[k2] = match