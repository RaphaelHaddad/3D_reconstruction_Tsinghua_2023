import pickle
import numpy as np


def concat_keypoints(keypoints1_pickle,keypoints2_pickle):
    keypoints1 = None
    keypoints2 = None

    if(keypoints1_pickle):
        with open(keypoints1_pickle, "rb") as file:
            keypoints1 = pickle.load(file)

    if(keypoints2_pickle):
        with open(keypoints2_pickle, "rb") as file:
            keypoints2 = pickle.load(file)


    concatenated = {}

    for image,v in keypoints1.items():
        concatenated[image] = {}
        for image_pair,pair_keypoints in v.items():
            
            kp0 = pair_keypoints["keypoints0"]
            kp1 = pair_keypoints["keypoints0"]

            kp00 = keypoints2[image][image_pair]["keypoints0"]
            kp11 = keypoints2[image][image_pair]["keypoints0"]

            concatenated[image][image_pair] = {
                "keypoints0":np.vstack((kp0,kp00)),
                "keypoints1":np.vstack((kp1,kp11)) 
            }
    
    return concatenated

def build_superlist(keypoints):
    superlist = {} 
    temp_counter = {}

    def get_super_pair(latest_super_pair):
        super_pair = None
        for image_pair,pair_keypoints in keypoints[latest_super_pair].items():
            kp = pair_keypoints["keypoints0"]
            if(image_pair not in superlist and (latest_super_pair not in superlist or len(kp) > temp_counter[latest_super_pair])):
                temp_counter[latest_super_pair] = len(kp)
                superlist[latest_super_pair] = image_pair
                super_pair = image_pair
        return super_pair
    
    latest_super_pair = list(keypoints.keys())[0]

    while(latest_super_pair is not None):
        next_super_pair = get_super_pair(latest_super_pair)
        superlist[latest_super_pair] = next_super_pair
        latest_super_pair = next_super_pair

    return superlist
