import pickle
import numpy as np
import torch, platform


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


def split_image_into_regions(image):

    # Create empty lists for tiles from both images and their offsets
    tiles = []
    offsets = []

    height, width = image.shape[-2], image.shape[-1]

    # Calculate the center coordinates of the image
    center_x, center_y = width // 2, height // 2

    # Define the four corners of the image with offsets
    top_left = (0, 0)
    top_right = (center_x + 0, 0)
    bottom_left = (0, center_y + 0)
    bottom_right = (center_x + 0, center_y + 0)

    # Slice the image into four tiles using the specified offsets
    tile_top_left = image[..., top_left[1]:center_y + 0, top_left[0]:center_x + 0]
    tile_top_right = image[..., top_right[1]:center_y + 0, center_x + 0:]
    tile_bottom_left = image[..., center_y + 0:, top_left[0]:center_x + 0]
    tile_bottom_right = image[..., center_y + 0:, center_x + 0:]

    
    tiles.append(tile_top_left)
    tiles.append(tile_top_right)
    tiles.append(tile_bottom_left)
    tiles.append(tile_bottom_right)

    offsets.append(top_left)
    offsets.append(top_right)
    offsets.append(bottom_left)
    offsets.append(bottom_right)

    
    return tiles, offsets

def split_images_into_regions(img1, img2):
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])

    # Create empty lists for tiles from both images and their offsets
    tiles_img1 = []
    tiles_img2 = []
    offsets = []

    # Split each image into four parts
    w2 = min_w // 2
    h2 = min_h // 2
    
    # Top-left region
    tiles_img1.append(img1[:h2, :w2])
    tiles_img2.append(img2[:h2, :w2])
    offsets.append((0, 0))

    # Top-right region
    tiles_img1.append(img1[:h2, w2:])
    tiles_img2.append(img2[:h2, w2:])
    offsets.append((w2, 0))

    # Bottom-left region
    tiles_img1.append(img1[h2:, :w2])
    tiles_img2.append(img2[h2:, :w2])
    offsets.append((0, h2))

    # Bottom-right region
    tiles_img1.append(img1[h2:, w2:])
    tiles_img2.append(img2[h2:, w2:])
    offsets.append((w2, h2))

    return tiles_img1, tiles_img2, offsets


def set_torch_device():
    # Get the current operating system
    operating_system = platform.system()

    # Set the PyTorch device based on the operating system
    if operating_system == "Windows":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif operating_system == "Linux":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif operating_system == "Darwin":  # macOS
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")  # Default to CPU if the OS is not recognized

    return device
