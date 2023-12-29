import gc
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
from kp_imc23.external.dioad.dioad import infer as DioadInfer
import numpy as np
from tqdm import tqdm
import os

def split_images(images_dir: Path,image_list: List[str],output_dir: Path):
    print("Split images")
    
    # Create output folder if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    new_image_list = []

    for image_file in tqdm(image_list, desc=f"Splitting {images_dir.name}", ncols=80):
        # Construct the full path to the input image
        input_image_path = images_dir / image_file

        # Read the image
        image = cv2.imread(str(input_image_path))

        if image is not None:
            # Get the height and width of the image
            height, width = image.shape[:2]

            # Calculate the center of the image
            center_x, center_y = width // 2, height // 2

            # Split the image into four parts
            top_left = image[0:center_y, 0:center_x]
            top_right = image[0:center_y, center_x:width]
            bottom_left = image[center_y:height, 0:center_x]
            bottom_right = image[center_y:height, center_x:width]

            # Create output subfolder for the current image
            image_filename, _ = os.path.splitext(image_file)
            output_subfolder = f"{output_dir}/{image_filename}"
            output_subfolder.mkdir(exist_ok=True)

            # Save the individual parts
            cv2.imwrite(f"{output_subfolder}_top_left.jpg"    , top_left)
            cv2.imwrite(f"{output_subfolder}_top_right.jpg"   , top_right)
            cv2.imwrite(f"{output_subfolder}_bottom_left.jpg" , bottom_left)
            cv2.imwrite(f"{output_subfolder}_bottom_right.jpg", bottom_right)
            new_image_list.append(f"{output_subfolder}_top_left.jpg"    )
            new_image_list.append(f"{output_subfolder}_top_right.jpg"   )
            new_image_list.append(f"{output_subfolder}_bottom_left.jpg" )
            new_image_list.append(f"{output_subfolder}_bottom_right.jpg")
        
    return new_image_list