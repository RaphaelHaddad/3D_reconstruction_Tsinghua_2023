import gc
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
from kp_imc23.external.dioad.dioad import infer as DioadInfer
import numpy as np
from tqdm import tqdm

def split_images(images_dir: Path,image_list: List[str],output_dir: Path):
    print("Split images")
    
    # Create output folder if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in image_list:
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
            output_subfolder = output_dir / image_file.stem
            output_subfolder.mkdir(exist_ok=True)

            # Save the individual parts
            cv2.imwrite(str(output_subfolder / 'top_left.jpg'), top_left)
            cv2.imwrite(str(output_subfolder / 'top_right.jpg'), top_right)
            cv2.imwrite(str(output_subfolder / 'bottom_left.jpg'), bottom_left)
            cv2.imwrite(str(output_subfolder / 'bottom_right.jpg'), bottom_right)