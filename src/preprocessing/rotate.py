import gc
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import dioad.infer
import numpy as np
from tqdm import tqdm


def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image to max_size.

    Args:
        image (np.ndarray): Image to resize.
        max_size (int): Maximum size of the image.

    Returns:
        np.ndarray: Resized image.
    """
    img_size = image.shape[:2]
    if max(img_size) > max_size:
        ratio = max_size / max(img_size)
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)
    return image


def get_rotated_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
    """Rotate image by angle (rounded to 90 degrees).

    Args:
        image (np.ndarray): Image to rotate.
        angle (float): Angle to rotate the image by.

    Returns:
        Tuple[np.ndarray, float]: Rotated image and the angle it was rotated by.
    """
    if angle < 0.0:
        angle += 360
    angle = (round(angle / 90.0) * 90) % 360  # angle is now an integer in [0, 90, 180, 270]

    # rotate and save image
    if angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, int(angle)

def resize_images(input_dir: Path,image_list: List[str],output_dir: Path):
    for image_name in tqdm(image_list, desc=f"Rescaling {input_dir.name}", ncols=80):
        img_path = input_dir / "images" / image_name
        image = cv2.imread(str(img_path))
        resized = resize_image(image)

        if prev_shape is not None:
            same_original_shapes &= prev_shape == image.shape

        prev_shape = image.shape

        cv2.imwrite(str(output_dir / "images" / image_name), resized)

        return prev_shape

def rotate_images(images_dir: Path,image_list: List[str],output_dir: Path, model_weights_path):
    print("Rotate images")
    # rotate image
    n_rotated = 0
    n_total = len(image_list)

    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    deep_orientation = dioad.infer.Inference(load_model_path=model_weights_path)

    for image_name in tqdm(image_list, desc=f"Rotating {images_dir.name}", ncols=80):
        img_path = images_dir / image_name

        angle = deep_orientation.predict("vit", str(img_path))

        image = cv2.imread(str(img_path))
        image, angle = get_rotated_image(image, angle)

        if angle != 0:
            n_rotated += 1

        cv2.imwrite(str(output_dir / image_name), image)

    # free cuda memory
    del deep_orientation
    gc.collect()

    logging.info(f"Rotated {n_rotated} of {n_total} images.")
