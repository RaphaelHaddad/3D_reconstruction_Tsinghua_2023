
from pathlib import Path

class DataPaths:
    def __init__(self, data_dir: str, output_dir: str, dataset: str, scene: str, mode: str):
        """Class to store paths.

        Args:
            data_dir (str): Path to data directory.
            output_dir (str): Path to output directory.
            dataset (str): Dataset name.
            scene (str): Scene name.
            mode (str): Mode (train or test).
        """
        if mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}")

        self.model_weights = Path(f"{data_dir}/weights")
        self.rotation_model_weights = self.model_weights / "model-vit-ang-loss.h5"

        self.input_dir = Path(f"{data_dir}/{mode}/{dataset}/{scene}")
        self.input_dir_images = Path(f"{data_dir}/{mode}/{dataset}/{scene}/images")
        
        self.output_scene_dir = output_dir / dataset / scene

        
        # for rotation 
        self.rotated_image_dir = self.output_scene_dir / "images_rotated"
        self.rotated_features_path = self.output_scene_dir / "features_rotated.h5"
        
        # for pairs 
        self.pairs_path = self.output_scene_dir / "pairs.txt"
        self.features_retrieval = self.output_scene_dir / "features_retrieval.h5"
        self.features_path = self.output_scene_dir / "features.h5"
        self.matches_path = self.output_scene_dir / "matches.h5"

        self.cropped_image_dir = self.output_scene_dir / "images_cropped"
        # create directories
        self.output_scene_dir.mkdir(parents=True, exist_ok=True)
        self.rotated_image_dir.mkdir(parents=True, exist_ok=True)
