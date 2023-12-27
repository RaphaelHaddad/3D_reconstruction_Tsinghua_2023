def rotate_keypoints(self) -> None:
        """Rotate keypoints back after the rotation matching."""
        self.log_step("Rotating keypoints")

        if not self.use_rotation_matching:
            logging.info("Not using rotation matching")
            return

        logging.info(f"Using rotated features from {self.paths.rotated_features_path}")
        shutil.copy(self.paths.rotated_features_path, self.paths.features_path)

        logging.info(f"Writing rotated keypoints to {self.paths.features_path}")
        with h5py.File(str(self.paths.features_path), "r+", libver="latest") as f:
            for image_fn, angle in self.rotation_angles.items():
                if angle == 0:
                    continue

                self.n_rotated += 1

                keypoints = f[image_fn]["keypoints"].__array__()
                y_max, x_max = cv2.imread(str(self.paths.rotated_image_dir / image_fn)).shape[:2]

                new_keypoints = np.zeros_like(keypoints)
                if angle == 90:
                    # rotate keypoints by -90 degrees
                    # ==> (x,y) becomes (y, x_max - x)
                    new_keypoints[:, 0] = keypoints[:, 1]
                    new_keypoints[:, 1] = x_max - keypoints[:, 0] - 1
                elif angle == 180:
                    # rotate keypoints by 180 degrees
                    # ==> (x,y) becomes (x_max - x, y_max - y)
                    new_keypoints[:, 0] = x_max - keypoints[:, 0] - 1
                    new_keypoints[:, 1] = y_max - keypoints[:, 1] - 1
                elif angle == 270:
                    # rotate keypoints by +90 degrees
                    # ==> (x,y) becomes (y_max - y, x)
                    new_keypoints[:, 0] = y_max - keypoints[:, 1] - 1
                    new_keypoints[:, 1] = keypoints[:, 0]
                f[image_fn]["keypoints"][...] = new_keypoints