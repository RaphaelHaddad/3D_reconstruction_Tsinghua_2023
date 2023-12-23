import argparse
import os

from typing import Any, Dict, Tuple
from src.config.paths import DataPaths

def match(
    paths: DataPaths,
    args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Matches pairs of images and outputs keypoints.

    Though here, the whole matching process will be done with both a detector-free and a detector-based method
    that will be merged (first SP/SG & DKMv3).

    Args:s
        paths (DataPaths): contains all defined paths for the computation results.
        args (argparse.Namespace): Arguments.
    """

    # Matches from DKMv3
    matches_from_detector_free = None

    # Matches from SP/SG
    matches_from_detector_based = None

