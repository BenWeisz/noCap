"""
    vision.py

    Helper functions that are used to tell
    where certain tracking points are in the
    input image.
"""

import cv2 as cv
import numpy as np

from colours import *


def compute_masks(hsv):
    """
    Compute the masks for as given
    input hsv opencv frame.

    hsv: The opencv input frame in hsv format.

    return: List of OpenCV frames of shape (height, width, 3)
            one for each tracking point.
    """

    masks = {}
    for colour in colour_map.keys():
        colour_data = colour_map[colour]
        colour_mask = (
            cv.inRange(
                hsv,
                colour_data["lower"],
                colour_data["upper"],
            )
            / 255.0
        )

        masks[colour] = colour_mask

    return masks


def compute_patch_center(masks):
    """
    Predict the approximate tracking point
    centers.

    masks: The opencv colour masks for each
           tracking point.

    return: Dictionary of tuples representing
            predicted tracking point locations.
    """
    patch_centers = {}
    for colour in colour_map.keys():
        y, x = np.where(masks[colour] == 1)

        y_center = None
        x_center = None

        if y.shape[0] != 0:
            y_center = int(np.sum(y) / y.shape[0])
        if x.shape[0] != 0:
            x_center = int(np.sum(x) / x.shape[0])

        patch_centers[colour] = (x_center, y_center)

    return patch_centers
