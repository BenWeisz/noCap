"""
	labeling.py

	Additional frame labeling functions to
	enhance user experience.
"""

import numpy as np
import cv2 as cv

from colours import *


def create_colour_labeled_image(masks):
    """
    Create the composite image of motion
    capture predicted joint tracking points.
    Label each mask point with its respective
    labeling colour from colours.py.

    masks: List of opencv mask frames for each
               tracking point in the motion capture
               rig.

    return: OpenCV frame of shape (height, width, 3).
    """
    height = masks[0].shape[0]
    width = masks[0].shape[1]
    labeled = np.zeros((height, width, 3))

    for colour in colour_map.keys():
        colour_data = colour_map[colour]
        colour_mask = masks[colour].reshape((height, width, 1))
        colour_label = np.array(colour_data["label"]).reshape((1, 3))
        colour_mask = colour_mask * colour_label

        labeled += colour_mask

    return labeled / 255.0


def label_patch_centers(frame, centers):
    """
    Label the predicted tracking point centers
    directly on the input image from the camera.

    frame: The opencv input image.
    centers: Dictionary of colours in colours.py
                     to tracking point center coordinates.

    return: OpenCV image of shape (height, width, 3).
    """
    labeled = frame
    for colour in colour_map.keys():
        if centers[colour][0] is not None and centers[colour][1] is not None:
            cv.circle(labeled, centers[colour], 5, colour_map[colour]["label"], -1)

    return labeled


def overlay_model_points(frame, verts):
    """
            Draw out the verticies of the input model for
            calibration purposes.

            frame: The opencv input image.
    verts: |V| x 2 numpy matrix of 2d coordinates representing
           the locations of the model's mesh points.

            return: OpenCV frame of shape (height, width, 3).
    """
    for i in range(verts.shape[0]):
        frame = cv.circle(frame, tuple(verts[i]), 2, (255, 255, 255), -1)

    return frame
