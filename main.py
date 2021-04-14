"""
	main.py

	2d motion capture app.
"""

import cv2 as cv
import numpy as np
import argparse

from colours import *
from labeling import *
from model import *
from vision import *
from animation import *


def setup_camera(camera_num):
    """
    Set up the opencv camera input.

    camera_num: The camera num for the input camera.

    return: OpenCV camera resource.
    """
    return cv.VideoCapture(camera_num)


def tear_down_camera(camera):
    """
    Tear down the opencv camera resource.

    camera: The camera to tear down.
    """
    camera.release()
    cv.destroyAllWindows()


def handle_frames(camera, model_path):
    """
    Handle a single input frame from the opencv
    camera resource.

    camera: The opencv input camera.
        model_path: The path to the mesh model.
    """
    verts, edges = load_model_obj(model_path)
    calibrate_phase = True

    # Calibration Data
    skeleton = generate_skeleton()
    weights = None

    while True:
        ret, input_frame = camera.read()
        if not ret:
            print("No Camera Is Connected.")
            print("Please wait for in class demo")
            print("-----------------------------")
            print("If you wish to see an app demo, look")
            print("at this GIF: https://gph.is/g/4oXkOAK")
            break

        hsv = cv.cvtColor(input_frame, cv.COLOR_BGR2HSV)

        masks = compute_masks(hsv)
        cmask = create_colour_labeled_image(masks)
        centers = compute_patch_center(masks)

        if calibrate_phase:
            input_frame = overlay_model_points(
                input_frame,
                verts,
            )
        else:
            animated_frame = np.zeros(input_frame.shape)
            animated_frame = animate(
                animated_frame, skeleton, centers, verts, edges, weights
            )
            cv.imshow("Animation", animated_frame)

        input_frame = label_patch_centers(input_frame, centers)

        cv.imshow("Motion Capture", input_frame)
        cv.imshow("Key Point Map", cmask)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and calibrate_phase:
            for bone in skeleton:
                bone.set_positions(centers)
            weights = compute_skinning_weights(skeleton, verts)
            calibrate_phase = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Run motion capture with a different input mesh.",
        type=str,
    )
    args = parser.parse_args()

    mesh_model_path = "model.obj"
    try:
        mesh_model_path = parser.model
    except AttributeError:
        pass

    camera = setup_camera(0)
    handle_frames(
        camera,
        mesh_model_path,
    )
    tear_down_camera(camera)
