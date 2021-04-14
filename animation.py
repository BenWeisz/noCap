"""
    animation.py

    Animate the input model.
"""

import numpy as np
import cv2 as cv

from kinematics import *


def animate_bones(frame, skeleton, transformations):
    """
    Animate the position of the bones of the motion
    capture rig.

    frame: The opencv frame on which to draw the animated
           set of bones
    skeleton: List of bone objects representing the skeleton
              of the model.
    transformations: List of 2d Affine transformation matricies for each
                     bone in the motion capture rig.

    return: OpenCV frame of shape (height, width, 3).
    """
    for bone in skeleton:
        start = np.array([bone.start[1][0], bone.start[1][1], 1])
        end = np.array([bone.end[1][0], bone.end[1][1], 1])

        transform = transformations[str(bone)]
        new_start = transform.dot(start)
        new_start = (int(new_start[0]), int(new_start[1]))

        new_end = transform.dot(end)
        new_end = (int(new_end[0]), int(new_end[1]))

        end_colour = colour_map[bone.end[0]]["label"]
        colour = (end_colour[0] / 255.0, end_colour[1] / 255.0, end_colour[2] / 255.0)

        cv.line(frame, new_start, new_end, colour, 1)

    return frame


def animate_model(frame, skeleton, verts, edges, transformations, weights):
    """
    Animate the model verticies based on the location of the bones

    frame: The opencv frame on which to draw the animated
           set of bones
    skeleton: List of bone objects representing the skeleton
              of the model.
    verts: |V| x 2 numpy matrix of 2d coordinates representing
           the locations of the model's mesh points.
    edges: |E| x 2 numpy matrix of indicies representing
           the verticies used in the i-th(row of <edges>) edge
           of the model's mesh.
    transformations: List of 2d Affine transformation matricies for each
                     bone in the motion capture rig.
    weights: The skinning proximity weights of each model vertex
             based on the point's proximity to the motion capture
             rig bones during the calibration phase.

    return: OpenCV frame of shape (height, width, 3).
    """
    for e in range(edges.shape[0]):
        edge = edges[e, :]

        start = tuple(verts[edge[0], :])
        end = tuple(verts[edge[1], :])

        colour = get_edge_colour(skeleton, weights, edge)

        cv.line(frame, start, end, colour, 1)

    return frame


def animate(frame, skeleton, centers, verts, edges, weights):
    """
    Animate the bones and the mesh model.

    frame: The opencv frame on which to draw the animated
           set of bones
    skeleton: List of bone objects representing the skeleton
              of the model.
    centers: Dictionary of joint centers by colour keys from
             colours.py.
    verts: |V| x 2 numpy matrix of 2d coordinates representing
           the locations of the model's mesh points.
    edges: |E| x 2 numpy matrix of indicies representing
           the verticies used in the i-th(row of <edges>) edge
           of the model's mesh.
    weights: The skinning proximity weights of each model vertex
             based on the point's proximity to the motion capture
             rig bones during the calibration phase.

    return: OpenCV frame of shape (height, width, 3).
    """
    animated_frame = frame
    try:
        transformations = compute_transformations(skeleton, centers)
        verts = compute_transformed_verts(verts, transformations, skeleton, weights)
        animated_frame = animate_bones(animated_frame, skeleton, transformations)
        animated_frame = animate_model(
            animated_frame, skeleton, verts, edges, transformations, weights
        )

        return animated_frame
    except (ValueError, IndexError, ArithmeticError, KeyError, TypeError) as e:
        return animated_frame
