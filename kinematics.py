"""
    kinematics.py

    Methods in this file are concerned with calculating values
    that are related to the kinematics of the model.
"""

import numpy as np
import cv2 as cv
import math

from colours import *


def angle(p1, p2):
    """
    Compute the direction vector between p1, p2 --> d.
    Then compute the angle that d makes with the x-axis.

    p1: iterable object of the form (x, y)
    p2: iterable object of the form (x, y)

    return: Angle in radians.
    """
    p1_np = np.array(p1)
    p2_np = np.array(p2)

    d = p2_np - p1_np
    u = np.array([1, 0])

    angle_dist = math.acos(d.dot(u) / np.linalg.norm(d))
    actual_angle = -(angle_dist if d[1] > 0 else (2 * math.pi) - angle_dist)

    return actual_angle


def rotation_matrix(theta):
    """
    Compute the 2d affine transformation matrix given angle theta

    theta: The angle that the rotation matrix should represent

    return: 2d Affine matrix.
    """
    rot = np.zeros((3, 3))
    rot[2, 2] = 1

    cos_rot = math.cos(theta)
    sin_rot = math.sin(theta)

    rot[0, 0] = cos_rot
    rot[0, 1] = -sin_rot
    rot[1, 0] = sin_rot
    rot[1, 1] = cos_rot

    return rot


class Bone:
    """
    This class represents one bone in a kinematics skeleton.
    """

    def __init__(self, start=None, end=None, parent=None):
        """
        start: (COLOUR, pos)
        end:   (COLOUR, pos)
        parent: Bone
        """
        self.start = start
        self.end = end
        self.parent = parent

    def set_positions(self, centers):
        """
        Given center positions by colour,
        set the start and end to the appropriate
        positions.

        centers: Dictionary of joint centers by colour keys from
                 colours.py.
        """
        self.start = (self.start[0], centers[self.start[0]])
        self.end = (self.end[0], centers[self.end[0]])

    def compute_transform(self, parent_transformation, new_theta):
        """
        Compute the 2d Affine transformation for this bone
        given its parent transformation, and the current theta
        offset.

        parent_transformation: The 2d Affine transformation matrix
                               for this bone's parent bone.
        new_theta: The new rotation theta for the bone.

        return: 2d Affine transformation matrix.
        """
        rest_theta = angle(self.start[1], self.end[1])
        rest_rot = rotation_matrix(rest_theta)
        rest_rot[0, 2] = self.start[1][0]
        rest_rot[1, 2] = self.start[1][1]

        new_rot = rotation_matrix(rest_theta - new_theta)
        rot_mat = parent_transformation.dot(
            rest_rot.dot(new_rot.dot(np.linalg.inv(rest_rot)))
        )

        return rot_mat

    def __str__(self):
        """
        Return the string representation of this bone.

        return: string representation.
        """

        return "Start: " + str(self.start) + " End: " + str(self.end)


def compute_transformations(skeleton, centers):
    """
    Compute the 2d Affine transformations for all the bones.

    centers: Dictionary of joint centers by colour key provided
             in colours.py.

    return: List of 2d Affine transformation matricies.
    """
    transformations = {}
    for bone in skeleton:
        curr_bone = bone
        bone_stack = []
        while curr_bone is not None:
            if str(curr_bone) not in transformations:
                bone_stack.append(curr_bone)
                curr_bone = curr_bone.parent
            else:
                break

        while len(bone_stack) > 0:
            bone = bone_stack.pop()
            if str(bone) in transformations:
                continue

            parent_transform = np.eye(3)
            if bone.parent is not None:
                parent_transform = transformations[str(bone.parent)]

            new_theta = angle(centers[bone.start[0]], centers[bone.end[0]])
            transformations[str(bone)] = bone.compute_transform(
                parent_transform, new_theta
            )

    return transformations


def compute_skinning_weights(skeleton, verts):
    """
    Compute the skinning weights for each point
    in the 2d model mesh. Assign relationship weight
    between each point and bone based on the inverse
    distance from the model point to the respective bone.
    Use the softmax function to emphasize the weights of
    nearby bones.

    skeleton: List of bone objects representing the skeleton
              of the model.
    verts:  |V| x 2 numpy matrix of 2d coordinates representing
            the locations of the model's mesh points.

    return |bones| x |V| matrix of skinning weights.
    """
    num_bones = len(skeleton)

    # Step 1 Compute the parameters t for each vert that
    # is closest to the point on the line
    s = np.zeros((num_bones, 2))
    e = np.zeros((num_bones, 2))
    for i in range(num_bones):
        s[i, :] = np.array(skeleton[i].start[1])
        e[i, :] = np.array(skeleton[i].end[1])

    d = e - s
    de = np.sum(d * e, axis=1)
    ds = np.sum(d * s, axis=1)

    d_tiled = np.tile(d, (verts.shape[0], 1, 1))
    d_tiled = d_tiled.transpose((1, 0, 2))

    dq = np.sum(d_tiled * verts, axis=2).transpose()
    t = np.clip((dq - ds) / (de - ds), 0, 1)

    # Step 2 Compute the actual points on the line
    # based on the clamped t values

    d_tiled = d_tiled.transpose((2, 1, 0))
    s_tiled = np.tile(s, (verts.shape[0], 1, 1)).transpose(2, 0, 1)
    p = s_tiled + (d_tiled * t)

    # Step 3 Compute the distances between the points on the bones
    # and the verticies
    dist = p.transpose((2, 1, 0)) - verts
    dist = dist ** 2
    dist = np.sum(dist, axis=2)
    dist = np.sqrt(dist)

    closest_p = np.argmin(dist, axis=0)

    # Compute the softmax blending co-efficients
    mixing_coefficient = 10

    inv_dist = 1 / dist
    normed_inv_dist = inv_dist / np.sum(inv_dist, axis=0)
    zexp = np.exp(normed_inv_dist * mixing_coefficient)
    zexp_tot = np.sum(zexp, axis=0)
    weights = zexp / zexp_tot

    return weights


def get_edge_colour(skeleton, weights, edge):
    """
    Get the colour of a given model edge.
    The colour of the edge is based on the
    proximity weighted colour average of the
    edge to its closest bones. These weights
    being based on the centers from the
    calibration phase.

    skeleton: List of bone objects representing the skeleton
              of the model.
    weights: The skinning proximity weights of each model vertex
             based on the point's proximity to the motion capture
             rig bones during the calibration phase.
    edge: (v1, v2) vertex index pair into columns of weights

    return: RGB tuple of colour.
    """
    colour = np.zeros(3)
    start_ind = edge[0]
    start_weights = weights[:, start_ind]

    for i in range(start_weights.shape[0]):
        bone_colour = np.array(colour_map[skeleton[i].end[0]]["label"])
        colour += bone_colour * start_weights[i]

    colour /= 255.0
    return tuple(colour)


def compute_transformed_verts(verts, transformations, skeleton, weights):
    """
    Compute the transformed model verticies based on the computed
    transformations for the verticies in the given frame, as well
    as the skinning weights.

    verts: |V| x 2 numpy matrix of 2d coordinates representing
           the locations of the model's mesh points.
    transformations: List of 2d Affine transformation matricies for each
                     bone in the motion capture rig.
    skeleton: List of bone objects representing the skeleton
              of the model.
    weights: The skinning proximity weights of each model vertex
             based on the point's proximity to the motion capture
             rig bones during the calibration phase.

    return: |V| x 2 matrix of transformed mesh verticies.
    """
    transformed_verts = np.zeros(verts.shape)
    for i, bone in enumerate(skeleton):
        bone_transform = transformations[str(bone)]

        affine_verts = np.ones((verts.shape[0], 3))
        affine_verts[:, :2] = verts
        affine_verts = affine_verts.transpose()

        transformed_affine_verts = bone_transform.dot(affine_verts)
        bone_weights = weights[i, :]

        weighted_transformed_verts = (transformed_affine_verts * bone_weights)[:2, :]

        transformed_verts += weighted_transformed_verts.transpose()

    return np.floor(transformed_verts).astype(int)
