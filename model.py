"""
    model.py

    Helper functions used to set up the model.
"""

import numpy as np

from kinematics import *
from colours import *


def load_model_obj(path):
    """
    Load the mesh model from the given model path.

    path: The path to the model file.

    return:
        verts: |V| x 2 numpy matrix of mesh verticies.
        edges: |E| x 2 numpy matrix of mesh vertex index pairs.
    """
    verts = []
    edges = []

    with open(path, "r") as model_file:
        lines = model_file.readlines()
        for line in lines:
            tokens = line.strip().split(" ")
            if tokens[0] == "V":
                verts.append([int(tokens[1]), int(tokens[2])])
            elif tokens[0] == "E":
                edges.append([int(tokens[1]), int(tokens[2])])

    verts = np.array(verts)
    edges = np.array(edges)

    return verts, edges


def generate_skeleton():
    """
    Set up the relational data for the bones in the
    model skeleton.

    return: List of Bone objects making up skeleton.
    """
    torso = Bone((GREEN, None), (BLUE, None))
    right_leg = Bone((BLUE, None), (BROWN, None), torso)
    left_leg = Bone((BLUE, None), (PINK, None), torso)
    right_arm = Bone((GREEN, None), (GRAY, None))
    left_arm = Bone((GREEN, None), (YELLOW, None))
    head = Bone((GREEN, None), (RED, None))

    return [torso, right_leg, left_leg, right_arm, left_arm, head]
