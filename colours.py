"""
    colours.py

    This file provides colour information for the project to user.
    Each colour has a specified hsv upper and lower bound for detecting
    the colour with a camera. Label (r, g, b) values are also provided
    to easy label points with a representative colour.
"""

RED = 0
GREEN = 1
GRAY = 2
YELLOW = 3
BLUE = 4
BROWN = 5
PINK = 6

colour_map = {
    RED: {"lower": (0, 254, 45), "upper": (10, 255, 202), "label": (0, 0, 255)},
    GREEN: {"lower": (34, 149, 76), "upper": (67, 255, 151), "label": (0, 255, 0)},
    GRAY: {"lower": (77, 139, 0), "upper": (119, 167, 71), "label": (104, 104, 117)},
    YELLOW: {"lower": (17, 188, 149), "upper": (34, 255, 176), "label": (33, 248, 255)},
    BLUE: {"lower": (114, 192, 71), "upper": (129, 255, 163), "label": (255, 0, 0)},
    BROWN: {"lower": (0, 107, 29), "upper": (26, 255, 64), "label": (16, 72, 125)},
    PINK: {"lower": (142, 221, 63), "upper": (255, 255, 255), "label": (219, 81, 237)},
}
