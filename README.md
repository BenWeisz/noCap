# 2D Motion Capture App
## Motivation
---
The use of motion capture in animated movies / games seems like a really cool alternative to using a model that is animated through the use of inverse kinematics. Unfortunatly this can be quite expensive and as an undergraduate with limited funds somewhat hard to do. In this project I make an attempt at reproducing motion capture in 2D with a simple 2D skeleton rig. 
## Introduction
---
The goal of this project is to try to get motion capture working in 2D with a webcam. In this project I made a small rig with 7 coloured tracking points to model the tracking points in a real motion capture system. 

The webcam picks up on the image of the rig using OpenCV. This image is then converted to HSV and is filtered for the various tracking points. 

Once these tracking points have been found, we take the average of the activated pixels to compute the approximate patch center of these joints. These joints are then mapped to a skeleton structure of bones.

We then compute the change in angle from the rest bone angles to compute the rotation matricies for each bone (Forward Kinematics). These bones are then drawn onto the screen.

In addition to these bones, we also have a 2d mesh model that surrounds these bones. During the calibration phase (when the user indicates that the mesh is calibrated), the mesh points are queried for their closest bone, and are approprate skinning weights are assigned for each vertex / bone combination. 

Using the computed transformation angles from above, and the skinning weights we transform the vertex points of the mesh model to their new positions.

## How to use (May be hard to set up, please look at giphy instead: https://gph.is/g/4oXkOAK)
DEMO WILL BE SHOWN IN CLASS

Step 1:
- Make a mesh model using the provided processing project (.pde file)
    -   You will have to download the processing app from processing.org/downloads to run the file.
    -   Follow the instrutions in the app
Step 2:
-   Once the model is exported, plug in your webcam

Step 3:
-   Launch `python calibrate.py`. Use the HSV sliders to calibrate the lower and upper bound for each tracking colour. Update the bounds of each of the colours in the `colours.py` file.

Step 4:
-   A screen will show up with the model points. Align the rig bones within the model to ensure that the model will be properly calibrated. Press C to complete the calibration of the rig.

Step 5:
-   Move the model around with a stick to see the rig move! Just like a real motion capture rig!
