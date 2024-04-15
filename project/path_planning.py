from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d

class PathPlanning:

    def __init__(self):
        pass

    def plan(self, left_boundry, right_boundry):
        points = calculate_path(self, left_boundry, right_boundry)
        middle = validate(points)
        curvature = calculate_curvature(middle)
        return middle, curvature

def calculate_path(self, left, right):
    #Check if only lane is availible on the screen
    if len(left) == 0 or len(right) == 0:
        if len(left):
            return left
        elif len(right):
            return right
        else:
            return []

    # Convert to numpy arrays
    left = np.array(left)
    right = np.array(right)

    # Create an array of x values from 0 to 1
    x = np.linspace(0, 1, max(len(left), len(right)))

    # Create interpolation functions for the x and y coordinates of the left and right boundaries
    f_left_x = interp1d(np.linspace(0, 1, len(left)), left[:, 0], kind='linear', fill_value="extrapolate")
    f_left_y = interp1d(np.linspace(0, 1, len(left)), left[:, 1], kind='linear', fill_value="extrapolate")
    f_right_x = interp1d(np.linspace(0, 1, len(right)), right[:, 0], kind='linear', fill_value="extrapolate")
    f_right_y = interp1d(np.linspace(0, 1, len(right)), right[:, 1], kind='linear', fill_value="extrapolate")

    # Interpolate the x and y coordinates of the left and right boundaries to the common x values
    left_interp = np.column_stack((f_left_x(x), f_left_y(x)))
    right_interp = np.column_stack((f_right_x(x), f_right_y(x)))

    # Calculate the middle path
    middle_points = (left_interp + right_interp) / 2

    return middle_points

def validate(middle):
    distances = np.linalg.norm(middle[1:] - middle[:-1], axis=1)
    mask = distances <= 10
    validpoints = middle[np.append(mask, True)]
    return validpoints


def calculate_curvature(middle):
    # Calculate the first derivatives
    dx = np.gradient(middle[:, 0])
    dy = np.gradient(middle[:, 1])


    # Calculate the second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate the curvature
    curvature = np.sum(np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)
    print("curvature: ", curvature)
    return curvature