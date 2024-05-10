from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from queue import PriorityQueue
import heapq
class PathPlanning:
    @classmethod
    def plan(cls, left_boundary, right_boundary, distance_threshold=10) -> tuple[np.ndarray, float]:
        path = cls.calculate_path(left_boundary, right_boundary)
        valid_path = cls.validate(path, distance_threshold)
        curvature = cls.calculate_curvature(valid_path)
        # curvature_ahead = cls.calculate_curvature_ahead(path, look_ahead=10)
        # path = cls.calculate_target_line(left_boundary, right_boundary, 0.2, curvature_ahead)
        return path, curvature

    @staticmethod
    def calculate_path(left, right):
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
    
 
    @staticmethod
    def validate(middle, distance_threshold) -> np.ndarray:
        # Convert lists to numpy arrays before subtracting
        middle = np.array(middle)
        distances = np.linalg.norm(middle[1:] - middle[:-1], axis=1)
        mask = distances <= distance_threshold
        valid_points = middle[np.append(mask, True)]
        return valid_points

    def calculate_curvature(path):
        # Calculate curvature of the path
        if np.count_nonzero(~np.isnan(path[:, 0])) >= 2:
            dx = np.gradient(path[:, 0])
        else:
            return 0
        if np.count_nonzero(~np.isnan(path[:, 1])) >= 2:
            dy = np.gradient(path[:, 1])
        else:
            return 0

        if np.count_nonzero(~np.isnan(dx)) >= 2:
            ddx = np.gradient(dx)
        else:
            return 0

        if np.count_nonzero(~np.isnan(dy)) >= 2:
            ddy = np.gradient(dy)
        else:
            return 0
        
        
        curvature = np.sum(np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)
        return curvature
    
    def calculate_curvature_ahead(path, look_ahead):
        # Initialize an empty list to store the curvature ahead for each point
        curvatures_ahead = []

        # Loop over each point in the path
        for i in range(len(path) - look_ahead):
            # Determine the subset of the path that is ahead of the current position
            path_ahead = path[i:i + look_ahead]

            # Calculate curvature of the path ahead
            dx = np.gradient(path_ahead[:, 0])
            dy = np.gradient(path_ahead[:, 1])

            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            curvature_ahead = np.sum(np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)

            # Append the curvature ahead to the list
            curvatures_ahead.append(curvature_ahead)

        # For the last 'look_ahead' points, we can't look ahead. So, we can either append zeros or copy the last computed curvature.
        curvatures_ahead.extend([0]*look_ahead)  # or curvatures_ahead.extend([curvatures_ahead[-1]]*look_ahead)

        # Convert the list of curvatures ahead to a numpy array
        curvatures_ahead = np.array(curvatures_ahead)

        return curvatures_ahead
            

    # Introduced BIAS Variable which can move the target line inside and outside
    # bias = 0 => left boundry; bias = -1 =>

    def calculate_target_line(left_boundary, right_boundary, bias, curvature_ahead):
        
        if len(left_boundary) == 0 or len(right_boundary) == 0:
            if len(left_boundary):
                return left_boundary
            elif len(right_boundary):
                return right_boundary
            else:
                return []

        
        # Interpolate _boundry and right boundaries
        x_left, y_left = left_boundary[:, 0], left_boundary[:, 1]
        x_right, y_right = right_boundary[:, 0], right_boundary[:, 1]

        cs_left = CubicSpline(np.arange(len(x_left)), x_left), CubicSpline(np.arange(len(y_left)), y_left)
        cs_right = CubicSpline(np.arange(len(x_right)), x_right), CubicSpline(np.arange(len(y_right)), y_right)

        # Calculate middle line biased towards the outer boundary
        num_points = max(len(x_left), len(x_right))
        x = np.linspace(0, len(x_left)-1, num_points)

        # Adjust bias based on curvature ahead
        adjusted_bias = bias + curvature_ahead

        # Apply lateral offset to the middle line
        x_middle = (1 - bias) * cs_left[0](x) + bias * cs_right[0](x) 
        y_middle = (1 - bias) * cs_left[1](x) + bias * cs_right[1](x)

        target_line = np.column_stack((x_middle, y_middle))
        

        return target_line
    
   
