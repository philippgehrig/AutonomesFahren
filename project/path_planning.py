from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import heapq
class PathPlanning:
    def __init__(self):
        self.debug = 0
        pass


    @classmethod
    def plan(cls, left_boundary, right_boundary, distance_threshold=10) -> tuple[np.ndarray, float]:
        
        # 1: Normal Path Planning; 2: Target Line Path Planning
        planing_algorithm = 1

        # slim the left and right boundaries
        slim_factor = 1

        if len(left_boundary) > 0:
            left_boundary = left_boundary[::slim_factor]
        if len(right_boundary) > 0:
            right_boundary = right_boundary[::slim_factor]


        # NORMAL PATH PLANING: path always in the middle of the lane
        if planing_algorithm == 1: 
            path = cls.calculate_path(left_boundary, right_boundary)
            valid_path = cls.validate(path, distance_threshold)
            curvature_sum = cls.calculate_curvature(valid_path, True)
            curvature = cls.calculate_curvature(valid_path, False)
            inverted_path = cls.invert_path(valid_path, curvature)
            return inverted_path, curvature_sum
    
        # TARGET LINE PATH PLANNING: path based towards reducing the curvature
        elif planing_algorithm == 2:
            target_line = cls.calculate_target_line(left_boundary, right_boundary)
            valid_target_line = cls.validate(target_line, distance_threshold)
            curvature = cls.calculate_curvature(valid_target_line)
            return valid_target_line, curvature


    @staticmethod
    def calculate_path(left, right):
        """
        Calculates the middle path between the left and right boundaries.

        Args:
            left (list): List of points representing the left boundary.
            right (list): List of points representing the right boundary.

        Returns:
            numpy.ndarray: Array of points representing the middle path.
        """
        # Check if only lane is available on the screen
        if len(left) == 0 or len(right) == 0:
            if len(left):
                return left
            elif len(right):
                return right
            else:
                return []

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
        """
        Validates the given middle points of a path based on the distance threshold.

        Args:
            middle (np.ndarray): The middle points of the path.
            distance_threshold (float): The maximum allowed distance between consecutive points.

        Returns:
            np.ndarray: The valid middle points of the path.
        """
        if(len(middle) == 0):
            return np.array([])

        middle = np.array(middle)
        distances = np.linalg.norm(middle[1:] - middle[:-1], axis=1)
        mask = distances <= distance_threshold
        valid_points = middle[np.append(mask, True)]
        return valid_points

    def calculate_curvature(path, flag):
        """
        Calculate the curvature of a given path.

        Parameters:
        path (numpy.ndarray): The path coordinates as a 2D numpy array.

        Returns:
        float: The curvature of the path.

        """
        if(len(path) == 0):
            return 0


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
        
        if(flag):
            curvature = np.sum(np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)
            return curvature
        else:
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
            return curvature
        
    
    def invert_path(path, curvature):
        # I want to invert the path and reduce the amount of points to 12

        if len(path) == 0:
            return []
        
        if np.isscalar(curvature):
            curvature = [curvature]

        # Splice the first 10 points of the array:
        if len(curvature) < 10:
            next_points = curvature
        else:
            next_points = curvature[:10]
        
        #if curvature is larger than 40 within the next 10 points, decrease step size
        if any(point > 40 for point in next_points):
            path = np.array(path)
            path = path[::-1]  # Invert the path
            step = 5
            path = path[::step] #reduce the amount of points (only each 5th point)
            return path
        
        path = np.array(path)
        path = path[::-1]  # Invert the path
        step = 12 #reduce the amount of points (only each 12th point)
        path = path[::step]  # Select every step-th element
        return path
    
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

            curvature_ahead = np.abs(np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)

            # Append the curvature ahead to the list
            curvatures_ahead.append(curvature_ahead)

        # For the last 'look_ahead' points, we can't look ahead. So, we can either append zeros or copy the last computed curvature.
        curvatures_ahead.extend([0]*look_ahead)  # or curvatures_ahead.extend([curvatures_ahead[-1]]*look_ahead)

        # Convert the list of curvatures ahead to a numpy array
        curvatures_ahead = np.array(curvatures_ahead)

        return curvatures_ahead

    def calculate_target_line(left_boundary, right_boundary):
        
        def calculate_curvature(path):
            # Calculate curvature of the path
            dx = np.gradient(path[:, 0])
            dy = np.gradient(path[:, 1])

            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

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
        
        
        def calculate_bias(curvatures):
            # Define the bias for left turns, right turns, and straight paths
            bias_left_turn = 0.8
            bias_right_turn = 0.2
            bias_straight = 0.5

            # Calculate a moving average of the curvatures
            window_size = 10  # Adjust this value to change how quickly the bias responds to changes in the curvature
            curvatures_moving_average = np.convolve(curvatures, np.ones(window_size), 'valid') / window_size

            # Initialize an empty list to store the bias values
            bias = []

            # Loop over the moving average of the curvatures
            for curvature in curvatures_moving_average:
                # If the curvature is positive, a left turn is about to happen, so increase the bias
                if curvature > 0:
                    bias.append(bias_left_turn)

                # If the curvature is negative, a right turn is about to happen, so decrease the bias
                elif curvature < 0:
                    bias.append(bias_right_turn)

                # If the curvature is near zero, the path ahead is straight, so return the bias to a neutral value
                else:
                    bias.append(bias_straight)

            # Convert the list of bias values to a numpy array
            bias = np.array(bias)

            return bias

        if len(left_boundary) == 0 or len(right_boundary) == 0:
            if len(left_boundary):
                return left_boundary
            elif len(right_boundary):
                return right_boundary
            else:
                return []
            

        curvatures = calculate_curvature_ahead(left_boundary, 40)
        bias = calculate_bias(curvatures)

        # Interpolate left and right boundaries
        x_left, y_left = left_boundary[:, 0], left_boundary[:, 1]
        x_right, y_right = right_boundary[:, 0], right_boundary[:, 1]

        cs_left = CubicSpline(np.arange(len(x_left)), x_left), CubicSpline(np.arange(len(y_left)), y_left)
        cs_right = CubicSpline(np.arange(len(x_right)), x_right), CubicSpline(np.arange(len(y_right)), y_right)

        # Calculate middle line biased towards the outer boundary
        num_points = max(len(x_left), len(x_right))
        x = np.linspace(0, len(x_left)-1, num_points)

        bias_interp = interp1d(np.arange(len(bias)), bias, fill_value="extrapolate")
        bias_resampled = bias_interp(np.linspace(0, len(bias)-1, num_points))

        # Apply lateral offset to the middle line
        x_middle = (1 - bias_resampled) * cs_left[0](x) + bias_resampled * cs_right[0](x) 
        y_middle = (1 - bias_resampled) * cs_left[1](x) + bias_resampled * cs_right[1](x)

        target_line = np.column_stack((x_middle, y_middle))

        return target_line
    

        
   
