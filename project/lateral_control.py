from __future__ import annotations

import numpy as np


class LateralControl:

    def __init__(self, k=0.3, k_soft=0.8, delta_max=1):
        self._car_position = np.array([48, 62])
        self.k = k  # control gain
        self.k_soft = k_soft  # softening factor
        self.delta_max = delta_max  # max steering angle
        self.step = 0  # add a step counter
        self.clp = [0,0]  # closest lookahead point
        self.sclp = [0,0]  # second closest lookahead point

    def control(self, trajectory, speed):
        """
        Controls the lateral movement of the car based on the given trajectory and speed.

        Args:
            trajectory (numpy.ndarray): The trajectory of the car.
            speed (float): The current speed of the car.

        Returns:
            float: The calculated steering angle.

        Raises:
            None

        """

        # Check if the trajectory is empty
        if len(trajectory) == 0:
            print("Trajectory = 0") # debug message
            return 0 # car should not steer if no trajectory is found
        
        # Calculate the cross-track error
        # sharpe_turn_flag = 1 => sharp left; 2 => sharp right; 0 => normal
        cte, lookahead_index, sharp_turn_flag = self._calculate_cte(trajectory)
        # Check if the lookahead index is valid
        if(len(trajectory) < lookahead_index + 2):
            print("Trajectory index out of bounds") #debug message
            return 0 # car should not steer if trajectory index is out of bounds
        
        if(sharp_turn_flag == 1):
            # SHARP LEFT TURN => steer right with 0.3 until normal CLP can be calculated again
            return 0.3
        
        elif(sharp_turn_flag == 2):
            # SHARP RIGHT TURN => steer left with -0.3 until normal CLP can be calculated again
            return -0.3    
        
        else:
            desired_heading_angle = np.arctan2(trajectory[lookahead_index + 1, 1] - trajectory[lookahead_index, 1], trajectory[lookahead_index + 1, 0] - trajectory[lookahead_index, 0])    
            current_heading_angle = np.arctan2(self._car_position[1] - trajectory[0, 1], self._car_position[0] - trajectory[0, 0])
            he = desired_heading_angle - current_heading_angle if self.step > 10 else 0  # ignore the heading error for the first 10 frame => zoom in
        # Calculate the steering angle
        delta = np.arctan2(self.k * cte, speed + self.k_soft) + he
        # Limit the steering angle
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        self.step += 1  # increment the  dstep counter
        return delta
    
    def _calculate_cte(self, trajectory):
        """
        Calculates the cross-track error (cte) and the index of the lookahead point.

        Parameters:
        trajectory (numpy.ndarray): The trajectory points.

        Returns:
        cte (float): The cross-track error.
        lookahead_index (int): The index of the lookahead point in the trajectory.
        """
        # Calculate the distance to each point on the trajectory
        distances = np.linalg.norm(trajectory - self._car_position, axis=1)

        # Find the index of the lookahead point
        lookahead_distance = 0.0  # adjust this value as needed
        lookahead_index = np.argmin(np.abs(distances - lookahead_distance))


        self.clp = trajectory[lookahead_index]
        
        # Check if the lookahead index is valid
        if lookahead_index + 1 >= len(trajectory):
            print("Lookahead index out of bounds") #debug message
            self.sclp = None
        else:
            self.sclp = trajectory[lookahead_index + 1]


        # SHARP LEFT TRUN AHEAD
        if(self.clp[1] > self._car_position[1] +4 or self.clp[0] > self._car_position[0]+4):
            cte = distances[lookahead_index]
            return cte, lookahead_index, 1

        # SHARP RIGHT TURN AHEAD
        if(self.clp[1] < self._car_position[1] -4 or self.clp[0] < self._car_position[0]-4):
            cte = distances[lookahead_index]
            return cte, lookahead_index, 2


        # Calculate the cross-track error as the distance to the lookahead point
        cte = distances[lookahead_index]

        return cte, lookahead_index, 0

