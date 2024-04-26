from __future__ import annotations

import numpy as np


class LateralControl:

    def __init__(self, k=1.0, k_soft=0.5, delta_max=np.pi / 8):
        self._car_position = np.array([48, 67])
        self.k = k  # control gain
        self.k_soft = k_soft  # softening factor
        self.delta_max = delta_max  # max steering angle
        self.step = 0  # add a step counter

    def control(self, trajectory, speed):
        # Calculate the cross-track error
        cte = self._calculate_cte(trajectory)

        # Calculate the heading error
        theta_e = np.arctan2(trajectory[1, 1] - trajectory[0, 1], trajectory[1, 0] - trajectory[0, 0])
        theta_c = np.arctan2(self._car_position[1] - trajectory[0, 1], self._car_position[0] - trajectory[0, 0])
        he = theta_e - theta_c if self.step > 10 else 0  # ignore the heading error for the first 10 steps

        # Calculate the steering angle
        delta = np.arctan2(self.k * cte, speed + self.k_soft) + he

        # Add a correction term
        correction = 0.1  # adjust this value as needed
        delta += correction

        # Limit the steering angle
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        self.step += 1  # increment the step counter

        return delta
    
    def _calculate_cte(self, trajectory):
        # Calculate the distance to each point on the trajectory
        distances = np.linalg.norm(trajectory - self._car_position, axis=1)

        # Find the index of the lookahead point
        lookahead_distance = 5.0  # adjust this value as needed
        lookahead_index = np.argmin(np.abs(distances - lookahead_distance))

        # Calculate the cross-track error as the distance to the lookahead point
        cte = distances[lookahead_index]

        return cte

