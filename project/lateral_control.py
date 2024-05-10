from __future__ import annotations

import numpy as np


class LateralControl:

    def __init__(self, k=0.8, k_soft=0, delta_max=1):
        self._car_position = np.array([48, 62])
        self.k = k  # control gain
        self.k_soft = k_soft  # softening factor
        self.delta_max = delta_max  # max steering angle
        self.step = 0  # add a step counter
        self.clp = [0,0]  # closest lookahead point
        self.sclp = [0,0]  # second closest lookahead point

    def control(self, trajectory, speed):

        # Check if the trajectory is empty
        if len(trajectory) == 0:
            print("Trajectory = 0") # debug message
            return 0
        
        # Calculate the cross-track error
        # sharpe_turn_flag = 1 => sharp left; 2 => sharp right; 0 => normal
        cte, lookahead_index, sharp_turn_flag = self._calculate_cte(trajectory)
        print("cte: ", round(cte, 2))
        # Check if the lookahead index is valid
        if(len(trajectory) < lookahead_index + 2):
            print("Trajectory index out of bounds") #debug message
            return 0
        
        if(sharp_turn_flag == 1):
            # SHARP LEFT TURN => steer right with 0.3 until normal CLP can be calculated again
            return 0.3
        
        elif(sharp_turn_flag == 2):
            # SHARP RIGHT TURN => steer left with -0.3 until normal CLP can be calculated again
            return -0.3    
        
        else:
            desired_heading_angle = np.arctan2(trajectory[lookahead_index +1, 0] - trajectory[lookahead_index, 0], trajectory[lookahead_index+1, 1] - trajectory[lookahead_index, 1])    
            current_heading_angle =  np.pi /2 #np.arctan2(self._car_position[1] - trajectory[0, 1], self._car_position[0] - trajectory[0, 0])
            he = desired_heading_angle if self.step > 10 else 0  # ignore the heading error for the first 10 frame => zoom in
        # Calculate the steering angle
        print("speed", speed)
        delta = np.arctan2(self.k * cte, speed + self.k_soft) + he * 0.8
        # Limit the steering angle
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        self.step += 1  # increment the  dstep counter
        return delta
    
    def _calculate_cte(self, trajectory):
        # Calculate the distance to each point on the trajectory
        distances = np.linalg.norm(trajectory - self._car_position, axis=1)

        # Find the index of the lookahead point
        lookahead_distance = 0.0  # adjust this value as needed
        lookahead_index = np.argmin(np.abs(distances + lookahead_distance))

        self.clp = trajectory[lookahead_index]
        self.sclp = trajectory[lookahead_index + 1]


        # SHARP LEFT TRUN AHEAD
        if(self.clp[1] > self._car_position[1] +4 or self.clp[0] > self._car_position[0]+4):
            cte = distances[lookahead_index]
            return cte, lookahead_index, 0

        # SHARP RIGHT TURN AHEAD
        if(self.clp[1] < self._car_position[1] -4 or self.clp[0] < self._car_position[0]-4):
            cte = distances[lookahead_index]
            return cte, lookahead_index, 0


        # Calculate the cross-track error as the distance to the lookahead point
        cte = distances[lookahead_index]

        return cte, lookahead_index, 0

