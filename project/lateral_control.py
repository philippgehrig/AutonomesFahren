from __future__ import annotations
import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position_front = np.array([48, 65]) #Postion Front Axle for Bycicle Model
        self._car_position_back = np.array([48, 75]) #Postion Rear Axle for Bycicle Model 
        self.k = 0.3 # control gain
        self.k_soft = 0.8  # softening factor
        self.delta_max = 1  # max steering angle
        self.step = 0  # add a step counter
        self.clp = [0,0]  # closest lookahead point
        self.sclp = [0,0]  # second closest lookahead point
        self.debug = 0 # debug flag
        self.controller = 0 # 0 => Stannley; 1 => Pure Pursuit; 2 => Own Controller Creation

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
        if(self.controller == 0):
            return self.control_stanley(trajectory, speed)
        elif(self.controller == 1):
            return self.control_pure_pursuit(trajectory, speed)
        elif(self.controller == 2):
            return self.own_controller(trajectory, speed)
        else:
            return "Invalid Controller"



    def control_stanley(self, trajectory, speed):
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
            if(self.debug): print("Trajectory = 0") # debug message
            return 0 # car should not steer if no trajectory is found
        
        # Calculate the cross-track error
        # sharpe_turn_flag = 1 => sharp left; 2 => sharp right; 0 => normal
        cte, lookahead_index, sharp_turn_flag = self._calculate_cte(trajectory)
        # Check if the lookahead index is valid
        if(len(trajectory) < lookahead_index + 2):
            if(self.debug): print("Trajectory index out of bounds") #debug message
            return 0 # car should not steer if trajectory index is out of bounds
        
        if(sharp_turn_flag == 1):
            # SHARP LEFT TURN => steer right with 0.3 until normal CLP can be calculated again
            if(self.debug): print("SHARP LEFT")
            return 0.2
        
        elif(sharp_turn_flag == 2):
            # SHARP RIGHT TURN => steer left with -0.3 until normal CLP can be calculated again
            if(self.debug): print("SHARP RIGHT")
            return -0.2    
        
        else:
            desired_heading_angle = np.arctan2(trajectory[lookahead_index + 1, 1] - trajectory[lookahead_index, 1], trajectory[lookahead_index + 1, 0] - trajectory[lookahead_index, 0])    
            current_heading_angle = np.arctan2(self._car_position_front[1] - trajectory[0, 1], self._car_position_front[0] - trajectory[0, 0])
            he = desired_heading_angle - current_heading_angle if self.step > 10 else 0  # ignore the heading error for the first 10 frame => zoom in
        # Calculate the steering angle
        delta = np.arctan2(self.k * cte, speed + self.k_soft) + he
        # Limit the steering angle
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        self.step += 1  # increment the  dstep counter
        if(self.debug): print("DELTA: ", delta)
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
        distances = np.linalg.norm(trajectory - self._car_position_front, axis=1)
        lookahead_index = np.argmin(np.abs(distances))


        self.clp = trajectory[lookahead_index]
        
        # Check if the lookahead index is valid
        if lookahead_index + 1 >= len(trajectory):
            if(self.debug): print("Lookahead index out of bounds") #debug message
            self.clp = [0,0] # set the closest lookahead point to the origin
        else:
            self.clp = trajectory[lookahead_index]
            self.sclp = trajectory[lookahead_index + 1]
            

        

        # SHARP LEFT TRUN AHEAD
        if(self.clp[1] > self._car_position_front[1] +4 or self.clp[0] > self._car_position_front[0]+4):
            cte = distances[lookahead_index]
            return cte, lookahead_index, 1

        # SHARP RIGHT TURN AHEAD
        if(self.clp[1] < self._car_position_front[1] -4 or self.clp[0] < self._car_position_front[0]-4):
            cte = distances[lookahead_index]
            return cte, lookahead_index, 2

        # Calculate the cross-track error as the distance to the lookahead point
        cte = distances[lookahead_index]

        return cte, lookahead_index, 0

    def control_pure_pursuit(self, trajectory, speed):



        """
        Controls the lateral movement of the car based on the given trajectory and speed.
        Not functional at the monent

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
            if(self.debug): print("Trajectory = 0")
            return 0 # car should not steer if no trajectory is found
        
        # find cloesest point on trajectory
        distances = np.linalg.norm(trajectory - self._car_position_back, axis=1)
        closest_index = np.argmin(distances)
        self.clp = trajectory[closest_index]

        # calculate the steering angle
        lookahead_distance = int(round(2, 0))
        if(self.debug): print("Lookahead Distance: ", lookahead_distance)

        if closest_index + lookahead_distance >= len(trajectory):
            lookahead_distance = len(trajectory) - closest_index - 1
        vector = trajectory[closest_index + lookahead_distance] - self._car_position_back

        self.sclp = trajectory[closest_index + lookahead_distance]
        
        desired_heading_angle = np.arctan2(vector[1], vector[0])

        delta = np.clip(desired_heading_angle, -self.delta_max, self.delta_max)

        return delta
    
    def own_controller(self, trajectory, speed):
        """
        Controls the lateral movement of the car based on the given trajectory and speed.
        Not functional at the moment

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
            if(self.debug): print("Trajectory = 0")
            return 0
        
        distances = np.linalg.norm(trajectory - self._car_position_front, axis=1)
        closest_index = np.argmin(distances)
        self.clp = trajectory[closest_index]
        second_closest_index = np.argmin(np.partition(distances, 2)[:2])
        self.sclp = trajectory[second_closest_index]

        vector = self.sclp - self._car_position_front
        delta = np.clip(np.arctan2(vector[1], vector[0]), -self.delta_max, self.delta_max)
        return delta
        
