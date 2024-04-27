from __future__ import annotations
import numpy as np


class LongitudinalControl:
    def __init__(self):
        self.Kp = 0.05
        self.Ki = 0.0
        self.Kd = 0.0
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, current_speed, target_speed, steer_angle):
        speed_difference = target_speed - current_speed

        # PID-Controller
        error = speed_difference
        self.integral += error
        derivative = error - self.prev_error

        acceleration = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        acceleration = max(0.0, acceleration)
        braking = max(0.0, -acceleration)

        return acceleration, braking

    def predict_target_speed(self, trajectory, speed, steer_angle):
        # steer_angle: left = -1, right = 1, else = 0
        max_speed = 50.0
        min_speed = 5.0
        x = trajectory[:, 0]
        y = 0

        x_mean = np.mean(x)
        x_min = np.amin(x)
        x_max = np.amax(x)

        diff_min = x_mean - x_min
        diff_max = x_max - x_mean

        if diff_min > 10 or steer_angle == -1:
            min_index = np.argmin(trajectory[:, 0])

            y = trajectory[min_index, 1]
        elif diff_max > 10 or steer_angle == 1:
            max_index = np.argmax(trajectory[:, 0])

            y = trajectory[max_index, 1]
        else:
            y = -1

        target_speed = 0.0
        if y >= 0:
            # 80 is the width of the modified state_img
            target_speed = (min_speed - max_speed) / 80 * y + max_speed
        elif steer_angle is not 0:
            target_speed = speed / 2
        else:
            target_speed = max_speed

        return target_speed
    
