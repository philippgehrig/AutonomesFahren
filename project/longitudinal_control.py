from __future__ import annotations
import numpy as np


class LongitudinalControl:
    def __init__(self):
        self.Kp = 0.025
        self.Ki = 0.00001
        self.Kd = 0.000015
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, current_speed, target_speed, steer_angle):
        max_angle = 0.392699082
        steer_angle = abs(steer_angle)
        if steer_angle < 0.01: steer_angle = 0
        if steer_angle > max_angle: steer_angle = max_angle
        print(f'steer angle: {steer_angle}')
        # PID-Controller
        error = target_speed - current_speed
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        acceleration = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if steer_angle <= max_angle:
            acceleration = acceleration - acceleration / (64 * max_angle) * steer_angle
        else:
            acceleration = 0.75 * acceleration
        
        
        print(f'acceleration: {acceleration}')
        acceleration = acceleration
        braking = -acceleration

        return acceleration, braking

    # function for test_longitudinal_control.py

    # def predict_target_speed(self, trajectory, speed, steer_angle):
    #     # steer_angle: left = -1, right = 1, else = 0
    #     max_speed = 80.0
    #     min_speed = 35.0
    #     x = trajectory[:, 0]
    #     y = 0

    #     x_mean = np.mean(x)
    #     x_min = np.amin(x)
    #     x_max = np.amax(x)

    #     diff_min = x_mean - x_min
    #     diff_max = x_max - x_mean

    #     if diff_min > 10 or steer_angle == -1:
    #         min_index = np.argmin(trajectory[:, 0])

    #         y = trajectory[min_index, 1]
    #     elif diff_max > 10 or steer_angle == 1:
    #         max_index = np.argmax(trajectory[:, 0])

    #         y = trajectory[max_index, 1]
    #     else:
    #         y = -1

    #     target_speed = 0.0
    #     if y > 0:
    #         # 80 is the width of the modified state_img
    #         target_speed = (min_speed - max_speed) / 80 * y + max_speed
    #     else:
    #         target_speed = max_speed

    #     return target_speed
    
    # function for car.py

    def predict_target_speed(self, curvature):
        max_speed = 80
        min_speed = 35
        max_curvature = 20

        curvature = min(curvature, max_curvature)
        target_speed = max_speed - ((max_speed - min_speed) / max_curvature) * curvature
        return target_speed