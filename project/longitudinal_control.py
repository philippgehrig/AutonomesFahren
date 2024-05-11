from __future__ import annotations
import numpy as np


class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


class LongitudinalControl:
    def __init__(self):
        self.acceleration_controller = PIDController(0.035, 0.00001, 0.00015)
        self.braking_controller = PIDController(0.008, 0.00001, 0.002)

    def control(self, current_speed, target_speed, steer_angle):
        max_angle = 0.392699082
        steer_angle = max(min(abs(steer_angle), max_angle), 0) if steer_angle >= 0.01 else 0

        acceleration = self.acceleration_controller.control(target_speed, current_speed)
        braking = self.braking_controller.control(target_speed, current_speed)

        # Not necessary for braking because the car will break in front of the curve
        if steer_angle <= max_angle:
            acceleration -= acceleration / (64 * max_angle) * steer_angle
        else:
            acceleration *= 0.75

        return acceleration, -braking

    def predict_target_speed(self, curvature):
        max_speed = 80
        min_speed = 35
        max_curvature = 20

        curvature = min(curvature, max_curvature)
        target_speed = max_speed - ((max_speed - min_speed) / max_curvature) * curvature
        return target_speed


# Old Code:

# class LongitudinalControl:
#     def __init__(self):
#         self.Kp = 0.035
#         self.Ki = 0.00001
#         self.Kd = 0.00015
#         self.integral = 0.0
#         self.prev_error = 0.0

#     def control(self, current_speed, target_speed, steer_angle):
#         max_angle = 0.392699082
#         steer_angle = max(min(abs(steer_angle), max_angle), 0) if steer_angle >= 0.01 else 0

#         # PID-Controller
#         error = target_speed - current_speed
#         self.integral += error
#         derivative = error - self.prev_error
#         self.prev_error = error

#         acceleration = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

#         if steer_angle <= max_angle:
#             acceleration = acceleration - acceleration / (64 * max_angle) * steer_angle
#         else:
#             acceleration = 0.75 * acceleration
        
#         acceleration = acceleration
#         braking = -acceleration

#         return acceleration, braking

#     def predict_target_speed(self, curvature):
#         max_speed = 80
#         min_speed = 35
#         max_curvature = 20

#         curvature = min(curvature, max_curvature)
#         target_speed = max_speed - ((max_speed - min_speed) / max_curvature) * curvature
#         return target_speed