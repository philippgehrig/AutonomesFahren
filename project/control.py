from __future__ import annotations

import argparse

import cv2
import gymnasium as gym
import numpy as np

from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from lane_detection import LaneDetection
from path_planning import PathPlanning
from lateral_control import LateralControl
from longitudinal_control import LongitudinalControl


def run(env, input_controller: InputController):
    lane_detection = LaneDetection()
    lateral_control = LateralControl()
    longitudinal_control = LongitudinalControl()
    path_planning = PathPlanning()

    # Schwierige Seeds: 2(U-Turn), 3(Error), 5(Error), 6(Kurve), 7(Error), 8(Kurve)

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        left_lane_boundaries, right_lane_boundaries = lane_detection.detect(state_image)
        # trajectory, curvature = path_planning.plan(info['left_lane_boundary'], info['right_lane_boundary'])
        trajectory, curvature = path_planning.plan(left_lane_boundaries, right_lane_boundaries)
        steering_angle = lateral_control.control(info['trajectory'], info['speed'])
        # target_speed = longitudinal_control.predict_target_speed(info['trajectory'], info['speed'], steering_angle)
        target_speed = longitudinal_control.predict_target_speed(curvature)
        acceleration, braking = longitudinal_control.control(info['speed'], target_speed, steering_angle)
        
        cv_image = np.asarray(state_image, dtype=np.uint8)
        for point in info['trajectory']:
            if 0 < point[0] < 96 and 0 < point[1] < 84:
                cv_image[int(point[1]), int(point[0])] = [255, 255, 255]
        # add a blue dot on closest lookahead point
        cv_image[int(lateral_control.clp[1]), int(lateral_control.clp[0])] = [0, 0, 255] 
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
        cv2.imshow('Car Racing - Control', cv_image)
        cv2.waitKey(1)

        # Step the environment
        input_controller.update()
        a = [steering_angle, acceleration, braking]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Reset environment if the run is skipped
        input_controller.update()
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")

            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = CarRacingEnvWrapper(gym.make("CarRacing-v2", render_mode=render_mode, domain_randomize=False))
    input_controller = InputController()

    run(env, input_controller)
    env.reset()


if __name__ == '__main__':
    main()
