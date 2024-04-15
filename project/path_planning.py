from __future__ import annotations
import numpy as np

class PathPlanning:

    def __init__(self):
        self.debug_image = None
        pass

    def plan():
        regression()
        pass

def regression(self):
    # Fit a line to the bottom half of the image
    y = np.arange(40, 80)
    x = np.zeros(40)
    for i in range(40):
        x[i] = np.argmax(self.img[40 + i, :])
    coefficients = np.polyfit(y, x, 2)
    self.poly_func = np.poly1d(coefficients)
    self.img = np.zeros((80, 80))
    for i in range(80):
        # Clip the predicted index to ensure it is within the valid range
        predicted_index = int(self.poly_func(i))
        clipped_index = np.clip(predicted_index, 0, self.img.shape[1] - 1)
        self.img[i, clipped_index] = 255
        