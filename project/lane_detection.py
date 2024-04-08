from __future__ import annotations
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


convolution_factor = 1

convolutionMatrix = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]])


class LaneDetection:

    def __init__(self):
        self.debug_image = None

    def detect(self, state_image):
        self.img = np.array(state_image)[0:80, :]
        self.toGrayScale()
        self.convolution()
        self.relu()
        print(np.shape(self.img))       # shape of state_img: (96, 80)
        self.regression()
        self.debug_image = self.img

    def toGrayScale(self):
        coefficients = np.array([0.2126, 0.7152, 0.0722])
        gray_values = np.dot(self.img, coefficients)
        self.img = gray_values.astype(np.uint8)
        self.isGrayScale = True

    def convolution(self):
        if not self.isGrayScale:
            raise AttributeError("Image is not in Grayscale, please convert using .toGrayScale()")
    
        imgCopy = np.copy(self.img)
        kernel = convolution_factor * np.array(convolutionMatrix)
    
        # Apply convolution using numpy's correlate function
        result = scipy.signal.correlate2d(imgCopy, kernel, mode='same', boundary='wrap')
    
        # Clip the values to ensure they are within the valid range for pixel values
        imgCopy = np.clip(result, 0, 255)
        self.img = imgCopy.astype(np.uint8)
    
    def relu(self):
        threshold = 105
        self.img = np.where(self.img < threshold, 0, 255)
    
    def regression(self):
        # Fit a line to the bottom half of the image
        y = np.arange(40, 80)
        x = np.zeros(40)
        for i in range(40):
            x[i] = np.argmax(self.img[40 + i, :])
        coefficients = np.polyfit(y, x, 4)
        self.poly_func = np.poly1d(coefficients)
        self.img = np.zeros((80, 80))
        for i in range(80):
            # Clip the predicted index to ensure it is within the valid range
            predicted_index = int(self.poly_func(i))
            clipped_index = np.clip(predicted_index, 0, self.img.shape[1] - 1)
            self.img[i, clipped_index] = 255
            