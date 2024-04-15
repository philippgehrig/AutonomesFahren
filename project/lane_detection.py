from __future__ import annotations
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Das Auto befindet sich in:
# Zeile 67 mit Index 46 bis 49 bis
# Zeile 76 mit Index 46 bis 49

# Fahrbahnbreite: gerade Straße: Pixel 38 und 57


convolution_factor = 1.5

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
        self.find_car()
        #self.calculate_path()
        #self.regression()
        self.debug_image = self.img
        #print(np.shape(self.img))       # original shape of state_img: (96, 96)

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
        result = scipy.signal.convolve2d(imgCopy, kernel, mode='same', boundary='wrap')
    
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
        coefficients = np.polyfit(y, x, 3)
        self.poly_func = np.poly1d(coefficients)
        self.img = np.zeros((80, 80))
        for i in range(80):
            # Clip the predicted index to ensure it is within the valid range
            predicted_index = int(self.poly_func(i))
            clipped_index = np.clip(predicted_index, 0, self.img.shape[1] - 1)
            self.img[i, clipped_index] = 255

    def calculate_path(self):
        path_coordinates = []

        for row in range(self.img.shape[0]):
            white_pixel_indices = np.where(self.img[row] == 255)[0]

            white_sections = np.split(white_pixel_indices, np.where(np.diff(white_pixel_indices) != 1)[0]+1)
            for section in white_sections:
                if len(section) > 0:
                    mean_index = np.mean(section)
                    path_coordinates.append((row, mean_index))

        for coord in path_coordinates:
            row, col = coord
            self.img[row, int(col)] = 250
            self.img[self.img != 250] = 0

    def find_car(self):
        with open("C:/Informatik/car.txt", 'a') as file:
            for row in range(self.img.shape[0]):
                white_pixel_indices = np.where(self.img[row] == 255)[0]
                indices_str = f'Zeile {row}: ' + ' '.join(map(str, white_pixel_indices))
                file.write(indices_str + '\n')







