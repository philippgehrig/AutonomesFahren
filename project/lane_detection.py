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
        self.remove_pixel()
        self.debug_image = self.img
        #print(np.shape(self.img))       # original shape of state_img: (96, 96)
        left_lane, right_lane = self.build_lanes()
        print("Left Lane:")
        for point in left_lane:
            print(point)            # format: (x, y)

        print("Right Lane:")
        for point in right_lane:
            print(point)


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

    def remove_pixel(self):
        # overwrite the pixels of the car
        for i in range(67, 77):
            self.img[i][45] = 0
            self.img[i][46] = 0
            self.img[i][47] = 0
            self.img[i][48] = 0
            self.img[i][49] = 0
            self.img[i][50] = 0
        # overwrite the wrong pixels on the border of the image to prevent calculation errors
        self.img[0, :] = 0
        self.img[-1, :] = 0
        self.img[:, 0] = 0
        self.img[:, -1] = 0


    def build_lanes(self):
        left_lane = []
        right_lane = []

        centroid = np.mean(np.where(self.img == 255)[1])
        middle_index = self.img.shape[1] // 2

        for row in range(self.img.shape[0]):
            white_pixel_indices = np.where(self.img[row] == 255)[0]
            if len(white_pixel_indices) == 2:
                left_lane.append((row, white_pixel_indices[0]))
                right_lane.append((row, white_pixel_indices[1]))
            elif centroid < middle_index:
                # left curve
                if len(white_pixel_indices) == 1:
                    left_lane.append((row, 'NaN'))
                    right_lane.append((row, white_pixel_indices[0]))
                elif len(white_pixel_indices) == 3:
                    right_lane.append((row, white_pixel_indices[0]))
                    left_lane.append((row, white_pixel_indices[1]))
                    right_lane.append((row, white_pixel_indices[2]))
                elif len(white_pixel_indices) == 4:
                    right_lane.append((row, white_pixel_indices[0]))
                    left_lane.append((row, white_pixel_indices[1]))
                    left_lane.append((row, white_pixel_indices[2]))
                    right_lane.append((row, white_pixel_indices[3]))
                else:
                    print('more then four white pixels')
            elif centroid > middle_index:
                # right curve
                if len(white_pixel_indices) == 1:
                    right_lane.append((row, 'NaN'))
                    left_lane.append((row, white_pixel_indices[0]))
                elif len(white_pixel_indices) == 3:
                    left_lane.append((row, white_pixel_indices[0]))
                    right_lane.append((row, white_pixel_indices[1]))
                    left_lane.append((row, white_pixel_indices[2]))
                elif len(white_pixel_indices) == 4:
                    left_lane.append((row, white_pixel_indices[0]))
                    right_lane.append((row, white_pixel_indices[1]))
                    right_lane.append((row, white_pixel_indices[2]))
                    left_lane.append((row, white_pixel_indices[3]))
                else:
                    print('more then four white pixels')
            else:
                print('no curve detected')

        return left_lane, right_lane









