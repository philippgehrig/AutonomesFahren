from __future__ import annotations
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Das Auto befindet sich in:
# Zeile 67 mit Index 46 bis 49 bis
# Zeile 76 mit Index 46 bis 49
# Beim Lenken können die Räder jeweils einen Pixel weiter aussen stehen

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

        #print(np.shape(self.img))       # original shape of state_img: (96, 96)

        left, right = self.detect_lanes()
        for point in left:
            state_image[point[1], point[0]] = [255, 0, 0]
        for point in right:
            state_image[point[1], point[0]] = [0, 0, 255]

        self.img = state_image

        self.debug_image = self.img
        return left, right

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

    # def detect_curve(self):
    #     centroid = np.mean(np.where(self.img == 255)[1])
    #     middle_index = self.img.shape[1] // 2
    #     diff = centroid - middle_index

    #     idx = np.where(self.img[69] == 255)[0]
    #     if len(idx) > 0:
    #         idx_left = idx[idx <= 46]
    #         idx_right = idx[idx >= 49]

    #         if len(idx_left) > 0:
    #             diff_left = middle_index - idx_left[-1]
    #             diff = diff - 9 + diff_left
    #         elif len(idx_right) > 0:
    #             diff_right = idx_right[0] - middle_index
    #             diff = diff + 9 - diff_right
    #         else:
    #             print('Only car is found!')
    #     else:
    #         print('No lane found!')

    #     # diff < 0: left curve; diff > 0 right curve
    #     return diff

    def detect_lanes(self):

        def euclidean_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        lane_1 = []
        lane_2 = []

        threshold = 5

        white_idx = np.argwhere(self.img == 255)
        if len(white_idx) > 0:
            # Entferne das erste Pixel aus white_idx und füge es zu lane_1 hinzu
            last_white_pxl = white_idx[-1]
            lane_1.append((last_white_pxl[1], last_white_pxl[0]))
            white_idx = white_idx[:-1]

            for ref in lane_1:
                for white_pxl in reversed(white_idx):
                    dist = euclidean_distance((ref[0], ref[1]), (white_pxl[1], white_pxl[0]))
                    if dist < threshold:
                        lane_1.append((white_pxl[1], white_pxl[0]))
                        white_idx = white_idx[np.any(white_idx != white_pxl, axis=1)]

            for idx in white_idx:
                lane_2.append((idx[1], idx[0]))

            # Aufsummieren der x-Werte
            sum_lane_1 = sum(point[0] for point in lane_1)
            sum_lane_2 = sum(point[0] for point in lane_2)

            # Annahme: Die rechte Fahrspur hat die größere Summe
            if sum_lane_1 > sum_lane_2: return lane_2, lane_1
            if sum_lane_2 > sum_lane_1: return lane_1, lane_2

        else:
            print('Now white pixel found!')
            return [], []
