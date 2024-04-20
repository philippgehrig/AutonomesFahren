from __future__ import annotations
import numpy as np
import scipy
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

        self.build_lanes(np.array(state_image)[0:80, :])

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

    def build_lanes(self, state_image):

        def check_pixel(idx, coord):
            # idx: Index of white pixel
            # coord: x- and y- values off the last iteration
            part_of_lane = True
            d = coord[-1] - idx
            if abs(d) > 10:
                part_of_lane = False
            return part_of_lane

        left_lane = []
        right_lane = []

        centroid = np.mean(np.where(self.img == 255)[1])
        middle_index = self.img.shape[1] // 2
        diff = centroid - middle_index

        for row in range(self.img.shape[0] - 2, 0, -1):
            white_pixel_indices = np.where(self.img[row] == 255)[0]
            if len(white_pixel_indices) == 2 and abs(diff) < 10:
                left_lane.append((row, white_pixel_indices[0]))
                right_lane.append((row, white_pixel_indices[1]))

                # for debugging only
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]
                state_image[row, white_pixel_indices[1]] = [0, 0, 255]
            elif diff <= -10 and len(white_pixel_indices > 0):
                # left curve
                ref_pixel = white_pixel_indices[-1]
                right = True            # indicates whether pixel belongs to right or left lane
                check_lane = True       # necessary if picture contains more than three lanes

                right_lane.append((row, white_pixel_indices[-1]))
                state_image[row, white_pixel_indices[-1]] = [0, 0, 255]

                for idx in range(len(white_pixel_indices) - 2, 0, -1):
                    distance = ref_pixel - white_pixel_indices[idx]
                    if distance < 5 and right:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif distance < 29 and right:
                        right = False
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif distance < 5 and not right:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif not right and check_lane:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                        check_lane = False
                    elif not right:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                        right = True
                    elif distance >= 29 and right and not check_pixel(white_pixel_indices[idx], left_lane[-1]):
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    else:
                        print('Error Case')
            elif diff >= 10 and len(white_pixel_indices > 0):
                # right curve
                ref_pixel = white_pixel_indices[0]
                left = True
                check_lane = True

                left_lane.append((row, white_pixel_indices[0]))
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]

                for idx in range(1, len(white_pixel_indices)):
                    distance = white_pixel_indices[idx] - ref_pixel
                    if distance < 5 and left:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif distance < 29 and left:
                        left = False
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif distance < 5 and not left:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif not left and check_lane:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                        check_lane = False
                    elif not left:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                        left = True
                    elif distance >= 29 and left and not check_pixel(white_pixel_indices[idx], right_lane[-1]):
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    else:
                        print('Error Case')
            else:
                print('Error case')

        self.img = state_image
