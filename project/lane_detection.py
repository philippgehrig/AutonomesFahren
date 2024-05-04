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

        #self.build_lanes_test(np.array(state_image)[0:80, :])
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

    def detect_curve(self):
        centroid = np.mean(np.where(self.img == 255)[1])
        middle_index = self.img.shape[1] // 2
        diff = centroid - middle_index

        idx = np.where(self.img[69] == 255)[0]
        if len(idx) > 0:
            idx_left = idx[idx <= 46]
            idx_right = idx[idx >= 49]

            if len(idx_left) > 0:
                diff_left = middle_index - idx_left[-1]
                diff = diff - 9 + diff_left
            elif len(idx_right) > 0:
                diff_right = idx_right[0] - middle_index
                diff = diff + 9 - diff_right
            else:
                print('Only car is found!')
        else:
            print('No lane found!')

        # diff < 0: left curve; diff > 0 right curve
        return diff


    def detect_lanes(self):

        def euclidean_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        lane_1 = []
        lane_2 = []

        threshold = 5

        white_idx = np.argwhere(self.img == 255)
        if len(white_idx) > 0:
            # Entferne das erste Pixel aus white_idx und füge es zu lane_1 hinzu
            first_white_pxl = white_idx[0]
            lane_1.append((first_white_pxl[1], first_white_pxl[0]))
            white_idx = white_idx[1:]

            for ref in lane_1:
                for white_pxl in white_idx:
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

        # to be deleted
    def build_lanes_test(self, state_image):

        def check_column(idx, coord):
            # idx: Index of white pixel
            # coord: x- and y- values off the last iteration
            part_of_lane = True
            d = coord[-1] - idx
            if abs(d) > 5:
                part_of_lane = False
            return part_of_lane
        
        def check_row(idx, ref):
            part_of_lane = True
            d = ref - idx
            if abs(d) > 5:
                part_of_lane = False
            return part_of_lane

        left_lane = []
        right_lane = []

        # initial values
        left_lane.append((79, 38))
        right_lane.append((79, 57))

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
                right_lane.append((row, white_pixel_indices[-1]))
                state_image[row, white_pixel_indices[-1]] = [0, 0, 255]

                for idx in range(len(white_pixel_indices) - 2, 0, -1):
                    if check_column(white_pixel_indices[idx], right_lane[-1]) or check_row(white_pixel_indices[idx], ref_pixel):
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif check_column(white_pixel_indices[-1], left_lane[-1]) or not check_row(white_pixel_indices[idx], ref_pixel):
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif check_column(white_pixel_indices[-1], left_lane[-1]) or check_row(white_pixel_indices[idx], ref_pixel):
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    else:
                        print('Error case!')

            elif diff >= 10 and len(white_pixel_indices > 0):
                # right curve
                left_lane.append((row, white_pixel_indices[0]))
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]

                for idx in range(1, len(white_pixel_indices)):
                    if check_column(white_pixel_indices[idx], left_lane[-1]):
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                    elif check_column(white_pixel_indices[-1], right_lane[-1]):
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                    else:
                        print('Error case!')
            else:
                print('Error case')

        self.img = state_image