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

class LaneDetection:

    def __init__(self):
        self.debug_image = None

    def detect(self, state_image):
        self.img = np.array(state_image)[0:80, :]
        self.toGrayScale()
        self.img = self.edge_detection()
        self.relu()
        lane_1, lane_2, rest = self.area_detection()
        left, right = self.detect_lane_boundaries(lane_1, lane_2)

        # for debugging only

        first_image = np.array(state_image)[0:80, :]
        for point in lane_1:
            first_image[point[1], point[0]] = [255, 0, 0]
        for point in lane_2:
            first_image[point[1], point[0]] = [0, 0, 255]
        for point in rest:
            first_image[point[1], point[0]] = [0, 255, 0]

        second_image = np.array(state_image)[0:80, :]
        for point in left:
            second_image[point[1], point[0]] = [255, 0, 0]
        for point in right:
            second_image[point[1], point[0]] = [0, 0, 255]

        
        # self.img = np.stack((self.img,) * 3, axis=-1)
        # lane_detection_test_image = np.concatenate((self.img, first_image), axis=1)
        # self.debug_image = lane_detection_test_image
        detect_boundries_test_image = np.concatenate((first_image, second_image), axis=1)
        self.debug_image = detect_boundries_test_image

        return left, right
    
    def toGrayScale(self):
        coefficients = np.array([0.2126, 0.7152, 0.0722])
        gray_values = np.dot(self.img, coefficients)
        self.img = gray_values.astype(np.uint8)
        self.isGrayScale = True

    def edge_detection(self):
        # Sobel-Operator für die horizontale Kantenerkennung
        kernel_horizontal = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])

        # Sobel-Operator für die vertikale Kantenerkennung
        kernel_vertical = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]])

        # Führe die Faltung mit den Sobel-Operatoren durch
        smoothed_image = scipy.ndimage.gaussian_filter(self.img, sigma=1)
        edges_horizontal = scipy.signal.convolve2d(smoothed_image, kernel_horizontal, mode='same', boundary='symm')
        edges_vertical = scipy.signal.convolve2d(smoothed_image, kernel_vertical, mode='same', boundary='symm')

        # Kantenstärke berechnen
        edge_strength = np.sqrt(np.square(edges_horizontal) + np.square(edges_vertical))

        return edge_strength
    
    def relu(self):
        threshold = 105
        self.img = np.where(self.img < threshold, 0, 255)
    
    def area_detection(self):
        lane_1 = []
        lane_2 =[]
        rest = []

        values, num_areas = ndimage.label(self.img)
        area_lists = [[] for _ in range(num_areas)]
        for i in range(1, num_areas + 1):
            area_coordinates = np.where(values == i)
            area_lists[i - 1].extend([(x, y) for x, y in zip(area_coordinates[1], area_coordinates[0])])

        sizes = list(map(len, area_lists))
        area_lists_sorted = [x for _, x in sorted(zip(sizes, area_lists), key=lambda pair: pair[0], reverse=True)]

        # Aktuell werden restliche Areas ignoriert, sie können jedoch Teile von den Lanes beonhalten
        # Diese Teile können sowohl positive als auch negative Auswirkung haben
        if len(area_lists_sorted) == 2:
            lane_1 = area_lists_sorted[0]
            lane_2 = area_lists_sorted[1]
        elif len(area_lists_sorted) > 2:
            lane_1 = area_lists_sorted[0]
            lane_2 = area_lists_sorted[1]
            rest = area_lists_sorted[2]
        return lane_1, lane_2, rest
    
    def detect_lane_boundaries(self, lane_1, lane_2):
        if len(lane_1) > 0:
            lane_1_score = sum(point[0] for point in lane_1) / len(lane_1)
        else:
            lane_1_score = 0

        if len(lane_2) > 0:
            lane_2_score = sum(point[0] for point in lane_2) / len(lane_2)
        else:
            lane_2_score = 0

        if lane_1_score and lane_2_score:
            left_lane = lane_1 if lane_1_score < lane_2_score else lane_2
            right_lane = lane_1 if lane_1_score >= lane_2_score else lane_2
        else:
            print('Error: Value of lanes are None!')
            left_lane = []
            right_lane = []

        return left_lane, right_lane


# Old code versions: