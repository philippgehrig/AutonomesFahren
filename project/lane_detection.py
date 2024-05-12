from __future__ import annotations
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.interpolate import interp1d

class LaneDetection:

    def __init__(self):
        self.debug_image = None
        self.debug_flag = 0

    def detect(self, state_image):
        self.img = np.array(state_image)[0:84, :]
        self.toGrayScale()
        self.img = self.edge_detection()
        self.relu()
        lanes = self.area_detection()
        left, right = self.detect_lane_boundaries(lanes)
        # left = self.thin_out_lines(left)
        # right = self.thin_out_lines(right)

        # for debugging only

        if self.debug_flag == 1:     # test image for lane detection
            self.img = np.stack((self.img,) * 3, axis=-1)
            first_image = np.array(state_image)[0:84, :]
            test_image = np.concatenate((self.img, first_image), axis=1)
            self.debug_image = test_image
        elif self.debug_flag == 2:   # test image for boundry detection
            first_image = np.array(state_image)[0:84, :]
            i = 0
            for lane in lanes:
                for point in lane:
                    if i == 0:
                        first_image[point[1], point[0]] = [255, 0, 0]
                    elif i == 1:
                        first_image[point[1], point[0]] = [0, 0, 255]
                    elif i == 2:
                        first_image[point[1], point[0]] = [0, 255, 0]
                    elif i == 3:
                        first_image[point[1], point[0]] = [255, 255, 0]
                    else:
                        first_image[point[1], point[0]] = [0, 255, 255]
                i += 1

            second_image = np.array(state_image)[0:84, :]
            for point in left:
                second_image[point[1], point[0]] = [255, 0, 0]
            for point in right:
                second_image[point[1], point[0]] = [0, 0, 255]
        
            test_image = np.concatenate((first_image, second_image), axis=1)
            self.debug_image = test_image
        else:
            self.debug_image = state_image

        return np.array(left), np.array(right)
    
    def toGrayScale(self):
        coefficients = np.array([0.2126, 0.7152, 0.0722])
        gray_values = np.dot(self.img, coefficients)
        self.img = gray_values.astype(np.uint8)
        self.isGrayScale = True

    def edge_detection(self):
        if not self.isGrayScale:
            raise AttributeError("Image is not in Grayscale, please convert using .toGrayScale()")
        
        # Horizontal sobel kernel
        kernel_horizontal = np.array([[1, 1, 1],
                                      [0, 0, 0],
                                      [-1, -1, -1]])

        # Vertical sobel kernel
        kernel_vertical = np.array([[1, 0, -1],
                                    [1, 0, -1],
                                    [1, 0, -1]])

        # Convolution with sobel of the image     
        edges_horizontal = scipy.signal.convolve2d(self.img, kernel_horizontal, mode='same', boundary='symm')
        edges_vertical = scipy.signal.convolve2d(self.img, kernel_vertical, mode='same', boundary='symm')
        
        # Kantenstärke berechnen
        edge_strength = np.sqrt(np.square(edges_horizontal) + np.square(edges_vertical))
        return edge_strength
    
    def relu(self):
        threshold = 130
        self.img = np.where(self.img < threshold, 0, 255)  
    
    def area_detection(self):
        values, num_areas = ndimage.label(self.img)
        area_lists = [[] for _ in range(num_areas)]
        for i in range(1, num_areas + 1):
            area_coordinates = np.where(values == i)
            area_lists[i - 1].extend([(x, y) for x, y in zip(area_coordinates[1], area_coordinates[0])])

        sizes = list(map(len, area_lists))
        area_lists_sorted = [x for _, x in sorted(zip(sizes, area_lists), key=lambda pair: pair[0], reverse=True)]

        return area_lists_sorted
    
    def detect_lane_boundaries(self, lanes):
        score_lists = [[] for _ in range(len(lanes))]
        num_lanes = 0
        for i in range(0, len(lanes)):
        # Avoid dividing through zero, also the car is estimated to less than 75 pixels
            if len(lanes[i]) > 75:
                score_lists[i] = sum(point[0] for point in lanes[i]) / len(lanes[i])
                num_lanes += 1
            else:
                score_lists[i] = 0
        
        # The higher the score, the more right is the lane
        sorted_lanes = [x for _, x in sorted(zip(score_lists, lanes), reverse=True)]

        if num_lanes == 0:
            if self.debug_flag: print('Error: Value of lanes are 0 or None!')
            return [], []
        elif num_lanes == 2:
            return sorted_lanes[1], sorted_lanes[0]
        elif num_lanes == 3:
            left_lane = sorted_lanes[1] + sorted_lanes[2]
            right_lane = sorted_lanes[0]
            return left_lane, right_lane
        elif num_lanes == 4:
            left_lane = sorted_lanes[1] + sorted_lanes[2]
            right_lane = sorted_lanes[0] + sorted_lanes[3]
            return left_lane, right_lane
        else:
            return [], []
    
    def thin_out_lines(self, lane):
        # A lane can have a maximum of 2 pixels
        new_lane = []

        y_value = None
        x_value = None
        
        if len(lane) > 0:
            x_value = lane[0][0]
            y_value = lane[0][1]
            new_lane.append((x_value, y_value))
            for x, y in lane[1:]:
                diff = x - x_value

                if y == y_value and diff > 15:
                    new_lane.append((x, y))
                elif y != y_value:
                    x_value = x
                    y_value = y
                    new_lane.append((x_value, y_value))
                else:
                    pass
        else:
            if self.debug_flag: print('No lane found')

        return new_lane



# Old code versions:


# # Das Auto befindet sich in:
# # Zeile 67 mit Index 46 bis 49 bis
# # Zeile 76 mit Index 46 bis 49
# # Beim Lenken können die Räder jeweils einen Pixel weiter aussen stehen

# # Fahrbahnbreite: gerade Straße: Pixel 38 und 57

# convolution_factor = 1.5

# convolutionMatrix = np.array([
#     [1, 1, 1],
#     [1, -8, 1],
#     [1, 1, 1]])
# class LaneDetection:

#     def __init__(self):
#         self.debug_image = None

#     def detect(self, state_image):
#         self.img = np.array(state_image)[0:80, :]
#         self.toGrayScale()
#         self.convolution()
#         self.relu()
#         self.remove_pixel()

#         #print(np.shape(self.img))       # original shape of state_img: (96, 96)

#         left, right = self.detect_lanes()
#         for point in left:
#             state_image[point[1], point[0]] = [255, 0, 0]
#         for point in right:
#             state_image[point[1], point[0]] = [0, 0, 255]

#         self.img = state_image

#         self.debug_image = self.img
#         return left, right

#     def toGrayScale(self):
#         coefficients = np.array([0.2126, 0.7152, 0.0722])
#         gray_values = np.dot(self.img, coefficients)
#         self.img = gray_values.astype(np.uint8)
#         self.isGrayScale = True

#     def convolution(self):
#         if not self.isGrayScale:
#             raise AttributeError("Image is not in Grayscale, please convert using .toGrayScale()")
    
#         imgCopy = np.copy(self.img)
#         kernel = convolution_factor * np.array(convolutionMatrix)
    
#         # Apply convolution using numpy's correlate function
#         result = scipy.signal.convolve2d(imgCopy, kernel, mode='same', boundary='wrap')
    
#         # Clip the values to ensure they are within the valid range for pixel values
#         imgCopy = np.clip(result, 0, 255)
#         self.img = imgCopy.astype(np.uint8)
    
#     def relu(self):
#         threshold = 105
#         self.img = np.where(self.img < threshold, 0, 255)

#     def remove_pixel(self):
#         # overwrite the pixels of the car
#         for i in range(67, 77):
#             self.img[i][45] = 0
#             self.img[i][46] = 0
#             self.img[i][47] = 0
#             self.img[i][48] = 0
#             self.img[i][49] = 0
#             self.img[i][50] = 0
#         # overwrite the wrong pixels on the border of the image to prevent calculation errors
#         self.img[0, :] = 0
#         self.img[-1, :] = 0
#         self.img[:, 0] = 0
#         self.img[:, -1] = 0

#     # def detect_curve(self):
#     #     centroid = np.mean(np.where(self.img == 255)[1])
#     #     middle_index = self.img.shape[1] // 2
#     #     diff = centroid - middle_index

#     #     idx = np.where(self.img[69] == 255)[0]
#     #     if len(idx) > 0:
#     #         idx_left = idx[idx <= 46]
#     #         idx_right = idx[idx >= 49]

#     #         if len(idx_left) > 0:
#     #             diff_left = middle_index - idx_left[-1]
#     #             diff = diff - 9 + diff_left
#     #         elif len(idx_right) > 0:
#     #             diff_right = idx_right[0] - middle_index
#     #             diff = diff + 9 - diff_right
#     #         else:
#     #             print('Only car is found!')
#     #     else:
#     #         print('No lane found!')

#     #     # diff < 0: left curve; diff > 0 right curve
#     #     return diff

#     def detect_lanes(self):

#         def euclidean_distance(point1, point2):
#             return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

#         lane_1 = []
#         lane_2 = []

#         threshold = 5

#         white_idx = np.argwhere(self.img == 255)
#         if len(white_idx) > 0:
#             # Entferne das erste Pixel aus white_idx und füge es zu lane_1 hinzu
#             last_white_pxl = white_idx[-1]
#             lane_1.append((last_white_pxl[1], last_white_pxl[0]))
#             white_idx = white_idx[:-1]

#             for ref in lane_1:
#                 for white_pxl in reversed(white_idx):
#                     dist = euclidean_distance((ref[0], ref[1]), (white_pxl[1], white_pxl[0]))
#                     if dist < threshold:
#                         lane_1.append((white_pxl[1], white_pxl[0]))
#                         white_idx = white_idx[np.any(white_idx != white_pxl, axis=1)]

#             lane_2 = [(idx[1], idx[0]) for idx in white_idx]

#             # Aufsummieren der x-Werte
#             sum_lane_1 = sum(point[0] for point in lane_1)
#             sum_lane_2 = sum(point[0] for point in lane_2)

#             # Annahme: Die rechte Fahrspur hat die größere Summe
#             if sum_lane_1 > sum_lane_2: return lane_2, lane_1
#             if sum_lane_2 > sum_lane_1: return lane_1, lane_2

#         else:
#             print('No white pixel found!')
#             return [], []

        
#     # Old function for lane detection
        
#     def build_lanes(self, state_image):

#         # state_image is used for debugging

#         def check_column(idx, coord):
#             # idx: Index of white pixel
#             # coord: x- and y- values off the last iteration
#             part_of_lane = True
#             d = coord[-1] - idx
#             if abs(d) > 5:
#                 part_of_lane = False
#             return part_of_lane
        
#         def check_row(idx, ref):
#             part_of_lane = True
#             d = ref - idx
#             if abs(d) > 5:
#                 part_of_lane = False
#             return part_of_lane

#         left_lane = []
#         right_lane = []

#         # initial values
#         left_lane.append((79, 38))
#         right_lane.append((79, 57))

#         centroid = np.mean(np.where(self.img == 255)[1])
#         middle_index = self.img.shape[1] // 2
#         diff = centroid - middle_index

#         for row in range(self.img.shape[0] - 2, 0, -1):
#             white_pixel_indices = np.where(self.img[row] == 255)[0]
#             if len(white_pixel_indices) == 2 and abs(diff) < 10:
#                 left_lane.append((row, white_pixel_indices[0]))
#                 right_lane.append((row, white_pixel_indices[1]))

#                 # for debugging only
#                 state_image[row, white_pixel_indices[0]] = [255, 0, 0]
#                 state_image[row, white_pixel_indices[1]] = [0, 0, 255]
#             elif diff <= -10 and len(white_pixel_indices > 0):
#                 # left curve
#                 ref_pixel = white_pixel_indices[-1]
#                 right_lane.append((row, white_pixel_indices[-1]))
#                 state_image[row, white_pixel_indices[-1]] = [0, 0, 255]

#                 for idx in range(len(white_pixel_indices) - 2, 0, -1):
#                     if check_column(white_pixel_indices[idx], right_lane[-1]) or check_row(white_pixel_indices[idx], ref_pixel):
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif check_column(white_pixel_indices[-1], left_lane[-1]) or not check_row(white_pixel_indices[idx], ref_pixel):
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif check_column(white_pixel_indices[-1], left_lane[-1]) or check_row(white_pixel_indices[idx], ref_pixel):
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                     else:
#                         print('Error case!')

#             elif diff >= 10 and len(white_pixel_indices > 0):
#                 # right curve
#                 left_lane.append((row, white_pixel_indices[0]))
#                 state_image[row, white_pixel_indices[0]] = [255, 0, 0]

#                 for idx in range(1, len(white_pixel_indices)):
#                     if check_column(white_pixel_indices[idx], left_lane[-1]):
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                     elif check_column(white_pixel_indices[-1], right_lane[-1]):
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                     else:
#                         print('Error case!')
#             else:
#                 print('Error case')

#         self.img = state_image
#         return left_lane, right_lane
    
#     # first implemented lane detection

#     def build_lanes_initial(self, state_image):

#         def check_pixel(idx, coord):
#             # idx: Index of white pixel
#             # coord: x- and y- values off the last iteration
#             part_of_lane = True
#             d = coord[-1] - idx
#             if abs(d) > 10:
#                 part_of_lane = False
#             return part_of_lane

#         left_lane = []
#         right_lane = []

#         centroid = np.mean(np.where(self.img == 255)[1])
#         middle_index = self.img.shape[1] // 2
#         diff = centroid - middle_index

#         for row in range(self.img.shape[0] - 2, 0, -1):
#             white_pixel_indices = np.where(self.img[row] == 255)[0]
#             if len(white_pixel_indices) == 2 and abs(diff) < 10:
#                 left_lane.append((row, white_pixel_indices[0]))
#                 right_lane.append((row, white_pixel_indices[1]))

#                 # for debugging only
#                 state_image[row, white_pixel_indices[0]] = [255, 0, 0]
#                 state_image[row, white_pixel_indices[1]] = [0, 0, 255]
#             elif diff <= -10 and len(white_pixel_indices > 0):
#                 # left curve
#                 ref_pixel = white_pixel_indices[-1]
#                 right = True            # indicates whether pixel belongs to right or left lane
#                 check_lane = True       # necessary if picture contains more than three lanes

#                 right_lane.append((row, white_pixel_indices[-1]))
#                 state_image[row, white_pixel_indices[-1]] = [0, 0, 255]

#                 for idx in range(len(white_pixel_indices) - 2, 0, -1):
#                     distance = ref_pixel - white_pixel_indices[idx]
#                     if distance < 5 and right:
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif distance < 29 and right:
#                         right = False
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif distance < 5 and not right:
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif not right and check_lane:
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                         check_lane = False
#                     elif not right:
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                         right = True
#                     elif distance >= 29 and right and not check_pixel(white_pixel_indices[idx], left_lane[-1]):
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                     else:
#                         print('Error Case')
#             elif diff >= 10 and len(white_pixel_indices > 0):
#                 # right curve
#                 ref_pixel = white_pixel_indices[0]
#                 left = True
#                 check_lane = True

#                 left_lane.append((row, white_pixel_indices[0]))
#                 state_image[row, white_pixel_indices[0]] = [255, 0, 0]

#                 for idx in range(1, len(white_pixel_indices)):
#                     distance = white_pixel_indices[idx] - ref_pixel
#                     if distance < 5 and left:
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif distance < 29 and left:
#                         left = False
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif distance < 5 and not left:
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                     elif not left and check_lane:
#                         right_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
#                         ref_pixel = white_pixel_indices[idx]
#                         check_lane = False
#                     elif not left:
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                         left = True
#                     elif distance >= 29 and left and not check_pixel(white_pixel_indices[idx], right_lane[-1]):
#                         left_lane.append((row, white_pixel_indices[idx]))
#                         state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
#                         ref_pixel = white_pixel_indices[idx]
#                     else:
#                         print('Error Case')
#             else:
#                 print('Error case')

#         self.img = state_image
#         return left_lane, right_lane