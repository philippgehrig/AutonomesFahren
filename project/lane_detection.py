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

        #self.build_lanes(np.array(state_image)[0:80, :])
        self.build_lanes_new(np.array(state_image)[0:80, :])

        # left_lane, right_lane = self.build_lanes(self, state_image)
        # print("Left Lane:")
        # for point in left_lane:
        #     print(point)            # format: (x, y)

        # print("Right Lane:")
        # for point in right_lane:
        #     print(point)
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
        left_lane = []
        right_lane = []

        centroid = np.mean(np.where(self.img == 255)[1])
        middle_index = self.img.shape[1] // 2
        diff = centroid - middle_index

        for row in range(self.img.shape[0]):
            white_pixel_indices = np.where(self.img[row] == 255)[0]
            if abs(diff) < 10 and len(white_pixel_indices) == 2:    # starting condition
                left_lane.append((row, white_pixel_indices[0]))
                right_lane.append((row, white_pixel_indices[1]))

                # for debugging only
                print('straight road')
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]
                state_image[row, white_pixel_indices[1]] = [0, 0, 255]
            elif diff <= -10 and len(white_pixel_indices > 0):
                # left curve
                print('left curve')

                ref_pixel = white_pixel_indices[0]
                right_lane.append((row, white_pixel_indices[0]))
                left_lane.append((row, 'NaN'))

                # for debugging only
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]

                for idx in range(1, len(white_pixel_indices)):
                    distance = white_pixel_indices[idx] - ref_pixel
                    if distance < 5:
                        right_lane.append((row, white_pixel_indices[idx]))
                        left_lane.append((row, 'NaN'))
                        ref_pixel = white_pixel_indices[idx]

                        # for debugging only
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                    else:
                        left_lane.append((row, white_pixel_indices[idx]))
                        right_lane.append((row, 'NaN'))

                        # for debugging only
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]

                # if len(white_pixel_indices) == 1:
                #     left_lane.append((row, 'NaN'))
                #     right_lane.append((row, white_pixel_indices[0]))

                #     # for debugging only
                #     state_image[row, white_pixel_indices[0]] = [255, 0, 0]
                # elif len(white_pixel_indices) == 3:
                #     right_lane.append((row, white_pixel_indices[0]))
                #     left_lane.append((row, white_pixel_indices[1]))
                #     right_lane.append((row, white_pixel_indices[2]))

                #     # for debugging only
                #     state_image[row, white_pixel_indices[1]] = [255, 0, 0]
                #     state_image[row, white_pixel_indices[0]] = [0, 0, 255]
                #     state_image[row, white_pixel_indices[2]] = [0, 0, 255]
                # elif len(white_pixel_indices) == 4:
                #     right_lane.append((row, white_pixel_indices[0]))
                #     left_lane.append((row, white_pixel_indices[1]))
                #     left_lane.append((row, white_pixel_indices[2]))
                #     right_lane.append((row, white_pixel_indices[3]))

                #     # for debugging only
                #     state_image[row, white_pixel_indices[1]] = [255, 0, 0]
                #     state_image[row, white_pixel_indices[0]] = [0, 0, 255]
                #     state_image[row, white_pixel_indices[2]] = [255, 0, 0]
                #     state_image[row, white_pixel_indices[3]] = [0, 0, 255]
                # else:
                #     print('more then four white pixels')
            elif diff >= 10 and len(white_pixel_indices > 0):
                # right curve
                print('right curve')

                ref_pixel = white_pixel_indices[0]
                left_lane.append((row, white_pixel_indices[0]))
                right_lane.append((row, 'NaN'))

                state_image[row, white_pixel_indices[0]] = [255, 0, 0]
                for idx in range(1, len(white_pixel_indices)):
                    distance = white_pixel_indices[idx] - ref_pixel
                    if distance < 5:
                        left_lane.append((row, white_pixel_indices[idx]))
                        right_lane.append((row, 'NaN'))
                        ref_pixel = white_pixel_indices[idx]

                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                    else:
                        right_lane.append((row, white_pixel_indices[idx]))
                        left_lane.append((row, 'NaN'))

                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
            else:
                print('Error case')
                print(f'Diff: ', diff)
                print(f'Length: ', len(white_pixel_indices))

            #     if len(white_pixel_indices) == 1:
            #         right_lane.append((row, 'NaN'))
            #         left_lane.append((row, white_pixel_indices[0]))

            #         # for debugging only
            #         state_image[row, white_pixel_indices[0]] = [0, 0, 255]
            #     elif len(white_pixel_indices) == 3:
            #         left_lane.append((row, white_pixel_indices[0]))
            #         right_lane.append((row, white_pixel_indices[1]))
            #         left_lane.append((row, white_pixel_indices[2]))

            #         # for debugging only
            #         state_image[row, white_pixel_indices[1]] = [0, 0, 255]
            #         state_image[row, white_pixel_indices[0]] = [255, 0, 0]
            #         state_image[row, white_pixel_indices[2]] = [255, 0, 0]
            #     elif len(white_pixel_indices) == 4:
            #         left_lane.append((row, white_pixel_indices[0]))
            #         right_lane.append((row, white_pixel_indices[1]))
            #         right_lane.append((row, white_pixel_indices[2]))
            #         left_lane.append((row, white_pixel_indices[3]))

            #         # for debugging only
            #         state_image[row, white_pixel_indices[1]] = [0, 0, 255]
            #         state_image[row, white_pixel_indices[0]] = [255, 0, 0]
            #         state_image[row, white_pixel_indices[2]] = [0, 0, 255]
            #         state_image[row, white_pixel_indices[3]] = [255, 0, 0]
            #     else:
            #         print('more then four white pixels')
            # else:
            #     print('no curve detected')

        #return left_lane, right_lane
        self.img = state_image


    def build_lanes_new(self, state_image):
        left_lane = []
        right_lane = []

        # left_lane.append((79, 38))
        # right_lane.append((79, 57))

        centroid = np.mean(np.where(self.img == 255)[1])
        middle_index = self.img.shape[1] // 2
        diff = centroid - middle_index

        for row in range(self.img.shape[0] - 2, 0, -1):
            white_pixel_indices = np.where(self.img[row] == 255)[0]
            if len(white_pixel_indices) == 2:
                left_lane.append((row, white_pixel_indices[0]))
                right_lane.append((row, white_pixel_indices[1]))

                # for debugging only
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]
                state_image[row, white_pixel_indices[1]] = [0, 0, 255]
            elif diff <= -10 and len(white_pixel_indices > 0):
                # left curve
                ref_pixel = white_pixel_indices[-1]
                right = True
                check = True

                right_lane.append((row, white_pixel_indices[-1]))
                state_image[row, white_pixel_indices[-1]] = [0, 0, 255]

                for idx in range(len(white_pixel_indices) - 2, 0, -1):
                    distance = ref_pixel - white_pixel_indices[idx]
                    if distance < 5 and right:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif right:
                        right = False
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif distance < 5 and not right:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif not right and check:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                        check = False
                    elif not right:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                        right = True
                    else:
                        print('Error Case')
            elif diff >= 10 and len(white_pixel_indices > 0):
                # right curve
                ref_pixel = white_pixel_indices[0]
                left = True
                check = True

                left_lane.append((row, white_pixel_indices[0]))
                state_image[row, white_pixel_indices[0]] = [255, 0, 0]

                for idx in range(1, len(white_pixel_indices)):
                    distance = white_pixel_indices[idx] - ref_pixel
                    if distance < 5 and left:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                    elif left:
                        left = False
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif distance < 5 and not left:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                    elif not left and check:
                        right_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
                        ref_pixel = white_pixel_indices[idx]
                        check = False
                    elif not left:
                        left_lane.append((row, white_pixel_indices[idx]))
                        state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
                        ref_pixel = white_pixel_indices[idx]
                        left = True
                    else:
                        print('Error Case')
            else:
                print('Error case')

        # for row in range(self.img.shape[0] - 2, 0, -1):
        #     white_pixel_indices = np.where(self.img[row] == 255)[0]
        #     if len(white_pixel_indices) == 2:    # starting condition
        #         left_lane.append((row, white_pixel_indices[0]))
        #         right_lane.append((row, white_pixel_indices[1]))

        #         # for debugging only
        #         print('straight road')
        #         state_image[row, white_pixel_indices[0]] = [255, 0, 0]
        #         state_image[row, white_pixel_indices[1]] = [0, 0, 255]
        #     elif abs(diff) >= 10 and len(white_pixel_indices > 0):
        #         # curve detected
        #         print('curve')

        #         ref_left = int(left_lane[-1][1])
        #         ref_right = int(right_lane[-1][1])

        #         for idx in range(1, len(white_pixel_indices)):
        #             distance_left = abs(white_pixel_indices[idx] - ref_left)
        #             distance_right = abs(white_pixel_indices[idx] - ref_right)
        #             print(f'distance: ', distance_left)

        #             if distance_left < distance_right:
        #                 left_lane.append((row, white_pixel_indices[idx]))

        #                 # for debugging only
        #                 state_image[row, white_pixel_indices[idx]] = [255, 0, 0]
        #             elif distance_right < distance_left:
        #                 right_lane.append((row, white_pixel_indices[idx]))

        #                 # for debugging only
        #                 state_image[row, white_pixel_indices[idx]] = [0, 0, 255]
        #             else:
        #                 print('Wrong pixel detected!')
        #     elif diff and len(white_pixel_indices) == 0:
        #         print('No white pixels found!')
        #     else:
        #         print('Error case!')


        self.img = state_image









