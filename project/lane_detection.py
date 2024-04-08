from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


convolution_factor = -0.4

convolutionMatrix = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]])
smoothingMatrix = [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9]
]

class LaneDetection:

    def __init__(self):
        self.debug_image = None

    def detect(self, state_image):
        self.img = state_image
        self.toGrayScale()
        self.convolution()
        self.relu()
        self.debug_image = self.img
        print(self.img)

    def convolution(self):
        if not self.isGrayScale:
            raise AttributeError("Image is not in Grayscale, please convert using .toGrayScale()")
        imgCopy = np.copy(self.img)
        for x in range(self.img.shape[0]):
            for y in range(self.img.shape[1]):
                gray = multiplyConvolutionMatrix(imgCopy, (x, y), convolution_factor * convolutionMatrix)
                imgCopy[y][x] = int(gray)
#                imgCopy.putpixel((x, y), int(gray))
        self.img = imgCopy

    def toGrayScale(self):
        #print(self.img.shape)
        #self.img = ImageOps.grayscale(self.img)
        imgCopy = np.zeros((self.img.shape[0],self.img.shape[1]))
        for x in range(self.img.shape[1]):
            for y in range(self.img.shape[0]):
                #print(getPixel(self.img,x,y))
                imgCopy[y][x] = 0.2126*getPixel(self.img,x,y)[0]+0.7152*getPixel(self.img,x,y)[1]+0.0722*getPixel(self.img,x,y)[2]
        self.img = imgCopy
        self.isGrayScale = True
        pass
    
    def relu(self):
        for x in range(self.img.shape[0]):
            for y in range(self.img.shape[1]):
                p = getPixel(self.img,x,y)
                self.img[y][x] = 0 if p<105 else 255

    def smooth(self):
        if self.isGrayScale:
            raise AttributeError("Image is Grayscale but should be colored")
        imgCopy = np.copy(self.img)
        for x in range(self.img.shape[0]):
            for y in range(self.img.shape[1]):
                colors = multiplyMatrix(imgCopy, (x, y), smoothingMatrix)
                imgCopy[y][x] = colors
        self.img = imgCopy

def multiplyConvolutionMatrix(img: Image, pos: tuple[int, int], matrix) -> float:
    gray = 0
    l = [-1, 0, 1]
    for x_ in l:
        for y_ in l:
            px = pos[0] + x_
            py = pos[1] + y_
            if not (0 < px < img.shape[0] and 0 < py < img.shape[1]):
                continue
            gray += matrix[l.index(y_)][l.index(x_)] * getPixel(img,px,py)
    return (gray / 2) + 128

def multiplyMatrix(img: Image, pos: tuple[int, int], matrix) -> tuple[int]:
    colors = [0, 0, 0]
    l = [-1, 0, 1]
    for x_ in l:
        for y_ in l:
            px = pos[0] + x_
            py = pos[1] + y_
            if not (0 < px < img.shape[0] and 0 < py < img.shape[1]):
                continue
            pixel = getPixel(img,px,py)
            for i in range(3):
                colors[i] += matrix[l.index(y_)][l.index(x_)] * pixel[i]
    colors = tuple(int(v) for v in colors)
    return colors

def getPixel(arr, x, y):
    return arr[y][x]

def find_car_indices(self):
    # start in the middle of the array to look for the car
    mid_x_idx = self.shape[1] // 2
    mid_y_idx = self.shape[0] // 2

    car_idx = None
    mid_y_below = np.arange(mid_y_idx, self.shape[0])               # create an array with all lines of self below the middle
    car_y_idx = np.where(self[mid_y_below, mid_x_idx] == 0)[0]      # find the indices of the black pixels
    if car_y_idx.size > 0:
        car_idx = mid_y_idx + car_y_idx[0]

    return car_idx         # is the position of the car always the same?

def find_line_indices(self):
    # start in the middle of the array to look for both lines
    mid_x_idx = self.shape[1] // 2
    mid_y_idx = self.shape[0] // 2

    # search for the first black pixel on the left
    left_x_idx = mid_x_idx
    left_row = self[mid_y_idx, :mid_x_idx] == 0                 # converting the left half of the row to boolean: 0 -> True
    if np.any(left_row):                                        # np.any() checks if row has any black pixels
        left_x_idx = mid_x_idx - np.argmax(left_row[::-1])      # np.argmax() looks for the first True starting in the middle

    right_x_idx = mid_x_idx
    right_row = self[mid_y_idx, mid_x_idx:] == 0
    if np.any(right_row):
        right_x_idx = mid_x_idx + np.argmax(right_row)

    return left_x_idx, right_x_idx

def linespace(self, left, right):
    if left is not None and right is not None:
        mid = (left + right) // 2
        mid_val = 0
        left_val = -1
        right_val = 1
        interp_val = np.linspace(left_val, right_val, right - left + 1)
        self[mid, left:right + 1] = np.around(np.interp(np.arange(left, right + 1),[left, right], [left_val, right_val]))