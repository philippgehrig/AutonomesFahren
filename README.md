# Autonomes Fahren

This repository containts the final project for our course: T3INF4901 Autonomes Fahren TINF22

## Contributors

student number (Philipp Gehrig): 5622763  
student number (Jona Bergmann): 2950692

Team for the final project:

- [Philipp Konstantin Gehrig](https://github.com/philippgehrig)
- [Jona Bergmann](https://github.com/inf22037)

## Module

### Lane Detection

IMPORT PARAMETER:

- state_image: Colour image of the vehicle's surroundings

USED LIBRARIES:

- NUMPY
- SCIPY

OUTPUT PARAMETER:

- left: 2-dimensional Array with the left border
- right: 2-dimensional Array with the right border

The file contains the current code and older code versions to explain the advantages of the current one. The common feature of all versions is the initial transformation of the image into greyscale and the relu() after the convolution.

```python
def toGrayScale(self):
        coefficients = np.array([0.2126, 0.7152, 0.0722])
        gray_values = np.dot(self.img, coefficients)
        self.img = gray_values.astype(np.uint8)
        self.isGrayScale = True

def relu(self):
        threshold = 105
        self.img = np.where(self.img < threshold, 0, 255)       
```
The convolution was initially carried out using a Laplace kernel, which produced white lines at the edges of the lanes, some of which were undercut and only one pixel thick. For more precise edge detection, separate kernels were used for horizontal and vertical detection. In addition, the image is additionally smoothed before convolution.

```python
# Horizontal sobel kernel
kernel_horizontal = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])

# Vertical sobel kernel
kernel_vertical = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

# Convolution with sobel of smoothed image
smoothed_image = scipy.ndimage.gaussian_filter(self.img, sigma=1)
edges_horizontal = scipy.signal.convolve2d(smoothed_image, kernel_horizontal, mode='same', boundary='symm')
edges_vertical = scipy.signal.convolve2d(smoothed_image, kernel_vertical, mode='same', boundary='symm')

# Kantenstärke berechnen
edge_strength = np.sqrt(np.square(edges_horizontal) + np.square(edges_vertical))      
```

The biggest difference between the code versions lies in the assignment of the recognised pixels to the lanes. The original idea was to use the average of the coordinates to determine whether the current image represents a left or right curve:

```python
def detect_curve(self):
    centroid = np.mean(np.where(self.img == 255)[1])
    middle_index = self.img.shape[1] // 2
    diff = centroid - middle_index

    # diff < 0: left curve; diff > 0 right curve
    return diff      
```

Based on the calculated route, it should be possible to judge which side of the road a pixel should belong to. The pixels were assigned line by line using a for loop.
Particularly with sharp curves, it was only possible to assign the pixels to the lanes using nested if-conditions and sometimes only by calculating the Euclidean distance. If the distance is large, it must be a new lane, if it is small, it must be the same lane. However, a lane can occur several times per image line, which often leads to errors in the lane assignment.

The current solution does without the loops and the if conditions. Instead, scipy is used to perform vector-based area recognition, which recognises the related white pixels as objects within the image.

```python
lane_1 = []
lane_2 =[]
rest = []

values, num_areas = ndimage.label(self.img)
area_lists = [[] for _ in range(num_areas)]
for i in range(1, num_areas + 1):
    area_coordinates = np.where(values == i)
    area_lists[i - 1].extend([(x, y) for x, y in zip(area_coordinates[1], area_coordinates[0])])     
```

The recognised areas have a certain number of pixels. The number of pixels is used to recognise whether it is a lane, the vehicle or noise. The two lanes are detected by sorting the objects in descending order.

```python
sizes = list(map(len, area_lists))
area_lists_sorted = [x for _, x in sorted(zip(sizes, area_lists), key=lambda pair: pair[0], reverse=True)]

if len(area_lists_sorted) == 2:
    lane_1 = area_lists_sorted[0]
    lane_2 = area_lists_sorted[1]
elif len(area_lists_sorted) > 2:
    lane_1 = area_lists_sorted[0]
    lane_2 = area_lists_sorted[1]
    rest = area_lists_sorted[2]    
```

Once the two lane boundaries have been recognised, the system determines which boundary is the right-hand boundary and which is the left-hand boundary.
The fact that the average of the x-values of the right-hand lane must be greater than that of the left-hand lane is used for this purpose:

```python
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
```

This type of lane detection is significantly less error-prone than the old code versions, and the vector programming method is also less complex. Finally, the lists containing the coordinates of the lanes in the format (x, y) are formatted in a numpy array.

```python
return np.array(left), np.array(right)   
```

### Path Planning

IMPORT PARAMETER:

- left: 2-dimensional Array with the left border
- right: 2-dimensional Array with the right border
- distance_threshold: maximum distance between middle points for validation of points

USED LIBARIES:

- NUMPY
- SCIPY

OUTPUT PARAMETER:

- valid_path: validated path of points for the car
- curvature: the curvature at the current position of the car

There are 2 different path planning algorithms.

```python
# 1 = normal path planning; 2 = improved target line planning
planing_algorithm = 1
```

This defines which planning algorithm is beeing used.
The improved target line planning is less performant and not working at the moment.

##### Normal Path Planning

In case that either the left, the right boundry or no boundries are not recognized, this if/else - Statement handles these cases and returns an according value so that the game doesn't crash:

```python
if len(left) == 0 or len(right) == 0:
    if len(left):
        return left
    elif len(right):
        return right
    else:
        return []
```

Using the inerp1d-function from scipy both Arrays are interpolated along both axis:

```python
from scipy.interpolate import interp1d

f_left_x = interp1d(np.linspace(0, 1, len(left)), left[:,0], kind='linear', fill_value="extrapolate")
f_left_y = interp1d(np.linspace(0, 1, len(left)), left[:, 1], kind='linear', fill_value="extrapolate")
f_right_x = interp1d(np.linspace(0, 1, len(right)), right[:, 0], kind='linear', fill_value="extrapolate")
f_right_y = interp1d(np.linspace(0, 1, len(right)), right[:, 1], kind='linear', fill_value="extrapolate")
```

Furthermore we need to create a linspace of the maximum amount of points that are on both lines:

```python
import numpy as np

x = np.linspace(0, 1, max(len(left), len(right)))
```

Using the column_stack()-function from numpy, we can recreate a 2-dimensional Array with the linspace

```python
import numpy as np

left_interp = np.column_stack((f_left_x(x), f_left_y(x)))
right_interp = np.column_stack((f_right_x(x), f_right_y(x)))
```

Last but not least, we can divide the left + right Interpolation by 2 to find the middle points

```python
middle_points = (left_interp + right_interp) / 2
```

which we can return as a 2-dimensional array of the path we need to take

#### Improved Target Line Planning

#### Validation

The validation checks if there are single points that are not supposed to exsits. These points might be created due to errors in the Lane Detection Module.

The basic premiss of this function is that no point should be allowed to exsist outside of an euclidian distance of 10 or higher to the next point.

The euclidian distance is calculated using th np.linalg.nrom()-function:

```python
import numpy as np

distances = np.linalg.norm(middle[1:] - middle[:-1], axis=1)
```

The function then creates a boolean mask where each element is True if the corresponding distance in distances is less than or equal to distance_threshold, and False otherwise. This mask is used to filter out the points in middle that are too far apart.

```python
import numpy as np

mask = distances <= distance_threshold
valid_points = middle[np.append(mask, True)]
return valid_points
```

### Longintudal Control

INPUT PARAMETER:

- curvature
- current_speed
- steering_angle

USED LIBRARIES:

- NUMPY

OUTPUT PARAMETER:

- acceleration
- braking

A PID controller is used for longitudinal control:

```python
class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
```

Firstly, the curvature of the lane is used to calculate the target speed of the vehicle. If the curvature is minimal, the speed should be maximum and vice versa. If the value of the curvature is between the minimum and maximum, the speed value should be interpolated linearly:

```python
max_speed = 80
min_speed = 35
max_curvature = 20

curvature = min(curvature, max_curvature)
target_speed = max_speed - ((max_speed - min_speed) / max_curvature) * curvature
```

The control is performed via the difference between the calculated target speed and the actual speed to calculate the acceleration. This is reduced again if there is a significant steering angle:

```python
max_angle = 0.392699082
steer_angle = max(min(abs(steer_angle), max_angle), 0) if steer_angle >= 0.01 else 0
# Not necessary for braking because the car will break in front of the curve
if steer_angle <= max_angle:
    acceleration -= acceleration / (64 * max_angle) * steer_angle
else:
    acceleration *= 0.75
```

The parameterisation of the values - e.g. the maximum steering angle - was based on the values determined from the test runs of the other modules. The parameters for the PID controller were also determined experimentally. Calculating the values according to Ziegler-Nichols did not work because no dynamic oscillation of the current velocity could be set.

```python
self.acceleration_controller = PIDController(0.035, 0.00001, 0.00015)
self.braking_controller = PIDController(0.008, 0.00001, 0.002)
```

The brake controller responds less aggressively to avoid unnecessary heavy braking.

### Lateral Control

Lateral Control has curerently implemented a modified Stanley Contoller.

INPUT PARAMETER:

- trajectory: valid points from path planning
- speed: current speed of the car

INITAL VARIABLES:

- car-position: position of car 2-dimensional array
- k: control gain factor
- k_soft: control softening factor
- delta_max: maximum steering angle
- step: step counter
- clp: closest lookahead point

USED LIBARIES:

- numpy

OUTPUT PARAMETER:

- delta: steering angle as value of -1 (hard left) to +1 (hard right)

First of all the trajectory needs to be validated:

```python
if len(trajectory) == 0:
    print("Trajectory = 0") # debug message
    return 0 # car should not steer if no trajectory is found
```

Once the trajectory has been validated, we can calculate the Cross Track Erorr:

```python
import numpy as np

def _calculate_cte(self, trajectory):
    # Calculate the distance to each point on the trajectory
    distances = np.linalg.norm(trajectory - self._car_position, axis=1)

    # Find the index of the lookahead point
    lookahead_distance = 0.0  # adjust this value as needed
    lookahead_index = np.argmin(np.abs(distances - lookahead_distance))

    self.clp = trajectory[lookahead_index]

    # Calculate the cross-track error as the distance to the lookahead point
    cte = distances[lookahead_index]

    return cte, lookahead_index
```

In this Cross Track-Error is the modification in comparison to the normal Stanley Controller. By adding a lookahead distance we can modify the outputs to some degree. However during testing we realised that 0.0 (which basically removes the modification) works best for the Stanley Controller.

Now the lookahead_index need to be validated, so that it is element of the trajectory:

```python
if(len(trajectory) < lookahead_index + 2):
    print("Trajectory index out of bounds") #debug message
    return 0 # car should not steer if trajectory index is out of bounds
```

Now we can calculate the heading angle the car needs to take in akkording to the Stanley Controller Formula:

```python
import numpy as np

desired_heading_angle = np.arctan2(trajectory[lookahead_index + 1, 1] - trajectory[lookahead_index, 1], trajectory[lookahead_index + 1, 0] - trajectory[lookahead_index, 0])
current_heading_angle = np.arctan2(self._car_position[1] - trajectory[0, 1], self._car_position[0] - trajectory[0, 0])
he = desired_heading_angle - current_heading_angle if self.step > 10 else 0  # ignore the heading error for the first 10 frame => zoom in
```

With the heading angle we can finally calculate delta

```python
import numpy as np

# Calculate the steering angle
delta = np.arctan2(self.k * cte, speed + self.k_soft) + he

# Limit the steering angle
delta = np.clip(delta, -self.delta_max, self.delta_max)
```

## Contribunting

Since this project is graded we unfortunately cannot accept any contributions at the moment!
