# Autonomes Fahren

This repository containts the final project for our course: T3INF4901 Autonomes Fahren TINF22

## Contriburors

student number (Philipp Gehrig): 5622763  
student number (Jona Bergmann):

Team for the final project:

- [Philipp Konstantin Gehrig](https://github.com/philippgehrig)
- [Jona Bergmann](https://github.com/inf22037)

## Module

### Lane Detection

### Path Planning

IMPORT PARAMETER:

- left: 2-dimensional Array with the left border
- right: 2-dimensional Array with the right border
- distance_threshold: maximum distance between middle points for validation of points

USED LIBARYS:

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

### Lateral Control

## Contribunting

Since this project is graded we unfortunately cannot accept any contributions at the moment!
