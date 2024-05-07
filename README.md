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

#### Improved Target Line Planning

### Longintudal Control

### Lateral Control

## Contribunting

Since this project is graded we unfortunately cannot accept any contributions at the moment!
