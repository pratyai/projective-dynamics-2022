import numpy as np
import scipy.constants

# dimensions of the geometric space
D = 3

# two real numbers closer than EPS are allowed to be considered the same.
EPS = 1e-6

# Standard acceleration of gravity
gravity_acceleration = np.array([0, 0, scipy.constants.g])
