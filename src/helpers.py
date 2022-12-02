import numpy as np
import numpy.typing as npt
import numpy.linalg as la


def unit(v: npt.NDArray):
    '''
    Return the unit vector of the non-zero vector v.
    '''
    return v / la.norm(v)
