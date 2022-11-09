'''
Description of different components and readable reference implementations.
The actual solvers may need more efficient data structures and solutions.
'''

import numpy.typing as npt
from numpy import array, identity


class Constraint:
    '''
    Interface for a general constraint. Specific kind of constraints are to
    implement this interface.
    '''

    # dimensions of the geometric space
    D = 3

    def __init__(self, w: float = 1.0, A: array = identity(D), B: array = identity(D)):
        self.w = w
        self.A = A
        self.B = B

    def project(self, q: npt.NDArray):
        '''
        "Project" a single point q on the constraint.
        '''
        raise RuntimeError('Constraint.project() unimplemented')
