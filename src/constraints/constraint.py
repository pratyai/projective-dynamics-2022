'''
Description of different components and readable reference implementations.
The actual solvers may need more efficient data structures and solutions.
'''

import numpy.typing as npt
import numpy as np
import constants as const


class Constraint:
    '''
    Interface for a general constraint. Specific kind of constraints are to
    implement this interface.
    '''

    def __init__(self,
                 w: float = 1.0,
                 A: npt.NDArray = np.identity(const.D),
                 B: npt.NDArray = np.identity(const.D)):
        self.w = w
        self.A = A
        self.B = B

    def project(self, q: npt.NDArray):
        '''
        "Project" a single point q on the constraint.
        '''
        raise NotImplementedError('Constraint.project() unimplemented')
