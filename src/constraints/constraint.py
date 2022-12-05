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

    def project(self):
        '''
        Return a list of tuples. Each tuple is of the form `(projected_point#i, index_of_point#i)`.
        The list has the projections of all the vertices involved with this constraint.
        '''
        raise NotImplementedError('Constraint.project() unimplemented')
