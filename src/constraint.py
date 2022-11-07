'''
Description of different components and readable reference implementations.
The actual solvers may need more efficient data structures and solutions.
'''

import numpy.typing as npt


class Constraint:
    '''
    Interface for a general constraint. Specific kind of constraints are to
    implement this interface.
    '''

    # dimensions of the geometric space
    D = 3

    def wAA(self):
        '''
        Return w_i * A_i' * A_i for this constraint, excluding the selection term S_i.
        '''
        raise RuntimeError('Constraint.wAA() unimplemented')

    def wABp(self, q: npt.NDArray):
        '''
        Return w_i * A_i' * B_i * p for this constraing, excluding the selection term S_i.
        q = a single point we want to project on this constraint.
        '''
        raise RuntimeError('Constraint.wABp() unimplemented')
