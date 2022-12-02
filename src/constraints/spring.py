import sys
import os

# Add path of current package (not any higher level in the hierarchy) so as to allow this module to access the "constraints" module.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))


# from demos import demo2
from .constraint import Constraint
import numpy as np
import numpy.typing as npt
import numpy.linalg as la
import constants as const
import helpers as hlp


class Spring(Constraint):
    def __init__(self, k: float, L: float, q: callable, **kwargs):
        '''
        Spring constant that always wants to move a 3d point to a resting location,
        which is a fixed distance away from the other end of the spring.
        k := spring stiffness constant
        L := rest length
        q := a callable that returns a list of tuples. The list is of size 2,
             one tuple for each end of the spring, each tuple of the form `(point#i, index_of_point#i)`.
        kwargs = remaining keyword argument
        '''
        super(Spring, self).__init__(**kwargs)
        self.k = k
        self.w *= k
        self.L = L
        self.q = q

    def project(self):
        '''
        Return the projections of all the vertices involved with this constraint.

        How to compute projections:
        Step 1) Get the center of gravity (CG) of the spring. This CG will also be the CG of the projection.
                NOTE: We will assume equal masses on both ends for now. This may make the simulation
                unstable for unequal masses.
        Step 2) Assign p[i] = CG + unit(q[i] - CG) * L/2 for i in [0, 1].
        Step 3) Return as documented in `Constraint.project()`.
        '''
        q = self.q()
        assert len(q) == 2
        if la.norm(q[0][0] - q[1][0]) < const.EPS:
            raise RuntimeError('undefined [spring is too compressed]')
        CG = np.sum([qi for (qi, idx) in q], axis=0) / len(q)
        p = [(CG + hlp.unit(qi - CG) * self.L / 2, idx) for (qi, idx) in q]
        return p

    def q(self):
        '''
        Placeholder for `self.q()` as documented in `__init__()`.
        '''
        raise NotImplementedError('Spring.q() unimplemented')


if __name__ == '__main__':
    q0 = np.array([0, 0, 0])
    q1 = np.array([0.5, 0, 0])
    c = Spring(k=1, L=1, q=lambda: [(q0, 0), (q1, 1)])
    p = c.project()
    d = la.norm(p[0][0] - p[1][0])
    print(f"p = {p}, d = {d}")
