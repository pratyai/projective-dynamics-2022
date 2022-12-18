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
import helpers as hlp
import numba as nb


class Discrete(Constraint):
    '''
    This class expresses the discretization of the strain energy on a (triangle) mesh primitive.
    Provided an object at rest position, it is discretized by constructing a mesh of triangles
    that encompasses the whole object's geometry.
    The initial (relative) positions of the vertices of the triangles are stored.
    This defines the initial configuration of any triangle.
    '''

    def __init__(
            self,
            ref: npt.NDArray,
            q: callable,
            sigrange: tuple = (
                1,
                1),
            **kwargs):
        '''
        Discrete strain constraint that always wants to have a certain length,
        which is a fixed distance away from the other end of the spring.
        ref := (3x3) matrix where `ref[i]` is i-th vertex of the triangle at resting config.
        q := a callable that returns a list of tuples. The list is of size 3,
             one tuple for each vertex of the triangle, each tuple of the form `(point#i, index_of_point#i)`.
        sigrange := a 2-tuple for clipping the singular values when projecting.
        kwargs := remaining keyword argument
        '''
        super(Discrete, self).__init__(**kwargs)
        self.ref = ref
        Xg = Discrete.config(ref[0], ref[1], ref[2])
        self.invXg = la.pinv(Xg)
        self.q = q
        self.sigmin, self.sigmax = sigrange

    @staticmethod
    @nb.njit
    def config(p0, p1, p2):
        '''
        Returns the "configuration" of the triangle, i.e. the X_{g/f} matrix from the paper.
        p := a (3x3) matrix, p[i, :] being i-th vertex of the triangle.

        The paper suggests the configuration to be a (2x2) matrix, on the triangle plane.
        Turns out, the (3x3) matrix of [p1 - p0, p2 - p0, 0].T works just fine, since it does give
        a valid transformation, and we're computing SVD and clipping the singular values anyway.
        '''
        X = np.zeros((3, 3))
        X[:, 0] = p1 - p0
        X[:, 1] = p2 - p0
        return X

    def project(self):
        '''
        Return the projections of all the vertices involved with this constraint.
        How to compute projections:
        Step 1) Get the center of gravity (CG) of the triangle. This CG will also be the CG of the projection.
                NOTE: We will assume uniform density for now. This may make the simulation
                unstable for ununiform density.
        Step 2) Compute `U, s, Vh = svd(Xf @ inv(Xg))`.
        Step 3) Compute `T`, the target transformation, by clipping the singular values appropriately.
        Step 4) Compute `p[i] = T @ ref[i]`.
        Step 5) Compute CG for the projection. Move `p[i]` to match the two CGs.
        Return as documented in `Constraint.project()`.
        '''
        q = self.q()
        assert len(q) == 3
        CG = np.sum([qi for (qi, idx) in q], axis=0) / 3
        T = self.T()
        p = [(T @ self.ref[i, :], idx) for (i, (qi, idx)) in enumerate(q)]
        cg = np.sum([pi for (pi, idx) in p], axis=0) / 3
        p = [((CG - cg) + pi, idx) for (pi, idx) in p]
        return p

    @staticmethod
    @nb.njit
    def _T(invXg, sigmin, sigmax, Xf):
        '''
        Find the appropriate transformation `T` that minimises the strain energy.
        '''
        u, s, vh = la.svd(Xf @ invXg)
        s = np.clip(s, sigmin, sigmax)
        S = np.diag(s)
        T = u @ S @ vh
        return T

    def T(self):
        '''
        Find the appropriate transformation `T` that minimises the strain energy.
        '''
        q = self.q()
        Xf = Discrete.config(q[0][0], q[1][0], q[2][0])
        T = Discrete._T(self.invXg, self.sigmin, self.sigmax, Xf)
        return T

    def q(self):
        '''
        Placeholder for `self.q()` as documented in `__init__()`.
        '''
        raise NotImplementedError('Discrete.q() unimplemented')


if __name__ == '__main__':
    ref = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    c = Discrete(ref=ref,
                 q=lambda: [
                     (np.array([0, 0, 0]), 0),
                     (np.array([0, 1, 0]), 1),
                     (np.array([1, 0, 0]), 2)])
    p = np.array([pi for (pi, idx) in c.project()])
    print(f"p = {p}")
