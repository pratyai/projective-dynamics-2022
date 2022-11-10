import constraint as con
import numpy as np
import numpy.typing as npt
import numpy.linalg as la

EPS = 1e-6


class Spring(con.Constraint):
    def __init__(self, k: float, L: float, p0: callable, **kwargs):
        '''
        Spring constant that always wants to move a 3d point to a resting location,
        which is a fixed distance away from the other end of the spring.
        k = spring constant
        L = rest length
        p0 = a callable that returns the resting location
        kwargs = remaining keyword argument
        '''
        super(Spring, self).__init__(**kwargs)
        self.k = k
        self.w *= k
        self.L = L
        self.p0 = p0

    def project(self, q: npt.NDArray):
        '''
        "Project" a single point q on the constraint.
        '''
        p0, p0_idx = self.p0()
        d = q - p0
        if la.norm(d) < EPS:
            raise RuntimeError('undefined [spring is too compressed]')
        u = d / la.norm(d)
        return p0 + u * self.L

    def p0(self):
        '''
        Find the other end of the spring.
        Since the other end is not necessarily fixed, and will typically refer
        to the current poision of another vertex, this function needs to be constructed dynamically.
        '''
        raise RuntimeError('Spring.p0() unimplemented')


if __name__ == '__main__':
    c = Spring(k=1, L=1, p0=lambda: np.array([0, 0, 0]))
    p = np.array([0.5, 0, 0])
    q = c.project(p)
    print(f"q= {q}")
