import constraint as con
import spring as spring
import numpy as np
import numpy.typing as npt
import scipy.linalg as la


class System:
    # dimensions of the geometric space
    D = 3

    def __init__(self,
                 q: npt.NDArray,
                 q1: npt.NDArray,
                 M: npt.NDArray):
        '''
        A system of discretiztion points, their states and their constraints.
        q = initial positions
        q1 = initial first derivatives of q (i.e. velocity)
        M = mass matrix
        cons = list of constraint associated with each q
        pinned = set of vertices (referred to by indices) to be pinned in position
        '''
        self.n = q.shape[0]

        self.q = q
        if q1 is None:
            q1 = np.zeros(q.shape)
        self.q1 = q1
        self.M = M
        self.cons = [[] for i in range(self.n)]
        self.pinned = set()

        # sanity checks
        assert q.shape == (self.n, System.D)
        assert q1.shape == (self.n, System.D)
        assert M.shape == (self.n * System.D, self.n * System.D)
        assert len(self.cons) == self.n

    def add_spring(self, k: float, L: float, q_idx: int, p0_idx: int):
        '''
        Construct and add a spring constraint to the system with the given parameters.
        The indices refer to the vertices already present in the system.
        '''
        assert q_idx != p0_idx
        c = spring.Spring(k, L, p0=lambda: (self.q[p0_idx], p0_idx))
        self.cons[q_idx].append(c)

    def f_ext(self):
        '''
        Returns external forces applied to each vertex from outside the system.
        TODO: forces need to be parameterized instead of hardcoded.
        '''
        gravity_force = np.array([np.array([0, 0, -0.1])] * self.q.shape[0])
        damping_force = -0.5*self.q1
        return gravity_force

    def wAtA(self):
        wAtA = [
            np.sum([c.w * c.A.T @ c.A for c in self.cons[i]], axis=0)
            if self.cons[i] else np.zeros((System.D, System.D))
            for i in range(self.n)
        ]
        # a giant matrix for all the points arranged in a block diagonal fashion,
        # since the different points' computations are independent (unless it
        # goes through a constraint).
        return la.block_diag(*wAtA)

    def wAtBp(self):
        wAtBp = [
            np.sum([c.w * c.A.T @ c.B @ c.project(self.q[i])
                   for c in self.cons[i]], axis=0)
            if self.cons[i] else np.zeros(System.D)
            for i in range(self.n)
        ]
        # a giant stack of all the points' column vectors.
        return np.array(wAtBp)

    def solve(self, h: float):
        '''
        Return the new q and q1 after a step of size h.
        h = step size
        '''
        # NOTE: All the local solutions happen inside `c.project()`.

        # All the selection matrices were ignored using the block-diagonal constructions.
        # We know that constraints are not shared between vertices in our current model.
        wAtA = self.wAtA()
        wAtBp = self.wAtBp()

        M_h2 = self.M / (h*h)
        s = self.q + h * self.q1 + \
            (la.inv(M_h2) @ self.f_ext().reshape(-1)).reshape(self.q.shape)

        lhs = M_h2 + wAtA
        rhs = M_h2 @ s.reshape(-1) + wAtBp.reshape(-1)

        q = la.solve(lhs, rhs).reshape(self.q.shape)
        # pin these vertices
        q[list(self.pinned), :] = self.q[list(self.pinned), :]
        q1 = (q - self.q) / h
        return (q, q1)

    def step(self, h: float):
        self.q, self.q1 = self.solve(h)


if __name__ == '__main__':
    # all the vertices in the system
    q = np.array([[0, 0, 0], [2.5, 0, 0]])

    s = System(q=q, q1=None, M=np.kron(
        np.identity(q.shape[0]), np.identity(System.D)))
    s.add_spring(k=1, L=1, q_idx=1, p0_idx=0)

    for i in range(100):
        s.step(0.01)
        print(f"{i}: p={s.q[1]} v={s.q1[1]}")
