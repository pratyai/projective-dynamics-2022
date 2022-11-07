import constraint as con
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
        c = con.Spring(k, L, p0=lambda: self.q[p0_idx])
        self.cons[q_idx].append(c)

    def f_ext(self):
        '''
        Returns external forces applied to each vertex from outside the system.
        '''
        gravity_force = np.array([np.array([0, 0, -0.1])] * self.q.shape[0])
        damping_force = -0.5*self.q1
        return gravity_force + damping_force

    def solve(self, h: float):
        '''
        Return the new q and q1 after a step of size h.
        h = step size
        '''
        # NOTE: All the local solutions happen inside `c.wABp()`.

        # All the selection matrices were ignored using the block-diagonal constructions.
        # We know that constraints are not shared between vertices in our current model.
        wABp = [np.sum([c.wABp(self.q[i]) for c in self.cons[i]], axis=0) if self.cons[i] else np.zeros(System.D)
                for i in range(self.n)]
        wABp = np.concatenate(wABp)

        wAA = [np.sum([c.wAA() for c in self.cons[i]], axis=0) if self.cons[i] else np.zeros((System.D, System.D))
               for i in range(self.n)]
        wAA = la.block_diag(*wAA)

        M_h2 = self.M / (h*h)
        s = self.q.reshape(-1) + h * self.q1.reshape(-1) + \
            la.inv(M_h2) @ self.f_ext().reshape(-1)

        q = la.solve(M_h2 + wAA, M_h2 @ s + wABp).reshape(self.q.shape)
        # pin these vertices
        for v in self.pinned:
            q[v, :] = self.q[v, :]
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
