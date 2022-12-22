from constraints.spring import Spring
from constraints.strain import Discrete
import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import constants as const
from typing import List, Tuple


class System:
    # dimensions of the geometric space
    D = 3

    def __init__(self,
                 q: npt.NDArray,
                 q1: npt.NDArray,
                 M: npt.NDArray):
        '''
         A system of discrete points, their states and their constraints.

        Inputs
        ------
        q := initial positions matrix
        q1 := initial first derivatives of q (i.e. velocity) matrix
        M := mass matrix.
            This matrix must be diagonal of the form R^{D*n} where:
            - D := dimensions of the problem. This D in practice is the global variable "D" included in the "constants" module.
                    Typically this would have the value of three ( D = 3) , as 3D environments are taken into consideration.
            - n := number of particles in the mass-spring system
        cons := list of constraints. TODO: Transform this into a 3D array/matrix (one selection array for each constraint)
        pinned := set of vertices (referred to by indices) to be pinned in position.
                    Equivalently, the ids included in the "pinned" set will not be subjected to any constraint (no spring force will be applied to them).
        '''
        self.n = q.shape[0]  # the object variable "n" denotes the number of particles in the mass-spring system.

        self.q = q
        # In the case of no initial velocity conditions from the client code,
        # the particles are considered to start from a rest state.
        if q1 is None:
            q1 = np.zeros(q.shape)
        self.q1 = q1
        self.M = M
        # self.cons[i, j] :=
        # i : id of point
        # j : the constraint id (to which point i is subjected)
        # This data structure (list of lists) keeps track of which particle (rows) is subject to which constraint (columns).
        # Initially, no particle (#particles = n) is constrained.
        self.cons = []
        self.pinned = set()

        # sanity checks
        assert q.shape == (self.n, const.D)
        assert q1.shape == (self.n, const.D)
        assert M.shape == (self.n * const.D, self.n * const.D)

        self.lhs_lu_pivot = None

    def add_spring(self, k: float, L: float, indices: List[int]):
        '''
        Construct and add a spring constraint to the system with the given parameters.
        The indices refer to the vertices already present in the system.
        '''
        assert len(indices) == 2
        for i in indices:
            assert 0 <= i < self.n
        assert indices[0] != indices[1]
        c = Spring(k, L, q=lambda: [(self.q[i], i) for i in indices])
        self.cons.append(c)

    def add_discrete_strain(
            self,
            ref: npt.NDArray,
            indices: List[int],
            sigrange: Tuple[int]):
        '''
        Construct and add a spring constraint to the system with the given parameters.
        The indices refer to the vertices already present in the system.
        '''
        assert len(indices) == 3
        assert indices[0] != indices[1]
        assert indices[0] != indices[2]
        assert indices[1] != indices[2]
        c = Discrete(ref=ref, q=lambda: [(self.q[i], i)
                                         for i in indices], sigrange=sigrange)
        self.cons.append(c)

    def f_ext(self):
        '''
        Returns external forces applied to each vertex from outside the system.
        TODO: forces need to be parameterized instead of hardcoded.
        '''
        gravity_force = np.array(
            [const.gravity_acceleration] * self.q.shape[0])
        gravity_force = (self.M @ gravity_force.reshape(-1)
                         ).reshape(self.q.shape)
        damping_force = -0.5 * self.q1
        return gravity_force

    def wAtA(self):
        wAtA = np.zeros((self.n, const.D, const.D))
        for c in self.cons:
            c_wAtA = c.w * c.A.T @ c.A
            for (q, idx) in c.q():
                wAtA[idx, :] += c_wAtA
        # a giant matrix for all the points arranged in a block diagonal fashion,
        # since the different points' computations are independent (unless it
        # goes through a constraint).
        return la.block_diag(*wAtA)

    def wAtBp(self):
        """
        This function refers to this expression in the Global Solve (equation 10).
        In mathematical notation, the following return value would be written as:
        Σ ωi Ai' Bi pi, ∀i in constraints
        The entries of the list "self.cons" belong in one of the two category-values:
        I) Empty list ( [] ): in this case, the i-th particle is not of interest (energy) optimization problem; it has no constraints acting upon it.
        II) Non-empty list. The i-th particle has at least one constraint acting upon it.
        The characteristics of the each constraint (distance metrics A and B, important weight w, and auxiliary variable pi)
        can be extracted explicitly (c.w, c.A, c.B) or implicitly (c.project() callable) by the "Constraint" object.
        TODO Maybe change the variable signature from 'i' to something else, as it may be misleading when compared to the notation
        TODO used in the paper. In the context of the paper, 'i' refers to the constraint index, not to the particle associated with it.

        Returns
        -------
        "numpy.array" object whose whose elements are the (3D) projections of each point.
        output[0] -> projection of point with id = 0 (vector)
        output[1] -> projection of point with id = 1 (vector)
        ...
        """
        wAtBp = np.zeros((self.n, const.D))
        for c in self.cons:
            wAtB = c.w * c.A.T @ c.B
            for (p, idx) in c.project():
                wAtBp[idx, :] += wAtB @ p

        # a giant stack of all the points' column vectors.
        return np.array(wAtBp)

    def solve(self, h: float):
        '''
        Return the new q and q1 after a step of size h.


        Parameters
        ----------
        h := step size of the simulation.

        Returns
        -------
        tuple (q, q1)
        q := updated positions of the points of the system
        q1 := updated velocities of the points the system
        '''
        # NOTE: All the local solutions happen inside `c.project()`.

        # * IMPORTANT NOTE:
        # * i) A linear system is written in the form Ax = y, where dim(A) = m x m, dim(x) = m x 1, dim(y) = m x 1
        # * ii) The mass matrix M is inserted as the Kronecker product (block diagonal matrix) of the particles' masses.
        # * In order to conform to form i) provided matrix M, all 3D information of the system (variables q, constants sn, wA'Bp)
        # * needs to be manipulated ("reshaped" in numpy terms) accordingly; n x 3 form is changed to 3n x 1, where n = #particles.

        # All the selection matrices were ignored using the block-diagonal constructions.
        # We know that constraints are not shared between vertices in our
        # current model.
        M_h2 = self.M / (h * h)
        wAtBp = self.wAtBp()
        # s_n = q_n + h v_n + h^2 M^{−1} fext
        s = self.q + h * self.q1 + \
            (la.inv(M_h2) @ self.f_ext().reshape(-1)).reshape(self.q.shape)

        if self.lhs_lu_pivot is None:
            wAtA = self.wAtA()

            # Writing the more readable form below in case we decided to utilize it. Needless to say, we would have to store M^{-1}
            # Software-engineering-wise, the current implementation is better, as the same value (M / h^2) is used in lhs and rhs as well.
            # s = self.q + h * self.q1 + \
            # (h**2 * la.inv(M) @ self.f_ext().reshape(-1)).reshape(self.q.shape)

            lhs = M_h2 + wAtA  # w A' A is already in matrix form (3n x 3n)
            self.lhs_lu_pivot = la.lu_factor(lhs)
        # M/h^2 (3n x 3n) . s (n x 3) --reshape-->  M/h^2 (3n x 3n) . s (3n x
        # 1) --result--> 3n x 1
        rhs = M_h2 @ s.reshape(-1) + wAtBp.reshape(-1)

        # After solving the linear system, revert the dimensions of the output
        # into n x 3.
        q = la.lu_solve(self.lhs_lu_pivot, rhs, overwrite_b=True,
                        check_finite=False).reshape(self.q.shape)
        # pin these vertices
        q[list(self.pinned), :] = self.q[list(self.pinned), :]
        # After computing the next position of the particles, we may trivially acquire the velocities of the particle in the next time step.
        # Apply the first equation of implicit Euler integration: q'_{t} = q_{t
        # - q_{t-1}} / h <=> q_{t} = q_{t-1} + h q'_{t}
        q1 = (q - self.q) / h
        return (q, q1)

    def step(self, h: float):
        self.q, self.q1 = self.solve(h)


if __name__ == '__main__':
    # all the vertices in the system
    q = np.array([[0, 0, 0], [2.5, 0, 0]])

    # Constructing a mass-spring system with unit masses (Mii = 1, β€ i \in {0, ..., n-1})
    # For this particular example,     [ 1 0 0 0 0 0 ] 1st particle X
    #                                  [ 0 1 0 0 0 0 ] 1st particle Y
    #                            M =   [ 0 0 1 0 0 0 ] 1st particle Z
    #                                  [ 0 0 0 1 0 0 ] 2nd particle X
    #                                  [ 0 0 0 0 1 0 ] 2nd particle Y
    #                                  [ 0 0 0 0 0 1 ] 2nd particle Z

    s = System(q=q, q1=None, M=np.kron(
        np.identity(q.shape[0]), np.identity(const.D)))
    # A spring force is applied to the particle with id = 1, relevant to the
    # position of the particle with id = 0.
    s.add_spring(k=1, L=1, indices=[0, 1])
    # Note: As there is no constraint on particle with id = 0, it is implied
    # that particle with id = 0 is fixed.

    for i in range(100):
        s.step(0.01)
        print(f"{i}: p={s.q[1]} v={s.q1[1]}")
