import numpy as np
import numpy.typing as npt
import constants as con


import openmesh as om

import system as msys
from enum import Enum


def make_a_two_point_system():
    q = np.array([[0, 0, 0], [0.25, 0, 0]])
    masses_per_point = np.array([10, 1]) # Masses per point. 0th position has the mass of the 0th particle, 1st positions has the mass of the 1st particle etc.
    M = np.kron(np.diag(masses_per_point), np.identity(con.D)) #? Wouldn't it be a better idea to refer to the dimensionality of the problem from a global variable (defined in the "constants.py" module), instead of keeping it inside the "System"?
    s = msys.System(q=q, q1=None, M=M)
    # Keeping the old implementation here as well in case of disagreement.
    # s = msys.System(q=q, q1=None, M=np.kron(``
    #     np.diagflat(m), np.identity(msys.System.D)))
    # the heavy one will be pinned at origin.
    # m = np.array([10, 1])
    # s = msys.System(q=q, q1=None, M=np.kron(
    #     np.diagflat(m), np.identity(msys.System.D)))
    # the heavy one will be pinned at origin.
    s.pinned.add(0)
    # the light one will be tied to the heavy one with a spring.
    s.add_spring(k=1, L=1, q_idx=1, p0_idx=0)
    return s


def make_a_three_point_system():
    q = np.array([[0, 0, 0], [0.25, 0, 0], [2.5, 2.5, 0]])
    m = np.array([10, 1, 1])
    s = msys.System(q=q, q1=None, M=np.kron(
        np.diagflat(m), np.identity(msys.System.D)))
    # the heavy one will be pinned at origin.
    s.pinned.add(0)
    # the light ones will be tied to the other two with springs.
    s.add_spring(k=1, L=1, q_idx=1, p0_idx=0)
    s.add_spring(k=1, L=1, q_idx=1, p0_idx=2)
    s.add_spring(k=1, L=1, q_idx=2, p0_idx=0)
    s.add_spring(k=1, L=1, q_idx=2, p0_idx=1)
    return s


class GridDiagonalDirection(Enum):
    TOPLEFT = 1
    TOPRIGHT = 2


def make_a_grid_system(
        diagtype: GridDiagonalDirection = GridDiagonalDirection.TOPLEFT):
    diagtype = GridDiagonalDirection(diagtype)
    # a grid system with N * N points.
    N = 10
    # create the points
    L = 4
    M = 1
    x, y, z = np.meshgrid(np.linspace(-L / 2, L / 2, num=N),
                          np.linspace(-L / 2, L / 2, num=N), [0])
    q = np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).T
    m = np.ones(q.shape[0]) * M / (N * N)
    s = msys.System(q=q, q1=None, M=np.kron(
        np.diagflat(m), np.identity(msys.System.D)))

    def vtxid(i, j): return i * N + j

    def axis_aligned_neighbors(i, j): return [(
        i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]

    def topleft_diagonal_neighbors(i, j): return [
        (i - 1, j - 1), (i + 1, j + 1)]

    def topright_diagonal_neighbors(i, j): return [
        (i - 1, j + 1), (i + 1, j - 1)]

    def neighbors(i, j):
        return axis_aligned_neighbors(
            i,
            j) + (
            topleft_diagonal_neighbors(
                i,
                j) if diagtype is GridDiagonalDirection.TOPLEFT else topright_diagonal_neighbors(
                i,
                j))

    # add the constraints by the grid lines
    for i in range(N):
        for j in range(N):
            for (ni, nj) in neighbors(i, j):
                if not (ni >= 0 and ni < N and nj >= 0 and nj < N):
                    # out of bounds
                    continue
                s.add_spring(k=1, L=L / (N - 1), q_idx=vtxid(i, j),
                             p0_idx=vtxid(ni, nj))
    s.pinned.add(vtxid(N - 1, N - 1))
    s.pinned.add(vtxid(N - 1, 0))
    s.pinned.add(vtxid(0, N - 1))
    return s


def make_triangle_mesh_system(
        filename: str,
        point_mass: float,
        spring_stiffnes: float):
    # Read the mesh from a the file
    mesh = om.read_trimesh(filename)

    q = mesh.points()  # The Vx3 configuration matrix
    V = mesh.points().shape[0]  # V
    m = np.ones(q.shape[0]) / V * point_mass  # Vx1 mass per point matrix
    M = np.kron(np.diagflat(m), np.identity(msys.System.D))  # VxV mass matrix

    # Create our system
    s = msys.System(q=q, q1=None, M=M)

    # Add a spring for each edge
    for [i, j] in mesh.edge_vertex_indices():
        # Edge information
        u = mesh.points()[i]  # Source vertex
        v = mesh.points()[j]  # Target vertex

        # Spring parameters
        k = spring_stiffnes  # Spring stiffness
        L = np.linalg.norm(v - u)  # Spring length

        # Add our spring
        s.add_spring(k=k, L=L, q_idx=i, p0_idx=j)
        s.add_spring(k=k, L=L, q_idx=j, p0_idx=i)

    # Manually pin the first vertex
    s.pinned.add(0)

    # Return the system
    return s
