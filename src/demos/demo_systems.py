import numpy as np
import numpy.typing as npt
import constants as con
import helpers as hlp


# import openmesh as om

import system as msys
from enum import Enum, auto


def make_a_two_point_system():
    q = np.array([[0, 0, 0], [0.25, 0, 0]])
    # Masses per point. 0th position has the mass of the 0th particle, 1st
    # positions has the mass of the 1st particle etc.
    masses_per_point = np.array([10, 1])
    # ? Wouldn't it be a better idea to refer to the dimensionality of the problem from a global variable (defined in the "constants.py" module), instead of keeping it inside the "System"?
    M = np.kron(np.diag(masses_per_point), np.identity(con.D))
    s = msys.System(q=q, q1=None, M=M)
    # Keeping the old implementation here as well in case of disagreement.
    # s = msys.System(q=q, q1=None, M=np.kron(``
    #     np.diagflat(m), np.identity(con.D)))
    # the heavy one will be pinned at origin.
    # m = np.array([10, 1])
    # s = msys.System(q=q, q1=None, M=np.kron(
    #     np.diagflat(m), np.identity(con.D)))
    # the heavy one will be pinned at origin.
    s.pinned.add(0)
    # the light one will be tied to the heavy one with a spring.
    s.add_spring(k=1, L=1, indices=[0, 1])
    return s


def make_a_three_point_system():
    q = np.array([[0, 0, 0], [0.25, 0, 0], [2.5, 2.5, 0]])
    m = np.array([10, 1, 1])
    s = msys.System(q=q, q1=None, M=np.kron(
        np.diagflat(m), np.identity(con.D)))
    # the heavy one will be pinned at origin.
    s.pinned.add(0)
    # the light ones will be tied to the other two with springs.
    # id0 -- id1
    #     \      \
    #        \     \
    #           \    \
    #             \   \
    #               id2
    # No force is applied on the particle with id = 0 (a.k.a. it is fixed).
    s.add_spring(k=1, L=1, indices=[0, 1])
    s.add_spring(k=1, L=1, indices=[1, 2])
    s.add_spring(k=1, L=1, indices=[0, 2])
    return s


class GridDiagonalDirection(Enum):
    TOPLEFT = auto()
    TOPRIGHT = auto()


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
        np.diagflat(m), np.identity(con.D)))

    def vtxid(i, j): return i * N + j

    def axis_aligned_neighbors(i, j):
        # Return: Up, left, down, right #? Is the convention "i+1 = i + one
        # down" and "j+1 = j + one on the right" correct? The same question
        # applies for the functions below.
        return [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]

    def topleft_diagonal_neighbors(i, j):
        # Return: diagonal up-left, diagonal down-right
        return [(i - 1, j - 1), (i + 1, j + 1)]

    def topright_diagonal_neighbors(i, j):
        # Return: diagonal up-right, diagonal down-left
        return [(i - 1, j + 1), (i + 1, j - 1)]

    def neighbors(i, j):
        return axis_aligned_neighbors(i, j) + (topleft_diagonal_neighbors(i, j)
                                               if diagtype is GridDiagonalDirection.TOPLEFT
                                               else topright_diagonal_neighbors(i, j))

    # add the constraints by the grid lines
    for i in range(N):
        for j in range(N):
            for (ni, nj) in neighbors(i, j):
                if not (ni >= 0 and ni < N and nj >= 0 and nj < N):
                    # out of bounds
                    continue
                s.add_spring(k=1, L=L / (N - 1),
                             indices=[vtxid(i, j), vtxid(ni, nj)])
    s.pinned.add(vtxid(N - 1, N - 1))
    s.pinned.add(vtxid(N - 1, 0))
    s.pinned.add(vtxid(0, N - 1))
    return s


def make_triangle_mesh_system(
        mesh: None,
        point_mass: float,
        spring_stiffness: float):

    q = mesh.points()  # The Vx3 configuration matrix
    V = mesh.points().shape[0]  # V
    m = np.ones(q.shape[0]) / V * point_mass  # Vx1 mass per point matrix
    M = np.kron(np.diagflat(m), np.identity(con.D))  # VxV mass matrix

    # Create our system
    s = msys.System(q=q, q1=None, M=M)

    # Add a spring for each edge
    for [i, j] in mesh.edge_vertex_indices():
        # Edge information
        u = mesh.points()[i]  # Source vertex
        v = mesh.points()[j]  # Target vertex

        # Spring parameters
        k = spring_stiffness  # Spring stiffness
        L = np.linalg.norm(v - u)  # Spring length

        # Add springs between every two vertices.
        s.add_spring(k=k, L=L, indices=[i, j])

    # Return the system
    return s


def make_a_three_point_strain_system():
    q = np.array([[0, 0, 0], [0.25, 0, 0], [2.5, 2.5, 0]])
    M = hlp.mass_matrix_fem_trimesh(q, indices=np.array([[0, 1, 2]]))
    M = np.kron(M, np.identity(con.D))
    s = msys.System(q=q, q1=None, M=M)
    s.pinned.add(0)
    s.add_discrete_strain(ref=q, indices=[0, 1, 2])
    return s


def make_a_strain_grid_system():
    # a grid system with N * N points.
    N = 10
    L = 4

    # create the points
    x, y, z = np.meshgrid(np.linspace(-L / 2, L / 2, num=N),
                          np.linspace(-L / 2, L / 2, num=N), [0])
    q = np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).T

    # create the elements
    def vtxid(i, j): return i * N + j

    def neighbors(i, j): return [
        (i - 1, j), (i, j + 1),
        (i + 1, j + 1), (i + 1, j),
        (i, j - 1), (i - 1, j - 1),
    ]

    def good(i, j): return (i >= 0 and i < N and j >= 0 and j < N)

    tri = []
    for i in range(N):
        for j in range(N):
            ns = neighbors(i, j)
            for k1 in range(len(ns)):
                k2 = (k1 + 1) % len(ns)
                if not good(*ns[k1]) or not good(*ns[k2]):
                    continue
                tri.append([vtxid(i, j), vtxid(*ns[k1]), vtxid(*ns[k2])])
    tri = np.array(tri)

    # create the mass matrix
    M = hlp.mass_matrix_fem_trimesh(q, indices=tri)
    M = np.kron(M, np.identity(con.D))

    # create the system
    s = msys.System(q=q, q1=None, M=M)
    s.pinned.add(vtxid(N - 1, N - 1))
    s.pinned.add(vtxid(N - 1, 0))
    s.pinned.add(vtxid(0, N - 1))
    for t in tri:
        s.add_discrete_strain(ref=q[t, :], indices=t)
    return s
