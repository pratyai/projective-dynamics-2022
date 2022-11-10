import numpy as np
import system as msys


def make_a_two_point_system():
    q = np.array([[0, 0, 0], [0.25, 0, 0]])
    m = np.array([10, 1])
    s = msys.System(q=q, q1=None, M=np.kron(
        np.diagflat(m), np.identity(msys.System.D)))
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


def make_a_grid_system():
    # a grid system with N * N points.
    N = 10
    # create the points
    L = 4
    M = 1
    x, y, z = np.meshgrid(np.linspace(-L/2, L/2, num=N),
                          np.linspace(-L/2, L/2, num=N), [0])
    q = np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).T
    m = np.ones(q.shape[0]) * M / (N*N)
    s = msys.System(q=q, q1=None, M=np.kron(
        np.diagflat(m), np.identity(msys.System.D)))
    # add the constraints by the grid lines
    for i in range(N):
        for j in range(N):
            if i == 0 and j == 0:
                # s.pinned.add(i*N+j)
                continue
            elif i == N-1 and j == N-1:
                s.pinned.add(i*N+j)
                continue
            elif i == 0 and j == N-1:
                s.pinned.add(i*N+j)
                continue
            elif i == N-1 and j == 0:
                s.pinned.add(i*N+j)
                continue
            if i+1 < N:
                s.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=(i+1)*N+j)
            if i-1 >= 0:
                s.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=(i-1)*N+j)
            if j+1 < N:
                s.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=i*N+j+1)
            if j-1 >= 0:
                s.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=i*N+j-1)
    return s
