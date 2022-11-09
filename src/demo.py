import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mani
import system as msys
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def updater(f, g, s, h, dt):
    for _ in range(int(np.ceil(dt/h))):
        s.step(h)
    g._offsets3d = (s.q[:, 0], s.q[:, 1], s.q[:, 2])


def three_point_system():
    # a system with 3 points.
    q = np.array([[0, 0, 0], [0.25, 0, 0], [2.5, 2.5, 0]])
    m = np.array([10, 1, 1])
    s = msys.System(q=q, q1=None, M=np.kron(
        np.diagflat(m), np.identity(msys.System.D)))
    s.add_spring(k=1, L=1, q_idx=0, p0_idx=1)
    s.add_spring(k=1, L=1, q_idx=0, p0_idx=2)
    s.add_spring(k=1, L=1, q_idx=1, p0_idx=0)
    s.add_spring(k=1, L=1, q_idx=1, p0_idx=2)
    s.add_spring(k=1, L=1, q_idx=2, p0_idx=0)
    s.add_spring(k=1, L=1, q_idx=2, p0_idx=1)
    s.pinned.add(0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    g = ax.scatter(s.q[:, 0], s.q[:, 1], s.q[:, 2], s=5*m)

    def comp_line_segs(): return np.array([[q, c.p0()]
                                           for (q, cs) in zip(s.q, s.cons)
                                           for c in cs])
    ls = comp_line_segs()
    lc = Line3DCollection(ls)
    ax.add_collection(lc)

    h = 0.01
    fps = 30
    ival = int(np.floor(1000/fps))

    def cb(f):
        updater(f, g, s, h, ival/1000)
        ls[:] = comp_line_segs()[:]
    ani = mani.FuncAnimation(fig, cb, interval=ival)
    plt.show()


def grid_system():
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

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    g = ax.scatter(s.q[:, 0], s.q[:, 1], s.q[:, 2])

    def comp_line_segs(): return np.array([[q, c.p0()]
                                           for (q, cs) in zip(s.q, s.cons)
                                           for c in cs])
    ls = comp_line_segs()
    lc = Line3DCollection(ls)
    ax.add_collection(lc)

    h = 0.01
    fps = 30
    ival = int(np.floor(1000/fps))

    def cb(f):
        updater(f, g, s, h, ival/1000)
        ls[:] = comp_line_segs()[:]
    ani = mani.FuncAnimation(fig, cb, interval=ival)
    plt.show()


if __name__ == '__main__':
    grid_system()
    # three_point_system()
