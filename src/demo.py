import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mani
import system as msys
import demo_systems as demos
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def updater(f, g, s, h, dt):
    for _ in range(int(np.ceil(dt/h))):
        s.step(h)
    g._offsets3d = (s.q[:, 0], s.q[:, 1], s.q[:, 2])


def play_system(s):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    g = ax.scatter(s.q[:, 0], s.q[:, 1], s.q[:, 2], s=5*np.diag(s.M)[::3])

    def comp_line_segs(): return np.array([[q, c.p0()[0]]
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
    s = demos.make_a_grid_system()
    play_system(s)
