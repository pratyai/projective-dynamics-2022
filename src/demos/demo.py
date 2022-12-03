'''
To watch:
$ python3 -m demos.demo

To save:
$ python3 -m demos.demo -w ../demos/mpl-demo.mp4
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mani
import system as msys
from . import demo_systems as demos
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys
import getopt
import os


def line_segs(c):
    '''
    Compute line segments to draw for constraint `c`.
    There maybe different kinds of line segments that, for example, we want to draw with different
    colours. So, just return a dictionary of lists, keys being the "group's name".
    '''
    n = len(c.q())
    q = np.array([qi for (qi, idx) in c.q()])
    p = np.array([pi for (pi, idx) in c.project()])
    # print(f'q = {q}\np = {p}\nref = {c.ref}')
    # exit(0)
    return {
        'current': [[q[i, :], q[j, :]] for i in range(n) for j in range(i+1, len(q))],
        'project': [[p[i, :], p[j, :]] for i in range(len(p)) for j in range(i+1, len(p))],
    }


def updater(f, g, s, h, dt):
    for _ in range(int(np.ceil(dt / h))):
        s.step(h)
    g._offsets3d = (s.q[:, 0], s.q[:, 1], s.q[:, 2])


def play_system(s, save=None, nframes=900):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

    g = ax.scatter(s.q[:, 0], s.q[:, 1], s.q[:, 2], s=5 * np.diag(s.M)[::3])

    ls = {}
    for c in s.cons:
        for (k, v) in line_segs(c).items():
            if k not in ls:
                ls[k] = []
            ls[k] += v

    COLORS = {'current': 'blue', 'project': 'red'}
    LINEWIDTHS = {'current': 1.5, 'project': .5}
    ZORDER = {'current': 2, 'project': 2.5}
    for (k, v) in ls.items():
        lc = Line3DCollection(v,
                              colors=COLORS[k],
                              linewidths=LINEWIDTHS[k],
                              zorder=ZORDER[k],
                              )
        ax.add_collection(lc)

    h = 0.01
    fps = 30
    ival = int(np.floor(1000 / fps))

    def cb(f):
        updater(f, g, s, h, ival / 1000)
        nls = {}
        for c in s.cons:
            for (k, v) in line_segs(c).items():
                if k not in nls:
                    nls[k] = []
                nls[k] += v
        for k in ls.keys():
            ls[k][:] = nls[k]
    ani = mani.FuncAnimation(
        fig, cb, frames=nframes, interval=ival)
    w = mani.FFMpegWriter(fps=30, bitrate=1800)
    if save is not None:
        ani.save(save, writer=w)
    else:
        plt.show()


if __name__ == '__main__':
    argv = sys.argv[1:]
    saveto = None
    nframes = 900
    meshfile = None
    try:
        opts, args = getopt.getopt(argv, 'w:n:m:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == '-w':
            saveto = a
        elif o == '-n':
            nframes = int(a)
        elif o == '-m':
            meshfile = a
    if saveto is not None:
        print(f'Will save {nframes} frams of the animation to {saveto}.')
        os.makedirs(os.path.dirname(saveto), exist_ok=True)

    # s = demos.make_a_three_point_strain_system()
    s = demos.make_a_strain_grid_system()
    '''
    s = demos.make_triangle_mesh_system(
        meshfile, 1, 10) if meshfile is not None else demos.make_a_grid_system(
        diagtype=demos.GridDiagonalDirection.TOPLEFT)
    '''
    play_system(s, save=saveto, nframes=nframes)
