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

    g = ax.scatter(s.q[:, 0], s.q[:, 1], s.q[:, 2], s=5 * np.diag(s.M)[::3])

    def comp_line_segs(): return np.array([[q, c.p0()[0]]
                                           for (q, cs) in zip(s.q, s.cons)
                                           for c in cs])
    ls = comp_line_segs()
    lc = Line3DCollection(ls)
    ax.add_collection(lc)

    h = 0.01
    fps = 30
    ival = int(np.floor(1000 / fps))

    def cb(f):
        updater(f, g, s, h, ival / 1000)
        ls[:] = comp_line_segs()[:]
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

    s = demos.make_triangle_mesh_system(
        meshfile, 1, 10) if meshfile is not None else demos.make_a_grid_system(
        diagtype=demos.GridDiagonalDirection.TOPLEFT)
    play_system(s, save=saveto, nframes=nframes)
