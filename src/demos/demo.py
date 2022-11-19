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
from matplotlib.gridspec import GridSpec


def updater(f, g, s, h, dt):
    for _ in range(int(np.ceil(dt / h))):
        s.step(h)
    g._offsets3d = (s.q[:, 0], s.q[:, 1], s.q[:, 2])


def play_system(s, save=None, nframes=900):
    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(2, 1, width_ratios=[1], height_ratios=[1, 3])

    plotax = fig.add_subplot(gs[0, 0])
    plotax.set_xlim(-3, 3)
    plotax.set_ylim(-3, 3)
    plotax.set_xlabel('X')
    plotax.set_ylabel('Y')

    simax = fig.add_subplot(gs[1, 0], projection='3d')
    simax.set_xlim3d(-3, 3)
    simax.set_ylim3d(-3, 3)
    simax.set_zlim3d(-3, 3)
    simax.set_xlabel('X')
    simax.set_ylabel('Y')
    simax.set_zlabel('Z')
    g = simax.scatter(s.q[:, 0], s.q[:, 1], s.q[:, 2], s=5 * np.diag(s.M)[::3])

    def comp_line_segs(): return np.array([[q, c.p0()[0]]
                                           for (q, cs) in zip(s.q, s.cons)
                                           for c in cs])
    ls = comp_line_segs()
    lc = Line3DCollection(ls)
    simax.add_collection(lc)

    h = 0.01
    fps = 30
    ival = int(np.floor(1000 / fps))

    def simcb(f):
        updater(f, g, s, h, ival / 1000)
        ls[:] = comp_line_segs()[:]

    frames = []
    energies = {}

    def plotcb(f):
        frames.append(f)
        for (k, v) in s.energy().items():
            if k not in energies:
                energies[k] = []
            energies[k].append(v)
        plotax.clear()
        for (k, v) in energies.items():
            plotax.plot(frames, v, label=k, linewidth=1)
        plotax.legend(fontsize='xx-small', loc='lower center', ncol=1)

    def cb(f):
        simcb(f)
        plotcb(f)

    ani = mani.FuncAnimation(
        fig, cb, frames=nframes, interval=ival)
    plt.tight_layout()
    if save is not None:
        w = mani.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(save, writer=w)
    else:
        plt.show()


if __name__ == '__main__':
    argv = sys.argv[1:]
    saveto = None
    nframes = 900
    try:
        opts, args = getopt.getopt(argv, 'w:n:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == '-w':
            saveto = a
        elif o == '-n':
            nframes = int(a)
    if saveto is not None:
        print(f'Will save {nframes} frams of the animation to {saveto}.')
        os.makedirs(os.path.dirname(saveto), exist_ok=True)

    s = demos.make_a_grid_system(diagtype=0)
    play_system(s, save=saveto, nframes=nframes)
