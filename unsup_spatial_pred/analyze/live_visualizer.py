import os
import sys
import numpy as np
import platform
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from _pickle import UnpicklingError
plt.ion()


def save_embedding(h, state, w, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    dic = {"h_grid": h,
           "state_grid": state,
           "W": w}
    np.save(path, dic, allow_pickle=True)


def start_display_server(path):
    root = os.path.realpath(__file__)
    if platform.system() == 'Windows':
        command = "python {} {}".format(root, path)
        proc = subprocess.Popen(command)
    elif platform.system() == 'Linux':
        command = "exec python3 {} {}".format(root, path)
        proc = subprocess.Popen([command], shell=True)
    else:
        proc = None
    return proc


def center_and_scale(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    center = np.mean(x, axis=0)
    scale = 0.5 * np.max(np.max(x, axis=0) - np.min(x, axis=0))
    return (x - center) / scale


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class LiveEmbeddingVisualizer:
    def __init__(self, path):
        self.path = path
        self.save_name = os.path.join(os.path.dirname(self.path),
                                      "embedding.png")
        self._run()

    def _display(self, h, state, w, plot_quiver=False):
        if not plt.fignum_exists(0):
            self.fig = plt.figure(num=0, figsize=(6, 6))
            self.ax = plt.subplot(111, projection='3d')
        self.ax.cla()
        h = center_and_scale(h)
        self.ax.plot(h[:, 0],
                     h[:, 1],
                     h[:, 2],
                     'r.')
        self.ax.plot(state[:, 0],
                     state[:, 1],
                     state[:, 2],
                     'k.',
                     alpha=1)
        if plot_quiver:
            self.ax.quiver(np.zeros(w.shape[0]),
                           np.zeros(w.shape[0]),
                           np.zeros(w.shape[0]),
                           w[:, 0],
                           w[:, 1],
                           w[:, 2],
                           color="b",
                           alpha=0.5)
        self.ax.set_xlabel('$h_1$')
        self.ax.set_ylabel('$h_2$')
        self.ax.set_zlabel('$h_3$')
        set_axes_equal(self.ax)  # make the axis scales equal
        plt.show(block=False)
        self.fig.savefig(self.save_name)

    def _run(self):
        moddate = 0
        while True:
            try:
                stamp = os.stat(self.path).st_mtime
                if stamp != moddate:
                    moddate = stamp
                    try:
                        dic = np.load(self.path, allow_pickle=True)[()]  # array of 0 dimensions
                        self._display(dic["h_grid"],
                                      dic["state_grid"],
                                      dic["W"])
                    except (OSError, UnpicklingError):
                        pass
            except FileNotFoundError:
                pass
            finally:
                plt.pause(0.2)


if __name__ == '__main__':
    d = LiveEmbeddingVisualizer(sys.argv[1])
