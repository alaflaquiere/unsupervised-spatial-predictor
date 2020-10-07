import os
import sys
import numpy as np
import platform
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    if platform.system() == 'Windows':
        command = "python analyze\\display_embedding.py {}".format(path)
        proc = subprocess.Popen(command)
    elif platform.system() == 'Linux':
        command = "exec python3 analyze\\display_embedding.py {}".format(path)
        proc = subprocess.Popen([command], shell=True)
    else:
        proc = None
    return proc


class DisplayEmbedding:
    """TODO"""
    def __init__(self, path):
        self.path = path
        self.save_name = os.path.join(os.path.dirname(self.path),
                                      "embedding.png")
        self._run()

    def _display(self, h, state, w):
        if not plt.fignum_exists(0):
            self.fig = plt.figure(num=0, figsize=(6, 6))
            self.ax = plt.subplot(111, projection='3d')
        self.ax.cla()
        self.ax.plot(h[:, 0],
                     h[:, 1],
                     h[:, 2],
                     'r.')
        self.ax.plot(state[:, 0],
                     state[:, 1],
                     state[:, 2],
                     'ko')
        self.ax.quiver(np.zeros(w.shape[0]),
                       np.zeros(w.shape[0]),
                       np.zeros(w.shape[0]),
                       w[:, 0],
                       w[:, 1],
                       w[:, 2],
                       color="b")
        self.ax.set_xlabel('$h_1$')
        self.ax.set_ylabel('$h_2$')
        self.ax.set_zlabel('$h_3$')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        plt.show(block=False)
        self.fig.savefig(self.save_name)

    def _run(self):
        # TODO: greater speed by checking when the file is changed: https://mygisblog.wordpress.com/2014/08/03/monitoring-if-a-file-has-changed-in-python/
        while True:
            if os.path.exists(self.path):
                try:
                    dic = np.load(self.path, allow_pickle=True)[()]  # array of 0 dimensions
                    self._display(dic["h_grid"],
                                  dic["state_grid"],
                                  dic["W"])
                finally:
                    plt.pause(5)
            else:
                plt.pause(1)


if __name__ == '__main__':
    d = DisplayEmbedding(sys.argv[1])
