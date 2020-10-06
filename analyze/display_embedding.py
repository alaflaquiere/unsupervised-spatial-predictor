import os
import sys
import numpy as np
import platform
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()


def save_embedding(h, state, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    dic = {"h_grid": h,
           "state_grid": state}
    np.save(path, dic, allow_pickle=True)


def start_display_server(path):
    if platform.system() == 'Windows':
        command = "python analyze\\display_embedding.py {}".format(path)
        proc = subprocess.Popen(command)
    elif platform.system() == 'Linux':
        command = "exec python3 analyze\\display_embedding.py {}".format(path)
        proc = subprocess.Popen([command], shell=True)
    return proc


class DisplayEmbedding:
    """TODO"""
    def __init__(self, path):
        self.path = path
        self.fig = plt.figure(num=0, figsize=(6, 6))
        self.ax = plt.subplot(111, projection='3d')
        self._run()

    def _display(self, h, state):
        self.ax.cla()
        self.ax.plot(h[:, 0],
                     h[:, 1],
                     h[:, 2],
                     'r.')
        self.ax.set_xlabel('$h_1$')
        self.ax.set_ylabel('$h_2$')
        self.ax.set_zlabel('$h_3$')
        self.ax.plot(state[:, 0],
                     state[:, 1],
                     state[:, 2],
                     'ko')

    def _run(self):
        while True:
            if os.path.exists(self.path):
                try:
                    dic = np.load(self.path, allow_pickle=True)[()]  # array of 0 dimensions
                    self._display(dic["h_grid"],
                                  dic["state_grid"])
                finally:
                    plt.show(block=False)
                    plt.pause(5)


if __name__ == '__main__':
    d = DisplayEmbedding(sys.argv[1])
