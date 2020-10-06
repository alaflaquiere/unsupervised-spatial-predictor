import os
import time
import subprocess
import tkinter as tk
from tkinter import filedialog


def get_git_hash():
    try:
        binary_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        hash_ = binary_hash.decode("utf-8")
    except Exception:
        hash_ = "no git commit"
    return hash_


def append_time(d):
    t = time.strftime("%Y-%m-%d--%H-%M-%S",
                      time.localtime())
    d = "-".join((d, t))
    print("experiment name updated to: {}".format(d))
    return d


def check_directory(directory):
    if not os.path.exists(directory):
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askdirectory(initialdir=os.getcwd())
    return directory
