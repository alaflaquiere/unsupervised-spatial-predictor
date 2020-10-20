import os
import time
import subprocess
from argparse import ArgumentParser
import yaml
import tkinter as tk
from tkinter import filedialog
from unsup_spatial_pred import run_experiment


def check_directory(directory):
    if not os.path.exists(directory):
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askopenfile(initialdir=os.getcwd())
    return directory


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
    print("experiment name: {}".format(d))
    return d


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="path to config file",
                        default="config/config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    config["git_commit"] = get_git_hash()

    config["files"]["data_directory"] = check_directory(
        config["files"]["data_directory"])

    if os.path.exists(config["files"]["save_directory"]):
        config["files"]["save_directory"] = append_time(
            config["files"]["save_directory"])

    run_experiment(config)
