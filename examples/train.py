import os
from argparse import ArgumentParser
import yaml
import time
import subprocess
import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog
from tqdm import trange
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from unsup_spatial_pred import SiameseSMPredictor
from unsup_spatial_pred import normalize_array, get_dataloader
from unsup_spatial_pred import save_embedding, start_display_server


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


def check_directory(directory):
    if not os.path.exists(directory):
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askopenfile(initialdir=os.getcwd())
    return directory


def train_epoch(loader, model, loss_fn, optimizer, scheduler):
    """TODO"""
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    model.train()
    epoch_loss = 0.
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = tuple(x.to(device) for x in inputs)
        targets = targets.to(device)
        outputs = model(*inputs)
        loss = loss_fn(outputs, targets)
        epoch_loss += loss.item()
        for param in model.parameters():  # computationally efficient way to do self.optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()
    scheduler.step()
    return epoch_loss / len(loader)


def send_embedding(model, motor_grid, state_grid, model_dir):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        model.eval()
        h_grid = model.get_representation(motor_grid.to(device))
        h_grid = normalize_array(h_grid.cpu().numpy())
        W = model.get_first_weight_vector().detach().cpu().numpy()
        save_embedding(h_grid,
                       state_grid,
                       W,
                       os.path.join(model_dir, "temp_emb.npy"))


def save_model(model, directory, conf):
    """TODO"""
    # todo: move to the network?
    # todo: add a load method to the network?
    torch.save(model.state_dict(),
               os.path.join(directory, "model.pth"))
    with open(os.path.join(directory, "config.yml"), "w") as file:
        yaml.dump(conf, file)


def load_and_save_regular_grid(dataset, experiment):
    # load the regular grid for evaluation
    with h5py.File(dataset, "r") as file:
        print("load and save regular grid")
        motor_grid = file["agent"]["motor_grid"][:]
        state_grid = file["agent"]["state_grid"][:]
        state_grid = normalize_array(state_grid)
    # save the regular grids at the root file of the experiment
    np.savez_compressed(os.path.join(experiment, "regular_grid.npz"),
                        motor_grid=motor_grid,
                        state_grid=state_grid)
    return motor_grid, state_grid


def draw_expl_indexes(n_envs, n_transitions, n_env_per_training):
    idx_envs = np.random.choice(n_envs,
                                n_env_per_training,
                                replace=False)
    idx_trans = np.random.choice(n_transitions,
                                 (n_env_per_training, n_transitions // n_env_per_training),
                                 replace=False)
    idx_trans = np.sort(idx_trans, axis=1)
    return idx_envs, idx_trans


def run(conf):
    """TODO"""
    dataset = conf["files"]["data_directory"]
    experiment = conf["files"]["save_directory"]

    os.makedirs(experiment)

    device = "cuda" if (conf["training"]["gpu"] and torch.cuda.is_available()) else "cpu"

    # read metaparameters
    with h5py.File(dataset, "r") as file:
        n_transitions = file.attrs["dataset.n_transitions"]
        n_envs = file.attrs["dataset.n_runs"]

    # save the regular grids at the root file of the experiment
    motor_grid, state_grid = load_and_save_regular_grid(dataset, experiment)

    for trial in range(conf["training"]["n_trials"]):
        idx_envs, idx_trans = draw_expl_indexes(n_envs, n_transitions,
                                                conf["training"]["n_env_per_training"])
        for mode in ["hopping_base", "static_base", "dynamic_base"]:
            # create the output folder
            model_dir = os.path.join(experiment,
                                     "trial{:03}".format(trial),
                                     mode)
            os.makedirs(model_dir)
            # start display server
            if conf["training"]["display"]:
                display_proc = start_display_server(
                    os.path.join(model_dir, "temp_emb.npy"))
            # get dataloader
            loader, dim_m, dim_s = get_dataloader(dataset, mode, idx_envs, idx_trans,
                                                  **conf["data_loader"])
            # create the network
            conf["network"]["dim_m"] = dim_m
            conf["network"]["dim_s"] = dim_s
            net = SiameseSMPredictor(**conf["network"]).to(device)
            # loss
            loss_fn = nn.MSELoss(reduction="mean")
            # optimizer
            optimizer = Adam(net.parameters(),
                             **conf["optimizer"])
            scheduler = lr_scheduler.ExponentialLR(optimizer,
                                                   1e-2**(1/conf["training"]["n_epochs"]))
            msg = "trial: {}/{}, {}, {} environment(s)".format(trial + 1,
                                                               conf["training"]["n_trials"],
                                                               mode,
                                                               conf["training"]["n_env_per_training"])
            for _ in trange(conf["training"]["n_epochs"], desc=msg):
                _ = train_epoch(loader, net, loss_fn, optimizer, scheduler)
                # visualize the embedding
                if conf["training"]["display"]:
                    send_embedding(net, motor_grid, state_grid, model_dir)
            # save model
            save_model(net, model_dir, conf)
            # kill display server
            if conf["training"]["display"]:
                display_proc.kill()
                os.remove(os.path.join(model_dir, "temp_emb.npy"))


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

    run(config)
