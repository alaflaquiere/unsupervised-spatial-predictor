import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_array(x):
    """normalize in [-1, 1] along axis=0"""
    mi, ma = np.min(x, axis=0, keepdims=True), np.max(x, axis=0, keepdims=True)
    return 2 * np.divide(x - mi, ma - mi, out=np.zeros_like(x), where=ma - mi != 0) - 1


def get_dataset_subfolders(directory):
    dir_datasets = sorted(
        glob.glob(os.path.join(directory, "run*"))
    )
    print("{} datasets found in [{}] ".format(len(dir_datasets),
                                              directory))
    return dir_datasets


def load_normalized_dataset(path):
    """
    Loads compressed sensorimotor transitions from
    a npz file created by room-explorer
    """
    print("loading data... [{}]".format(path))
    with np.load(path) as npzfile:
        data = dict(zip(npzfile.files,
                        [npzfile[x] for x in npzfile.files]))
    data["sensors_t"] = data["sensors_t"] / 127.5 - 1
    data["sensors_tp"] = data["sensors_tp"] / 127.5 - 1
    return data


def load_regular_grid(path):
    """
    Loads compressed sensorimotor transitions from
    a npz file created by room-explorer
    """
    path = os.path.join(path, "data_regular_grid.npz")
    print("loading regular grid... [{}]".format(path))
    with np.load(path) as npzfile:
        data = dict(zip(npzfile.files,
                        [npzfile[x] for x in npzfile.files]))
    motor_grid = torch.Tensor(data["motor_grid"])
    return motor_grid, data["state_grid"]


def get_dataloader(dataset_path, conf):
    # load transitions
    transitions = load_normalized_dataset(dataset_path)
    dim_m = transitions["motors_t"].shape[1]
    dim_s = transitions["sensors_t"].shape[1]
    # add sensorimotor noise
    for key in ["motors_t", "motors_tp"]:
        transitions[key] += conf["training"]["noise_motor"] \
                            * np.random.randn(*transitions[key].shape)
    for key in ["sensors_t", "sensors_tp"]:
        transitions[key] += conf["training"]["sensor_noise"] \
                            * np.random.randn(*transitions[key].shape)
    # create the dataloader
    dataset = TransitionsDataset(transitions)
    loader = DataLoader(dataset,
                        **conf["data_loader"])
    return loader, dim_m, dim_s


class TransitionsDataset(Dataset):
    """TODO"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["motors_t"].shape[0]

    def __getitem__(self, idx):
        m_t = torch.Tensor(self.data["motors_t"][idx])
        m_tp = torch.Tensor(self.data["motors_tp"][idx])
        s_t = torch.Tensor(self.data["sensors_t"][idx])
        s_tp = torch.Tensor(self.data["sensors_tp"][idx])
        return (m_t, m_tp, s_t), s_tp
