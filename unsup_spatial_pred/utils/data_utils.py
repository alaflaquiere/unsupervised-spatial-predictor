import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_array(x):
    """normalize in [-1, 1] along axis=0"""
    mi, ma = np.min(x, axis=0, keepdims=True), np.max(x, axis=0, keepdims=True)
    return 2 * np.divide(x - mi, ma - mi, out=np.zeros_like(x), where=ma - mi != 0) - 1


def load_normalized_dataset(file, i_env, i_tran, mode):
    subds = sorted([k for k in list(file.keys()) if "env" in k])
    data = dict()
    data["motors_t"] = file[subds[i_env]][mode]["motor_t"][:][i_tran, :]
    data["motors_tp"] = file[subds[i_env]][mode]["motor_tp"][:][i_tran, :]
    data["sensors_t"] = file[subds[i_env]][mode]["sensor_t"][:][i_tran, :]
    data["sensors_tp"] = file[subds[i_env]][mode]["sensor_tp"][:][i_tran, :]
    data["sensors_t"] = data["sensors_t"] / 127.5 - 1
    data["sensors_tp"] = data["sensors_tp"] / 127.5 - 1
    return data


def load_regular_grid(path):
    """TODO: to change
    Loads compressed sensorimotor transitions from
    a npz file created by room-explorer
    """
    fullpath = os.path.join(path, "regular_grid.npz")
    assert os.path.exists(os.path.join(fullpath)), \
        "no regular_grid.npz in the experiment directory"
    print("loading regular grid... [{}]".format(fullpath))
    with np.load(fullpath) as npzfile:
        data = dict(zip(npzfile.files,
                        [npzfile[x] for x in npzfile.files]))
    motor_grid = torch.Tensor(data["motor_grid"])
    return motor_grid, data["state_grid"]


def get_dataloader(dataset, mode, idx_envs, idx_trans,  noise_motor, noise_sensor,
                   batch_size, num_workers, drop_last, shuffle):

    with h5py.File(dataset) as file:
        transitions = dict()
        # combine the datasets
        for i_env, i_tran in zip(idx_envs, idx_trans):
            trans = load_normalized_dataset(file, i_env, i_tran, mode)
            if len(transitions) == 0:
                transitions = trans.copy()
                dim_m = transitions["motors_t"].shape[1]
                dim_s = transitions["sensors_t"].shape[1]
            else:
                for k in transitions.keys():
                    transitions[k] = np.vstack((transitions[k],
                                                trans[k].copy()))
    # add sensorimotor noise
    for key in ["motors_t", "motors_tp"]:
        transitions[key] += noise_motor * np.random.randn(*transitions[key].shape)
    for key in ["sensors_t", "sensors_tp"]:
        transitions[key] += noise_sensor * np.random.randn(*transitions[key].shape)
    # create the dataloader
    dataset = TransitionsDataset(transitions)  # TODO: MOVE THE SAMPLING IN THE TransitionDataset
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        shuffle=shuffle)
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
