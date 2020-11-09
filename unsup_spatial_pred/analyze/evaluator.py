import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn import linear_model
from scipy.spatial.distance import pdist
from unsup_spatial_pred import SiameseSMPredictor
from unsup_spatial_pred import load_regular_grid


def load_model(path):
    with open(os.path.join(path, "config.yml"), "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    device = "cuda" if (config["training"]["gpu"] and torch.cuda.is_available()) else "cpu"
    net = SiameseSMPredictor(**config["network"]).to(device)
    net.load(path)
    return net


class Evaluator:
    """Docstring"""
    def __init__(self, path):
        """Constructor"""
        assert os.path.exists(path), "incorrect experiment path"
        self.path = path
        self.num_trials = len(glob.glob(os.path.join(self.path, "trial*")))
        self.modes = ["dynamic_base", "static_base", "hopping_base"]
        # load the regular grid for evaluation of the models
        self.m_grid, self.x_grid = load_regular_grid(path)
        self.embeddings = dict()
        self.singular_values = dict()
        self.dissimilarities = dict()
        self.dissimilarity_profiles = dict()

    def _get_projection_in_embedding(self, h, x=None):
        x = self.x_grid if x is None else x
        reg = linear_model.LinearRegression(fit_intercept=True)
        reg.fit(x, h)
        x_grid_projection = reg.predict(x)
        return x_grid_projection

    def _get_embedding(self, trial, mode):
        path = os.path.join(self.path, "trial{:03}".format(trial), mode)
        # load the model
        net = load_model(path)
        device = "cuda" if next(net.parameters()).is_cuda else "cpu"
        # generate the embedding
        with torch.no_grad():
            net.eval()
            for k in np.arange(0, self.m_grid.shape[0], 256):  # in case we're sending too much to the gpu at once
                idx = np.arange(k, min(k + 256, self.m_grid.shape[0]))
                h_tensor = net.get_representation(self.m_grid[idx, :].to(device))
                if k == 0:
                    h_grid = h_tensor.detach().cpu().numpy()
                else:
                    h_grid = np.vstack((h_grid, h_tensor.detach().cpu().numpy()))
        return h_grid

    def _get_dissimilarity(self, h, weight=0):
        # TODO: instead of going with a linear regression that minimizes
        #  the distance between points and their projection, one could
        #  train a linear model to minimize the dissimilarities directly!
        # project the states in the embedding space
        x_projection = self._get_projection_in_embedding(h)
        # compute pairwise distances
        distances_h = pdist(h)
        distances_x = pdist(x_projection)
        # compute dissimilarity
        normalized_diffs = np.abs(distances_h - distances_x) / np.max(distances_h)
        weighting = np.exp(-weight * distances_x / np.max(distances_x))
        error = np.mean(normalized_diffs * weighting)
        return error

    def _get_dissimilarity_profile(self, h, weight_max=10):
        # project the states in the embedding space
        x_projection = self._get_projection_in_embedding(h)
        # compute pairwise distances
        distances_h = pdist(h)
        distances_x = pdist(x_projection)
        # compute dissimilarity
        normalized_diffs = np.abs(distances_h - distances_x) / np.max(distances_h)
        errors = []
        weights = np.linspace(0, weight_max, 10)
        for weight in weights:
            weighting = np.exp(-weight * distances_x / np.max(distances_x))
            errors.append(np.mean(normalized_diffs * weighting))
        return weights, errors

    def _get_embedding_sv(self, trial, mode):
        if len(self.embeddings) == 0:
            print("generate the embeddings first with Evaluator.compute_embeddings")
            return
        _, sv, _ = np.linalg.svd(self.embeddings[(trial, mode)])
        return sv

    def _get_first_linear_layer_sv(self, trial, mode):
        path = os.path.join(self.path, "trial{:03}".format(trial), mode)
        # load the model
        net = load_model(path)
        W = net.get_first_weight_vector()
        _, sv, _ = np.linalg.svd(W.detach().cpu().numpy())
        return sv

    def compute_embeddings(self):
        print("getting embeddings...")
        for trial in range(self.num_trials):
            for mode in self.modes:
                self.embeddings[(trial, mode)] = self._get_embedding(trial, mode)

    def compute_singular_values(self):
        if len(self.embeddings) == 0:
            self.compute_embeddings()
        print("getting singular values...")
        for trial in range(self.num_trials):
            for mode in self.modes:
                self.singular_values[(trial, mode, "embedding")] = self._get_embedding_sv(trial, mode)
                self.singular_values[(trial, mode, "first_layer")] = self._get_first_linear_layer_sv(trial, mode)

    def compute_dissimilarities(self):
        if len(self.embeddings) == 0:
            self.compute_embeddings()
        print("computing dissimilarities...")
        for trial in range(self.num_trials):
            for mode in self.modes:
                self.dissimilarities[(trial, mode, "topological")] =\
                    self._get_dissimilarity(self.embeddings[(trial, mode)], weight=10)
                self.dissimilarities[(trial, mode, "metric")] = \
                    self._get_dissimilarity(self.embeddings[(trial, mode)], weight=0)

    def compute_dissimilarities_profiles(self):
        if len(self.embeddings) == 0:
            self.compute_embeddings()
        print("computing dissimilarities profiles...")
        for trial in range(self.num_trials):
            for mode in self.modes:
                weights, self.dissimilarity_profiles[(trial, mode, "topological")] =\
                    self._get_dissimilarity_profile(self.embeddings[(trial, mode)], weight_max=10)
        self.dissimilarity_profiles["weights"] = weights

    def plot_experiment_stats(self):
        if len(self.dissimilarities) == 0:
            self.compute_dissimilarities()
        if len(self.singular_values) == 0:
            self.compute_singular_values()

        mode_positions = {mode: i for i, mode in enumerate(self.modes)}

        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 4, 1)
        for mode in self.modes:
            res = [self.dissimilarities[(trial, mode, "topological")] for trial in range(self.num_trials)]
            ax.violinplot(res, positions=[mode_positions[mode]])
        ax.set_xticks(list(mode_positions.values()))
        ax.set_xticklabels(self.modes)
        ax.set_title("topological dissimilarity")

        ax = plt.subplot(1, 4, 2)
        for mode in self.modes:
            res = [self.dissimilarities[(trial, mode, "metric")] for trial in range(self.num_trials)]
            ax.violinplot(res, positions=[mode_positions[mode]])
        ax.set_xticks(list(mode_positions.values()))
        ax.set_xticklabels(self.modes)
        ax.set_title("metric dissimilarity")

        ax = plt.subplot(1, 4, 3)
        for mode in self.modes:
            res = np.array([self.singular_values[(trial, mode, "embedding")] for trial in range(self.num_trials)])
            shifts = np.arange(-res.shape[1]//2, res.shape[1]//2) + 1
            positions = (res.shape[1] + 1) * mode_positions[mode] + shifts
            ax.violinplot(res, positions=positions)
        ax.set_xticks([(res.shape[1] + 1) * p for p in mode_positions.values()])
        ax.set_xticklabels(self.modes)
        ax.set_title("embedding singular values")

        ax = plt.subplot(1, 4, 4)
        for mode in self.modes:
            res = np.array([self.singular_values[(trial, mode, "first_layer")] for trial in range(self.num_trials)])
            shifts = np.arange(-res.shape[1] // 2, res.shape[1] // 2) + 1
            positions = (res.shape[1] + 1) * mode_positions[mode] + shifts
            ax.violinplot(res, positions=positions)
        ax.set_xticks([(res.shape[1] + 1) * p for p in mode_positions.values()])
        ax.set_xticklabels(self.modes)
        ax.set_title("first W singular values")

        plt.show(block=True)
        return fig

    def plot_dissimilarity_profile(self, trial=0):
        if len(self.embeddings) == 0:
            self.compute_embeddings()
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        for mode in self.modes:
            weights, values = self._get_dissimilarity_profile(self.embeddings[(trial, mode)])
            ax.plot(weights, values, label=mode)
            ax.set_xlim(weights[-1], weights[0])
            ax.set_xlabel("weighting")
            ax.set_ylabel("dissimilarity")
            ax.set_title("trial {}".format(trial, mode))
            plt.legend()
        plt.show(block=True)

    def plot_dissimilarity_profiles(self):
        if len(self.dissimilarity_profiles) == 0:
            self.compute_dissimilarities_profiles()
        plt.figure(figsize=(7, 7))
        ax = plt.subplot(1, 1, 1)
        weights = self.dissimilarity_profiles["weights"]
        for mode in self.modes:
            res = [self.dissimilarity_profiles[(trial, mode, "topological")] for trial in range(self.num_trials)]
            mean_res = np.mean(np.array(res), axis=0)
            std_red = np.std(np.array(res), axis=0)
            ax.plot(weights, mean_res, label=mode)
            ax.fill_between(weights, mean_res - std_red, mean_res + std_red, alpha=0.4)
        ax.set_xlim(weights[-1], weights[0])
        ax.set_title("dissimilarity profiles")
        ax.set_xlabel("weighting")
        ax.legend(loc="upper left")
        plt.show(block=True)

    def plot_pairwise_distances_comparison(self, trial=0, num=10000):
        if len(self.embeddings) == 0:
            self.compute_embeddings()
        index = np.random.choice(self.x_grid.shape[0] * (self.x_grid.shape[0] - 1) // 2,
                                 num,
                                 replace=False)  # limit the memory cost
        mode_subplot = {mode: i for i, mode in enumerate(self.modes)}
        fig = plt.figure(figsize=(12, 4))
        for mode in self.modes:
            emb = self.embeddings[(trial, mode)]
            distances_emb = pdist(emb)[index]
            distances_x = pdist(self.x_grid)[index]
            ax = fig.add_subplot(1, 3, mode_subplot[mode] + 1)
            ax.plot(distances_x, distances_emb, 'b.')
            ax.set_xlabel("ground-truth distances")
            ax.set_ylabel("embedded distances")
            ax.set_title("trial {}, mode {}".format(trial, mode))
        plt.show(block=True)
        return fig

    def plot_embedding(self, trial=0, rays=False):
        if len(self.embeddings) == 0:
            self.compute_embeddings()
        mode_subplot = {mode: i for i, mode in enumerate(self.modes)}
        fig = plt.figure(figsize=(12, 4))
        for mode in self.modes:
            emb = self.embeddings[(trial, mode)]
            x_projection = self._get_projection_in_embedding(emb)
            ax = fig.add_subplot(1, 3, mode_subplot[mode] + 1, projection="3d")
            ax.plot(x_projection[:, 0],
                    x_projection[:, 1],
                    x_projection[:, 2],
                    "ko")
            if rays:
                for x_proj, em in zip(x_projection, emb):
                    ax.plot([x_proj[0], em[0]],
                            [x_proj[1], em[1]],
                            [x_proj[2], em[2]],
                            "b")
            ax.plot(emb[:, 0],
                    emb[:, 1],
                    emb[:, 2],
                    'r.', alpha=0.4)
            ax.set_title(mode)
        plt.show(block=True)
