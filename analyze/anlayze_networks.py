import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import linear_model
from scipy.spatial.distance import pdist
from utils.data_utils import normalize_array
from network.siamese_network import SiameseSMPredictor
from utils.data_utils import load_regular_grid


class Evaluator:
    """Docstring"""
    def __init__(self, path):
        """Constructor"""
        self.path = path
        # collect all the hopping_base models
        search_str = os.path.join(self.path, "exp*", "hopping_base", "trial*")
        self.dir_hop = sorted(glob.glob(search_str))
        # collect all the static_base models
        search_str = os.path.join(self.path, "exp*", "static_base", "trial*")
        self.dir_sta = sorted(glob.glob(search_str))
        # collect all the dynamic_base models
        search_str = os.path.join(self.path, "exp*", "dynamic_base", "trial*")
        self.dir_dyn = sorted(glob.glob(search_str))
        # compile the results
        self.results_hop = self.evaluate_all(self.dir_hop)
        self.results_sta = self.evaluate_all(self.dir_sta)
        self.results_dyn = self.evaluate_all(self.dir_dyn)

    @staticmethod
    def compute_dissimilarities(state, h, topo_weight=10):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if h.ndim == 1:
            h = h.reshape(1, -1)

        reg = linear_model.LinearRegression(fit_intercept=True)
        reg.fit(state, h)
        states_up_projection = reg.predict(state)

        distances_h = pdist(h)
        distances_state = pdist(states_up_projection)

        normalized_diffs = np.abs(distances_h - distances_state) / distances_h.max()
        weighting = np.exp(-topo_weight * distances_h / distances_h.max())
        metric_error = np.mean(normalized_diffs)
        topo_error = np.mean(normalized_diffs * weighting)

        # TODO: instead of going with a linear regression that minimizes
        #  the distance between points and their projection, one could
        #  train a linear model to minimize the dissimilarities directly!

        return metric_error, topo_error

    def evaluate(self, model, m_grid, sta_grid):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        # get the embedding
        with torch.no_grad():
            model.eval()
            h_grid = model.get_representation(m_grid.to(device))
            h_grid = normalize_array(h_grid.detach().cpu().numpy())
        # compute dissimilarities
        metric_error, topo_error = self.compute_dissimilarities(sta_grid, h_grid)
        # compute singular values of the embedding
        _, sv_h, _ = np.linalg.svd(h_grid)
        # compute singular values of the first linear layer after the embedding
        W = model.get_first_weight_vector()
        _, sv_w, _ = np.linalg.svd(W.detach().cpu().numpy())
        return metric_error, topo_error, sv_h, sv_w

    def evaluate_all(self, directories):
        results = {"metric_errors": [],
                   "topo_errors": [],
                   "sv_hs": [],
                   "sv_ws": []}
        for d in directories:
            # load the network
            with open(os.path.join(d, "config.yml"), "r") as f:
                config = yaml.load(f, yaml.FullLoader)
            device = "cuda" if (config["training"]["gpu"] and torch.cuda.is_available()) else "cpu"
            net = SiameseSMPredictor(**config["network"]).to(device)
            net.load_state_dict(
                torch.load(
                    os.path.join(d, "model.pth")
                )
            )
            # load the regular grid for evaluation
            motor_grid, state_grid = load_regular_grid(d)
            # compute metrics
            metric_error, topo_error, sv_h, sv_w = self.evaluate(net, motor_grid, state_grid)
            results["metric_errors"].append(metric_error)
            results["topo_errors"].append(topo_error)
            results["sv_hs"].append(sv_h)
            results["sv_ws"].append(sv_w)
        return results

    def display_results(self):
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 4, 1)
        ax.violinplot(self.results_dyn["metric_errors"], positions=[1])
        ax.violinplot(self.results_sta["metric_errors"], positions=[2])
        ax.violinplot(self.results_hop["metric_errors"], positions=[3])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["dynamic_base", "static_base", "hopping_base"])
        ax.set_title("metric_errors")

        ax = plt.subplot(1, 4, 2)
        ax.violinplot(self.results_dyn["topo_errors"], positions=[1])
        ax.violinplot(self.results_sta["topo_errors"], positions=[2])
        ax.violinplot(self.results_hop["topo_errors"], positions=[3])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["dynamic_base", "static_base", "hopping_bas"])
        ax.set_title("topo_errors")

        ax = plt.subplot(1, 4, 3)
        data = np.array(self.results_dyn["sv_hs"])
        ax.violinplot(data, positions=[0, 1, 2])
        data = np.array(self.results_sta["sv_hs"])
        ax.violinplot(data, positions=[5, 6, 7])
        data = np.array(self.results_hop["sv_hs"])
        ax.violinplot(data, positions=[10, 11, 12])
        ax.set_xticks([1, 6, 11])
        ax.set_xticklabels(["dynamic_base", "static_base", "hopping_base"])
        ax.set_title("embedding singular values")

        ax = plt.subplot(1, 4, 4)
        data = np.array(self.results_dyn["sv_ws"])
        ax.violinplot(data, positions=[0, 1, 2])
        data = np.array(self.results_sta["sv_ws"])
        ax.violinplot(data, positions=[5, 6, 7])
        data = np.array(self.results_hop["sv_ws"])
        ax.violinplot(data, positions=[10, 11, 12])
        ax.set_xticks([1, 6, 11])
        ax.set_xticklabels(["dynamic_base", "static_base", "hopping_base"])
        ax.set_title("W singular values")

        plt.show(block=False)
        return fig
