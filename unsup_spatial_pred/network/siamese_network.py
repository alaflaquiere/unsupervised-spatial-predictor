import os
import yaml
import torch
import torch.nn as nn


class SiameseSMPredictor(nn.Module):
    def __init__(self, dim_m, dim_h, dim_s, activation):
        super().__init__()
        assert activation in ["selu", "relu"]
        self.dim_h = dim_h
        if activation == "selu":
            activ = nn.SELU
        elif activation == "relu":
            activ = nn.ReLU
        else:
            activ = None
        self.m_encoder = nn.Sequential(
            nn.Linear(dim_m, 150),
            activ(),
            nn.Linear(150, 100),
            activ(),
            nn.Linear(100, 50),
            activ(),
            nn.Linear(50, dim_h)
        )
        self.s_predictor = nn.Sequential(
            nn.Linear(2 * dim_h + dim_s, 200),
            activ(),
            nn.Linear(200, 150),
            activ(),
            nn.Linear(150, 100),
            activ(),
            nn.Linear(100, dim_s),
        )

    def forward(self, m_t, m_tp, s_t):
        h_t = self.m_encoder(m_t)
        h_tp = self.m_encoder(m_tp)
        hhs = torch.cat((h_t, h_tp, s_t), dim=1)
        s_pred = self.s_predictor(hhs)
        return s_pred

    def get_representation(self, m):
        return self.m_encoder(m)

    def get_first_linear_projection(self, m):
        h = self.m_encoder(m)
        W = self.s_predictor[0].weight[:, self.dim_h:2 * self.dim_h]
        b = self.s_predictor[0].bias
        return torch.mm(h, W.T) + b

    def get_first_weight_vector(self):
        return self.s_predictor[0].weight[:, self.dim_h:2 * self.dim_h]

    def save(self, path, conf):
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.exists(os.path.join(path, "model.pth")):
            print("'{}' already exists; enter a new directory to save the network".format(
                os.path.join(path, "model.pth")))
            path = input()
        pth_path = os.path.join(path, "model.pth")
        yml_path = os.path.join(path, "config.yml")
        torch.save(self.state_dict(), pth_path)
        with open(yml_path, "w") as file:
            yaml.dump(conf, file)

    def load(self, path):
        while not os.path.join(path, "model.pth"):
            print("'{}' does not exist; enter a new directory to load from".format(path))
            path = input()
        self.load_state_dict(
            torch.load(
                os.path.join(path, "model.pth")
            )
        )
