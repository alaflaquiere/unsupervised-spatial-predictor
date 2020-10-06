import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseSMPredictor(nn.Module):
    """TODO"""
    def __init__(self, dim_m, dim_h, dim_s, activation):
        super().__init__()
        assert activation in ["selu", "relu"]
        if activation is "selu":
            activ = nn.SELU
        elif activation is "relu":
            activ = nn.ReLU
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
