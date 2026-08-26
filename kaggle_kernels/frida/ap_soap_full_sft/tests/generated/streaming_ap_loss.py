"""Exact copy of StreamingAPLoss, extracted from frida_ap_soap_full_sft.py.
Regenerate with generate_extracted_sources.py -- do not hand-edit.
"""
import torch
import torch.nn as nn


AP_GAMMA = 0.9
AP_SIGMOID_SCALE = 2.0
AP_EPS = 1e-12


class StreamingAPLoss(nn.Module):
    """LibAUC-style stochastic AP gradient surrogate with persistent FP32 EMAs."""
    def __init__(self, n_examples, gamma=AP_GAMMA, scale=AP_SIGMOID_SCALE, eps=AP_EPS):
        super().__init__()
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.eps = float(eps)
        self.register_buffer("u_all", torch.zeros(n_examples, dtype=torch.float32))
        self.register_buffer("u_pos", torch.zeros(n_examples, dtype=torch.float32))

    def forward(self, scores, labels, row_indices, anchor_positions):
        scores = scores.float()
        labels = labels.float()
        if scores.dtype != torch.float32 or self.u_all.dtype != torch.float32:
            raise RuntimeError("AP scores and state must be FP32")
        anchor_scores = scores.index_select(0, anchor_positions)
        anchor_rows = row_indices.index_select(0, anchor_positions).long()
        pairwise = torch.sigmoid(
            self.scale * (scores.unsqueeze(0) - anchor_scores.unsqueeze(1))
        ).float()
        current_all = pairwise.mean(dim=1)
        current_pos = (pairwise * labels.unsqueeze(0)).mean(dim=1)
        updated_all = self.gamma * self.u_all[anchor_rows] + (1.0 - self.gamma) * current_all
        updated_pos = self.gamma * self.u_pos[anchor_rows] + (1.0 - self.gamma) * current_pos
        with torch.no_grad():
            self.u_all.index_copy_(0, anchor_rows, updated_all.detach())
            self.u_pos.index_copy_(0, anchor_rows, updated_pos.detach())
        denominator = updated_all.square().clamp_min(self.eps)
        coefficient = (
            updated_pos.unsqueeze(1) - updated_all.unsqueeze(1) * labels.unsqueeze(0)
        ) / denominator.unsqueeze(1)
        return (coefficient.detach() * pairwise).mean()

    @torch.no_grad()
    def reset_(self):
        self.u_all.zero_()
        self.u_pos.zero_()
