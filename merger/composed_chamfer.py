# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:33:57 2021

@author: eliphat
"""
import torch


torch.square = lambda x: x ** 2
torch.minimum = lambda x, y: torch.min(torch.stack((x, y)), dim=0)[0]


def composed_sqrt_chamfer(y_true, y_preds, activations):
    L = 0.0
    # activations: N x P where P: # sub-clouds
    # y_true: N x ? x 3
    # y_pred: P x N x ? x 3 
    part_backs = []
    for i, y_pred in enumerate(y_preds):
        # y_true: k1 x 3
        # y_pred: k2 x 3
        y_true_rep = torch.unsqueeze(y_true, axis=-2)  # k1 x 1 x 3
        y_pred_rep = torch.unsqueeze(y_pred, axis=-3)  # 1 x k2 x 3
        # k1 x k2 x 3
        y_delta = torch.sqrt(1e-4 + torch.sum(torch.square(y_pred_rep - y_true_rep), -1))
        # k1 x k2
        y_nearest = torch.min(y_delta, -2)[0]
        # k2
        part_backs.append(torch.min(y_delta, -1)[0])
        L = L + torch.mean(torch.mean(y_nearest, -1) * activations[:, i]) / len(y_preds)
    part_back_stacked = torch.stack(part_backs)  # P x N x k1
    sorted_parts, indices = torch.sort(part_back_stacked, dim=0)
    weights = torch.ones_like(sorted_parts[0])  # N x k1
    for i in range(len(y_preds)):
        w = torch.minimum(weights, torch.gather(activations, -1, indices[i]))
        L = L + torch.mean(sorted_parts[i] * w)
        weights = weights - w
    L = L + torch.mean(weights * 20.0)
    return L
