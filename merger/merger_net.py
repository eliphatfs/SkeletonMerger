# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:50:30 2020

@author: eliphat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnetpp.pointnet2_sem_seg_msg import get_model as PointNetPP


torch.square = lambda x: x ** 2


class PBlock(nn.Module):  # MLP Block
    def __init__(self, iu, *units, should_perm):
        super().__init__()
        self.sublayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.should_perm = should_perm
        ux = iu
        for uy in units:
            self.sublayers.append(nn.Linear(ux, uy))
            self.batch_norms.append(nn.BatchNorm1d(uy))
            ux = uy

    def forward(self, input_x):
        x = input_x
        for sublayer, batch_norm in zip(self.sublayers, self.batch_norms):
            x = sublayer(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = batch_norm(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = F.relu(x)
        return x


class Head(nn.Module):  # Decoder unit, one per line
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((200, 3)) * 0.002)

    def forward(self, KPA, KPB):
        dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(KPA - KPB), dim=-1))))
        count = min(200, max(15, int((dist / 0.01).item())))
        device = dist.device
        self.f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
        self.b_interp = 1.0 - self.f_interp
        # KPA: N x 3, KPB: N x 3
        # Interpolated: N x count x 3
        K = KPA.unsqueeze(-2) * self.f_interp + KPB.unsqueeze(-2) * self.b_interp
        R = self.emb[:count, :].unsqueeze(0) + K  # N x count x 3
        return R.reshape((-1, count, 3)), self.emb


class Net(nn.Module):  # Skeleton Merger structure
    def __init__(self, npt, k):
        super().__init__()
        self.npt = npt
        self.k = k
        self.PTW = PointNetPP(k)
        self.PT_L = nn.Linear(k, k)
        self.MA_EMB = nn.Parameter(torch.randn([k * (k - 1) // 2]))
        self.MA = PBlock(1024, 512, 256, should_perm=False)
        self.MA_L = nn.Linear(256, k * (k - 1) // 2)
        self.DEC = nn.ModuleList()
        for i in range(k):
            DECN = nn.ModuleList()
            for j in range(i):
                DECN.append(Head())
            self.DEC.append(DECN)

    def forward(self, input_x):
        APP_PT = torch.cat([input_x, input_x, input_x], -1)
        KP, GF = self.PTW(APP_PT.permute(0, 2, 1))
        KPL = self.PT_L(KP)
        KPA = F.softmax(KPL.permute(0, 2, 1), -1)  # [n, k, npt]
        KPCD = KPA.bmm(input_x)  # [n, k, 3]
        RP = []
        L = []
        for i in range(self.k):
            for j in range(i):
                R, EM = self.DEC[i][j](KPCD[:, i, :], KPCD[:, j, :])
                RP.append(R)
                L.append(EM)
        GFP = F.max_pool1d(GF, 16).squeeze()
        MA = F.sigmoid(self.MA_L(self.MA(GFP)))
        # MA = torch.sigmoid(self.MA_EMB).expand(input_x.shape[0], -1)
        LF = torch.cat(L, dim=1)  # P x 72 x 3
        return RP, KPCD, KPA, LF, MA
