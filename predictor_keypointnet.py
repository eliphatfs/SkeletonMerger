# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:48:13 2020

@author: eliphat
"""
import torch
import merger.merger_net as merger_net
import json
import tqdm
import numpy as np
import argparse

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='../keypointnet/annotations/chair.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='../keypointnet/pcd',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='merger.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-p', '--prediction-output', type=str, default='merger_prediction.npz',
                        help='Output file where prediction results are written.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()

net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    return pc


kpn_ds = json.load(open(ns.annotation_json))
out_kpcd = []
out_nfact = []
for i in tqdm.tqdm(range(0, len(kpn_ds), ns.batch), unit_scale=ns.batch):
    Q = []
    for j in range(ns.batch):
        if i + j >= len(kpn_ds):
            continue
        entry = kpn_ds[i + j]
        cid = entry['class_id']
        mid = entry['model_id']
        pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid))
        pcmax = pc.max()
        pcmin = pc.min()
        pcn = (pc - pcmin) / (pcmax - pcmin)
        pcn = 2.0 * (pcn - 0.5)
        Q.append(pcn)
        out_nfact.append([pcmax, pcmin])
    with torch.no_grad():
        recon, key_points, kpa, emb, null_activation = net(torch.Tensor(np.array(Q)).to(ns.device))
    for kp in key_points:
        out_kpcd.append(kp)
for i in range(len(out_kpcd)):
    out_kpcd[i] = out_kpcd[i].cpu().numpy()
np.savez(ns.prediction_output, kpcd=out_kpcd, nfact=out_nfact)
